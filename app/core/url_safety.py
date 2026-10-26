"""SSRF protection for URLs that users give us and the server later fetches.

SSRF (server-side request forgery): if a user can make the server fetch any URL, they can point
it at things only the server can reach, such as `http://169.254.169.254/` (cloud metadata),
`http://db/` (an internal service) or `http://localhost:6379/`.

IMPORTANT: validating a URL when it is saved is NOT enough. The check can be outdated by the
time the URL is fetched, because:
  * DNS can change: a hostname that resolved to a public address at creation time can later
    resolve to a private one (DNS rebinding), and even between our check and the connection;
  * a public URL can redirect to a private one (every redirect hop is a new URL);
  * an attacker can change what a feed server answers at any moment.
So the fetcher must call `validate_public_url` again before EVERY request and before following
EVERY redirect. Pinning the connection to the validated addresses closes the remaining gap
between check and connect; that hardening is planned for Phase 7.

Policy: only `http`/`https`, no credentials, only the scheme's standard port, only a real
fully-qualified hostname (no IP addresses in any notation), and every address the name resolves
to must be public. The DNS resolver is injectable so tests never touch the network.
"""

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.core.exceptions import DomainError

# A resolver turns a hostname into IP address strings (IPv4 and/or IPv6).
Resolver = Callable[[str], Awaitable[list[str]]]

MAX_URL_LENGTH = 2048
DNS_TIMEOUT_SECONDS = 5.0
STANDARD_PORTS = {"http": 80, "https": 443}
# Names that only make sense inside a private network.
INTERNAL_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain")

_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
# A last label like "1", "0x7f" or "2130706433" makes parsers read the whole host as an IPv4
# address (decimal, hex and octal forms), so it is treated as an IP literal and refused.
_NUMERIC_LABEL = re.compile(r"^(0x[0-9a-f]*|[0-9]+)$")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")

_NOT_PUBLIC = "The URL's host does not resolve to a public address"


class UnsafeUrlError(DomainError):
    status_code = 422
    code = "unsafe_url"


@dataclass(frozen=True)
class ValidatedUrl:
    url: str
    host: str
    addresses: tuple[str, ...]  # what the host resolved to at validation time


async def system_resolver(host: str) -> list[str]:
    """Resolve with the operating system's resolver (a thread, so the event loop isn't blocked)."""
    loop = asyncio.get_running_loop()
    infos = await asyncio.wait_for(
        loop.getaddrinfo(host, None, type=socket.SOCK_STREAM), DNS_TIMEOUT_SECONDS
    )
    return sorted({str(info[4][0]).split("%")[0] for info in infos})  # drop IPv6 "%zone" suffix


async def validate_public_url(url: str, resolver: Resolver = system_resolver) -> ValidatedUrl:
    """Return the URL if it is safe to fetch right now, or raise UnsafeUrlError."""
    url = url.strip()
    host = check_url_syntax(url)  # no network: obviously bad URLs never reach DNS
    return ValidatedUrl(url, host, await _resolve_public(host, resolver))


def check_url_syntax(url: str) -> str:
    """Check everything knowable without DNS and return the lowercase hostname."""
    if not url or len(url) > MAX_URL_LENGTH:
        raise UnsafeUrlError("URL is empty or too long")
    # Whitespace, control characters and backslashes are how parsers get tricked into reading
    # a URL differently from the HTTP client that will use it later.
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 or ch == "\\" for ch in url):
        raise UnsafeUrlError("URL contains illegal characters")
    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError for a malformed or out-of-range port
    except ValueError as error:
        raise UnsafeUrlError("URL is malformed") from error

    if parts.scheme not in STANDARD_PORTS:
        raise UnsafeUrlError("URL must use http or https")
    if "@" in parts.netloc:
        raise UnsafeUrlError("URL must not contain credentials")
    if port is not None and port != STANDARD_PORTS[parts.scheme]:
        raise UnsafeUrlError("URL must use the standard port for its scheme")

    host = (parts.hostname or "").removesuffix(".")  # "localhost." is the same host as "localhost"
    _check_hostname(host)
    return host


def _check_hostname(host: str) -> None:
    if not host:
        raise UnsafeUrlError("URL has no host")
    if not host.isascii():
        raise UnsafeUrlError("Hostnames must be ASCII (use punycode for international names)")
    if ":" in host or _NUMERIC_LABEL.match(host.rsplit(".", 1)[-1]):
        raise UnsafeUrlError("IP addresses are not allowed; use a hostname")
    if "." not in host:  # "localhost", "db", "redis", "intranet": names that only resolve locally
        raise UnsafeUrlError("Host must be a fully qualified domain name")
    if len(host) > 253 or not all(_LABEL.match(label) for label in host.split(".")):
        raise UnsafeUrlError("Host is not a valid domain name")
    if host.endswith(INTERNAL_SUFFIXES):
        raise UnsafeUrlError("Internal hostnames are not allowed")


async def _resolve_public(host: str, resolver: Resolver) -> tuple[str, ...]:
    try:
        addresses = await resolver(host)
    except OSError as error:  # DNS failure or timeout; same message as "private", see below
        raise UnsafeUrlError(_NOT_PUBLIC) from error
    # ANY non-public address rejects the host: a name with one public and one private record
    # would otherwise be a coin flip for the attacker. The message is the same for "does not
    # exist" and "is private", so it cannot be used to probe which internal names exist.
    if not addresses or not all(is_public_address(address) for address in addresses):
        raise UnsafeUrlError(_NOT_PUBLIC)
    return tuple(addresses)


def is_public_address(address: str) -> bool:
    """True only for globally routable unicast addresses."""
    try:
        ip = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return False
    # IPv6 addresses can wrap an IPv4 one (::ffff:127.0.0.1, 6to4, Teredo, NAT64). Judge the
    # wrapped address, otherwise ::ffff:10.0.0.5 would sneak a private IPv4 past the checks.
    embedded = _embedded_ipv4(ip)
    return all(_is_public(inner) for inner in embedded) if embedded else _is_public(ip)


def _embedded_ipv4(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> list[ipaddress.IPv4Address]:
    if not isinstance(ip, ipaddress.IPv6Address):
        return []
    if ip.ipv4_mapped:
        return [ip.ipv4_mapped]
    if ip.sixtofour:
        return [ip.sixtofour]
    if ip.teredo:
        return list(ip.teredo)  # (server, client): both must be public
    if ip in _NAT64:
        return [ipaddress.IPv4Address(ip.packed[-4:])]
    return []


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # `is_global` alone is not enough: for IPv4 it still says True for multicast addresses.
    return ip.is_global and not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )
