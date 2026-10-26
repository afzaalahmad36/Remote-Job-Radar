import pytest

from app.core.url_safety import (
    UnsafeUrlError,
    ValidatedUrl,
    check_url_syntax,
    is_public_address,
    system_resolver,
    validate_public_url,
)
from tests.fakes import FakeResolver

# URLs that are rejected from their text alone. The resolver must never even be asked.
REJECTED_WITHOUT_DNS = {
    # --- localhost and names that only resolve inside a private network
    "localhost": "http://localhost/feed",
    "localhost-with-port": "http://localhost:8000/feed",
    "localhost-trailing-dot": "http://LOCALHOST./feed",
    "localhost-subdomain": "http://foo.localhost/",
    "compose-db": "http://db/",
    "compose-redis": "http://redis/",
    "compose-redis-with-its-port": "http://redis:6379/",
    "single-label-intranet-name": "http://intranet/",
    "mdns-local": "http://printer.local/",
    "dot-internal": "http://service.internal/",
    "dot-localdomain": "http://host.localdomain/",
    # --- IPv4 literals, plain
    "loopback": "http://127.0.0.1/",
    "loopback-range": "http://127.8.9.10/feed",
    "private-10": "http://10.0.0.5/",
    "private-192-168": "http://192.168.1.10/",
    "private-172-16": "http://172.16.0.1/",
    "private-172-31": "http://172.31.255.255/",
    "metadata-endpoint": "http://169.254.169.254/latest/meta-data/",
    "carrier-grade-nat": "http://100.64.0.1/",
    "unspecified": "http://0.0.0.0/",
    "multicast": "http://224.0.0.1/",
    "reserved": "http://240.0.0.1/",
    "broadcast": "http://255.255.255.255/",
    "public-ip-literal": "http://93.184.216.34/",  # even public IPs: hostnames only
    # --- IPv4 in other notations that parsers and resolvers still understand
    "decimal-127.0.0.1": "http://2130706433/",
    "decimal-192.168.1.1": "http://3232235777/",
    "decimal-metadata": "http://2852039166/",
    "hex-127.0.0.1": "http://0x7f000001/",
    "hex-per-octet": "http://0x7f.0.0.1/",
    "octal": "http://0177.0.0.1/",
    "short-form-127.1": "http://127.1/",
    "short-form-zero": "http://0/",
    "mixed-notation": "http://0x7f.1/",
    # --- IPv6 literals
    "ipv6-loopback": "http://[::1]/",
    "ipv4-mapped-ipv6-loopback": "http://[::ffff:127.0.0.1]/",
    "ipv4-mapped-ipv6-hex-form": "http://[::ffff:7f00:1]/",
    "ipv4-mapped-ipv6-private": "http://[::ffff:10.0.0.5]/",
    "ipv6-link-local": "http://[fe80::1]/",
    "ipv6-unique-local": "http://[fc00::1]/",
    "ipv6-unspecified": "http://[::]/",
    "ipv6-documentation": "http://[2001:db8::1]/",
    "ipv6-public-literal": "http://[2606:2800:220:1:248:1893:25c8:1946]/",
    # --- credentials
    "user-and-password": "http://user:pass@example.com/",
    "user-only": "http://user@example.com/",
    "empty-credentials": "http://:@example.com/",
    "credentials-hide-the-real-host": "https://example.com@127.0.0.1/",
    "backslash-host-confusion": "http://example.com\\@127.0.0.1/",
    # --- ports
    "non-standard-port": "http://example.com:8080/",
    "https-on-port-80": "https://example.com:80/",
    "http-on-port-443": "http://example.com:443/",
    "port-zero": "http://example.com:0/",
    "port-out-of-range": "http://example.com:99999/",
    "port-not-a-number": "http://example.com:abc/",
    "database-port": "http://example.com:5432/",
    # --- schemes
    "ftp": "ftp://example.com/",
    "file": "file:///etc/passwd",
    "gopher": "gopher://example.com/",
    "javascript": "javascript:alert(1)",
    "data": "data:text/plain,hello",
    "websocket": "ws://example.com/",
    "scheme-relative": "//example.com/feed",
    "no-scheme": "example.com/feed",
    "empty": "",
    "only-whitespace": "   ",
    # --- malformed or ambiguous
    "no-host": "http:///path",
    "no-slashes": "http:example.com",
    "space-in-host": "http://exam ple.com/",
    "newline-injection": "http://example.com/\r\nHost: evil.com",
    "tab": "http://example.com/\tx",
    "null-byte": "http://exa\x00mple.com/",
    "cyrillic-homograph-host": "http://exаmple.com/",
    "fullwidth-dots": "http://127。0。0。1/",
    "fullwidth-letters": "http://ｅxample.com/",
    "leading-hyphen-label": "http://-bad.example.com/",
    "underscore-in-host": "http://ex_ample.com/",
    "empty-label": "http://example..com/",
    "label-too-long": "http://" + "a" * 64 + ".com/",
    "url-too-long": "http://example.com/" + "a" * 2100,
    "unbalanced-ipv6-brackets": "http://[::1/",
}


@pytest.mark.parametrize("url", REJECTED_WITHOUT_DNS.values(), ids=REJECTED_WITHOUT_DNS.keys())
async def test_unsafe_urls_are_rejected_before_any_dns_lookup(url: str) -> None:
    resolver = FakeResolver()

    with pytest.raises(UnsafeUrlError):
        await validate_public_url(url, resolver)

    assert resolver.calls == []


# Hostnames that look fine but whose DNS answer is not a public address.
REJECTED_AFTER_DNS = {
    "resolves-to-private-10": "http://internal.example.com/feed",
    "resolves-to-metadata-ip": "http://metadata.example.com/",
    "resolves-to-loopback": "http://loopback.example.com/",
    "one-public-one-private-record": "http://rebind.example.com/",
    "resolves-to-ipv6-loopback": "http://ipv6-loopback.example.com/",
    "resolves-to-ipv4-mapped-private": "http://mapped.example.com/",
    "resolves-to-nat64-wrapped-loopback": "http://nat64.example.com/",
    "resolves-to-6to4-wrapped-loopback": "http://sixtofour.example.com/",
    "resolves-to-ipv6-unique-local": "http://ula.example.com/",
    "resolves-to-ipv6-link-local": "http://linklocal.example.com/",
    "resolves-to-multicast": "http://multicast.example.com/",
    "resolves-to-carrier-grade-nat": "http://cgnat.example.com/",
    "resolves-to-nothing": "http://empty.example.com/",
    "does-not-resolve": "http://nxdomain.example.com/",
}


@pytest.mark.parametrize("url", REJECTED_AFTER_DNS.values(), ids=REJECTED_AFTER_DNS.keys())
async def test_hostnames_that_resolve_to_non_public_addresses_are_rejected(url: str) -> None:
    resolver = FakeResolver()

    with pytest.raises(UnsafeUrlError, match="public address"):
        await validate_public_url(url, resolver)

    assert len(resolver.calls) == 1  # it did look the name up, and the answer was the problem


async def test_a_resolver_that_says_a_single_label_name_is_public_is_not_believed() -> None:
    resolver = FakeResolver({"db": ["93.184.216.34"]})

    with pytest.raises(UnsafeUrlError):
        await validate_public_url("http://db/", resolver)

    assert resolver.calls == []  # rejected on the name alone, independent of DNS


async def test_unresolvable_and_private_hosts_get_the_same_message() -> None:
    # Different messages would let a caller probe which internal names exist.
    resolver = FakeResolver()
    with pytest.raises(UnsafeUrlError) as private:
        await validate_public_url("http://internal.example.com/", resolver)
    with pytest.raises(UnsafeUrlError) as missing:
        await validate_public_url("http://nxdomain.example.com/", resolver)

    assert private.value.detail == missing.value.detail


ACCEPTED = {
    "https": "https://example.com/feed.xml",
    "http": "http://example.com/feed.xml",
    "explicit-standard-https-port": "https://example.com:443/feed",
    "explicit-standard-http-port": "http://example.com:80/feed",
    "query-and-fragment": "https://example.com/feed?tag=python&x=1#top",
    "uppercase-scheme-and-host": "HTTPS://EXAMPLE.com/Feed",
    "surrounding-whitespace": "  https://example.com/feed  ",
    "dual-stack-host": "https://feeds.example.org/rss",
}


@pytest.mark.parametrize("url", ACCEPTED.values(), ids=ACCEPTED.keys())
async def test_public_urls_are_accepted(url: str) -> None:
    result = await validate_public_url(url, FakeResolver())

    assert isinstance(result, ValidatedUrl)
    assert result.url == url.strip()
    assert result.addresses  # what the host resolved to, for the fetcher to use


async def test_internationalized_hosts_are_accepted_in_punycode() -> None:
    resolver = FakeResolver({"xn--bcher-kva.example": ["93.184.216.34"]})

    result = await validate_public_url("https://xn--bcher-kva.example/rss", resolver)

    assert result.host == "xn--bcher-kva.example"


def test_syntax_check_returns_the_lowercase_host_without_a_trailing_dot() -> None:
    assert check_url_syntax("HTTPS://Feeds.Example.COM./rss") == "feeds.example.com"


def test_unsafe_url_errors_are_422_with_their_own_code() -> None:
    with pytest.raises(UnsafeUrlError) as error:
        check_url_syntax("ftp://example.com/")

    assert (error.value.status_code, error.value.code) == (422, "unsafe_url")


@pytest.mark.parametrize(
    "address",
    [
        "93.184.216.34",
        "8.8.8.8",
        "2606:2800:220:1:248:1893:25c8:1946",
        "2001:4860:4860::8888",
        "::ffff:8.8.8.8",  # IPv4-mapped, wrapping a public address
        "64:ff9b::808:808",  # NAT64 wrapping 8.8.8.8 (what DNS64 networks hand out)
        "2002:5db8:d822::",  # 6to4 wrapping 93.184.216.34
        "2001:0:808:808::a247:27dd",  # Teredo: public server 8.8.8.8, public client 93.184.216.34
    ],
)
def test_public_addresses_are_recognised(address: str) -> None:
    assert is_public_address(address) is True


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "255.255.255.255",
        "::1",
        "::",
        "fe80::1",
        "fe80::1%eth0",  # a zone suffix must not hide a link-local address
        "fc00::1",
        "ff02::1",
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.5",
        "64:ff9b::7f00:1",
        "2002:7f00:1::",
        "2001:0:808:808::f5ff:fffa",  # Teredo: public server, but the client is 10.0.0.5
        "2001:0:a00:1::a247:27dd",  # Teredo: private server 10.0.0.1, public client
        "not-an-ip",
        "",
    ],
)
def test_non_public_or_invalid_addresses_are_rejected(address: str) -> None:
    assert is_public_address(address) is False


async def test_system_resolver_passes_ip_addresses_through_without_dns() -> None:
    # Numeric hosts are parsed locally by getaddrinfo, so this makes no network request.
    assert await system_resolver("127.0.0.1") == ["127.0.0.1"]
    assert await system_resolver("::1") == ["::1"]
