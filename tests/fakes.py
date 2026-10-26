"""Test doubles that keep tests off the network."""

# Pretend DNS zone. The addresses mirror what the real names would resolve to.
DEFAULT_DNS_RECORDS: dict[str, list[str]] = {
    "example.com": ["93.184.216.34"],
    "feeds.example.org": ["93.184.216.35", "2606:2800:220:1:248:1893:25c8:1946"],
    "internal.example.com": ["10.0.0.5"],  # a public-looking name pointing at a private IP
    "metadata.example.com": ["169.254.169.254"],
    "loopback.example.com": ["127.0.0.1"],
    "rebind.example.com": ["93.184.216.34", "10.0.0.5"],  # one good record, one bad
    "ipv6-loopback.example.com": ["::1"],
    "mapped.example.com": ["::ffff:10.0.0.5"],
    "nat64.example.com": ["64:ff9b::7f00:1"],  # NAT64 wrapping 127.0.0.1
    "sixtofour.example.com": ["2002:7f00:1::"],  # 6to4 wrapping 127.0.0.1
    "ula.example.com": ["fc00::1"],
    "linklocal.example.com": ["fe80::1"],
    "multicast.example.com": ["224.0.0.1"],
    "cgnat.example.com": ["100.64.0.1"],
    "empty.example.com": [],
    # What Docker's DNS would answer for the Compose service names:
    "db": ["172.21.0.2"],
    "redis": ["172.21.0.3"],
}


class FakeResolver:
    """A DNS resolver that answers from a dict and records which names were asked for."""

    def __init__(self, records: dict[str, list[str]] | None = None) -> None:
        self.records = DEFAULT_DNS_RECORDS | (records or {})
        self.calls: list[str] = []

    async def __call__(self, host: str) -> list[str]:
        self.calls.append(host)
        if host not in self.records:
            raise OSError(f"cannot resolve {host}")  # what a failed DNS lookup raises
        return self.records[host]
