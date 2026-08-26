from abc import ABC, abstractmethod
import requests

class PostSource(ABC):
    @abstractmethod
    def get_new_posts(self) -> list[dict]:
        """Return posts newer than what's already been seen."""
        ...

class MockSource(PostSource):
    def __init__(self, base_url="http://127.0.0.1:8000", account="demo_user"):
        self.base_url = base_url
        self.account = account

    def get_new_posts(self) -> list[dict]:
        resp = requests.get(f"{self.base_url}/api/v1/accounts/{self.account}/statuses")
        resp.raise_for_status()
        return resp.json()

class TruthSocialSource(PostSource):
    """
    NOT production-ready. Documents the intended real integration based on
    reverse-engineering findings (see README) — Mastodon-shaped endpoint,
    likely requires an authenticated session. Left unimplemented/stubbed
    given the assignment's 72-hour window; see README's ingestion trade-off
    discussion for why MockSource is what's actually demoed.
    """
    def __init__(self, account_handle="realDonaldTrump"):
        self.account_handle = account_handle
        raise NotImplementedError(
            "Requires authenticated session against truthsocial.com's "
            "Mastodon-derived API — see README Part 1 for investigation notes."
        )

    def get_new_posts(self) -> list[dict]:
        ...