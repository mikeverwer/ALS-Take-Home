from abc import ABC, abstractmethod
import requests

class PostSource(ABC):
    @abstractmethod
    def get_new_posts(self) -> list[dict]:
        """
        Return all posts currently available from this source (newest first).
        Deduplication needs to be handled by the caller.
        """
        ...

class MockSource(PostSource):
    def __init__(self, base_url="http://127.0.0.1:8000", account="demo_user"):
        self.base_url = base_url
        self.account = account

    def get_new_posts(self) -> list[dict]:
        resp = requests.get(
            f"{self.base_url}/api/v1/accounts/{self.account}/statuses",
            timeout=5,
            )
        resp.raise_for_status()
        return resp.json()

class TruthSocialSource(PostSource):
    """
    NOT production ready. This class documents the intended real integration
    based on reverse-engineering. TruthSocial is a Mastadon fork, and so 
    shares the same shape of endpoint, which MockSource imitates. 
    The class is intentionally left unimplemented due to time constraints.
    """
    def __init__(self, account_handle="realDonaldTrump"):
        self.account_handle = account_handle

    def get_new_posts(self) -> list[dict]:
        raise NotImplementedError(
            "Requires authenticated session against truthsocial.com's "
            "Mastodon-derived API — see README Part 1 for investigation notes."
        )
