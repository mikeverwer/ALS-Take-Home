from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass
class ProbeResult:
    """Outcome of a one-shot public-access check. Not part of the
    PostSource contract - this is investigation output, not post data."""
    status_code: int
    content_type: str
    looks_like_json: bool
    post_count: int | None
    blocked: bool
    snippet: str


class TruthSocialSource(PostSource):
    """
    NOT production ready. This class documents the intended real integration
    based on reverse-engineering. TruthSocial is a Mastadon fork, and so 
    shares the same shape of endpoint, which MockSource imitates. 
    The class is intentionally left unimplemented due to time constraints.
    """
    # The real endpoint needs the numeric account id, not the @handle.
    # Sourced from w2rc/truthbrush#32.
    _ACCOUNT_IDS = {
        "realDonaldTrump": "107780257626128497",
    }
 
    def __init__(self, account_handle="realDonaldTrump"):
        self.account_handle = account_handle

    def get_new_posts(self) -> list[dict]:
        raise NotImplementedError(
            "Left unimplemented due to time constraints."
        )

    def probe_public_access(self, timeout: float = 10.0) -> ProbeResult:
        """
        One-shot diagnostic, NOT part of the PostSource contract.
 
        Sends a single unauthenticated GET against the real statuses
        endpoint, replicating the technique documented in
        w2rc/truthbrush#32, and reports whether it currently succeeds.
 
        This is meant to be run by hand, once, to check current status -
        e.g. `python sources.py`. It is deliberately NOT wired into
        agent.py or the poll loop: looping this would cross from a single
        investigative request into actual sustained traffic against a
        live account this project doesn't have permission to hammer,
        which is exactly the ToS-fragility the README's ethics section
        flags. One request, by hand, is the intended usage.
        """
        account_id = self._ACCOUNT_IDS.get(self.account_handle)
        if account_id is None:
            raise ValueError(
                f"No known numeric account id for {self.account_handle!r}; "
                "the real endpoint needs the numeric id, not the @handle."
            )
 
        url = f"https://truthsocial.com/api/v1/accounts/{account_id}/statuses"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) "
                "Gecko/20100101 Firefox/131.0"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://truthsocial.com/search?q=trump",
            "DNT": "1",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
 
        resp = requests.get(
            url,
            params={"exclude_replies": "true", "with_muted": "true"},
            headers=headers,
            timeout=timeout,
        )
 
        content_type = resp.headers.get("Content-Type", "")
        looks_like_json = "application/json" in content_type
        post_count = None
        if looks_like_json:
            try:
                post_count = len(resp.json())
            except ValueError:
                looks_like_json = False
 
        return ProbeResult(
            status_code=resp.status_code,
            content_type=content_type,
            looks_like_json=looks_like_json,
            post_count=post_count,
            blocked=not looks_like_json,
            snippet=resp.text[:300].replace("\n", " "),
        )
 
 
if __name__ == "__main__":
    # Manual, one-off check - see probe_public_access()'s docstring.
    # Run with: python sources.py
    result = TruthSocialSource().probe_public_access()
    print(result)
    if result.blocked:
        lowered = result.snippet.lower()
        if "cloudflare" in lowered or "just a moment" in lowered:
            print("\nLooks like a Cloudflare challenge page - consistent "
                  "with the Jan 2026 report in w2rc/truthbrush#32.")
        else:
            print("\nBlocked, but doesn't look like a typical Cloudflare "
                  "challenge page - worth a manual look at the snippet above.")
    else:
        print(f"\nSUCCESS: got JSON with {result.post_count} post(s). "
              "This would contradict the Jan 2026 report - worth double "
              "checking and updating the README if reproducible.")

