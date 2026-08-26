from abc import ABC, abstractmethod
from dataclasses import dataclass
import requests
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    post_id: int
    text: str
    created_at: str  # ISO timestamp from the source, as published
    tickers: list[str]
    companies: list[str]
 
 
class AlertChannel(ABC):
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Attempt one delivery. Return True on confirmed success."""
        ...
 
 
class ConsoleAlertChannel(AlertChannel):
    """Prints alerts to stdout. Useful for local demoing without setting
    up a real webhook/bot."""
 
    def send(self, alert: Alert) -> bool:
        print(
            f"[ALERT] post {alert.post_id} @ {alert.created_at} | "
            f"tickers={alert.tickers} companies={alert.companies}\n"
            f"        {alert.text!r}"
        )
        return True
 
 

class DiscordWebhookAlertChannel(AlertChannel):
    """
    Discord incoming-webhook channel. POSTs JSON with a "content" field.
    Docs: https://discord.com/developers/docs/resources/webhook#execute-webhook

    Notes:
      - Success is 204 No Content, not 200.
      - Discord rate-limits webhooks at ~5 requests / 2s per webhook and
        returns 429 + a "retry_after" field on the body when exceeded.
        send_with_retry will use this value automatically if present.
    """

    MAX_CONTENT_LEN = 2000  # Discord hard limit on the "content" field

    def __init__(self, webhook_url: str, timeout: float = 5.0):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.retry_after: float | None = None  # set by send() when Discord tells us to wait

    def send(self, alert: Alert) -> bool:
        self.retry_after = None  # reset each attempt so stale values don't leak forward

        message = (
            f"**Stock mention detected** ({', '.join(alert.tickers) or ', '.join(alert.companies)})\n"
            f"posted: {alert.created_at}\n"
            f"{alert.text}\n"
        )
        if len(message) > self.MAX_CONTENT_LEN:
            message = message[: self.MAX_CONTENT_LEN - 3] + "..."

        body = {"content": message}
        try:
            resp = requests.post(self.webhook_url, json=body, timeout=self.timeout)

            if resp.status_code == 429:
                # Discord puts retry_after (seconds, float) in the JSON body;
                # fall back to the Retry-After header if the body is missing it.
                retry_after = None
                try:
                    retry_after = resp.json().get("retry_after")
                except ValueError:
                    pass
                if retry_after is None:
                    retry_after = resp.headers.get("Retry-After")
                self.retry_after = float(retry_after) if retry_after is not None else None
                logger.warning("Rate limited by Discord; retry_after=%s", self.retry_after)
                return False

            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.warning("Webhook delivery failed: %s", e)
            return False
 
 
def send_with_retry(
        channel: AlertChannel, 
        alert: Alert, 
        max_attempts: int = 3, 
        backoff_base: float = 1.5,  
        max_wait: float = 60.0
        ) -> bool:
    """
    Retry delivery with exponential backoff. If the responce has a `retry_after`
    attribute, it will use that as the backoff time. Idempotency is enforced by
    the caller (Agent), it marks a post as alerted if this returns true. That 
    mark is persistent across restarts via saved states. A failed/aborted 
    attempt is safe to retry on the next poll.
    """
    for attempt in range(1, max_attempts + 1):
        if channel.send(alert):
            return True
        if attempt < max_attempts:
            retry_after = getattr(channel, "retry_after", None)
            if retry_after is not None:
                sleep_for = min(retry_after, max_wait)
                logger.info("Alert send attempt %d/%d rate-limited, waiting %.1fs (server-specified)", attempt, max_attempts, sleep_for)
            else:
                sleep_for = backoff_base ** attempt
                logger.info("Alert send attempt %d/%d failed, retrying in %.1fs", attempt, max_attempts, sleep_for)
            time.sleep(sleep_for)
    logger.error("Alert for post %s failed after %d attempts", alert.post_id, max_attempts)
    return False