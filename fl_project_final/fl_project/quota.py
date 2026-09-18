"""Groq API Quota Tracking and Safety Enforcement.

Provides real-time tracking of API rate-limit headers (RPM, RPD, TPM, TPD)
and enforces safety thresholds to pause experiments before quotas are exhausted.
"""

from typing import Any, Dict, Optional


class QuotaPauseException(BaseException):
    """Raised when available Groq API quota drops below configured safety margins."""
    pass


class QuotaTracker:
    """Tracks token consumption and rate-limit headers across LLM calls."""

    def __init__(
        self,
        daily_token_limit: int = 200000,
        daily_request_limit: int = 1000,
        token_safety_margin: float = 0.90,
        request_safety_margin: float = 0.90,
        min_remaining_tokens: int = 1000,
        min_remaining_requests: int = 20,
    ):
        self.daily_token_limit = daily_token_limit
        self.daily_request_limit = daily_request_limit
        self.token_safety_margin = token_safety_margin
        self.request_safety_margin = request_safety_margin
        self.min_remaining_tokens = min_remaining_tokens
        self.min_remaining_requests = min_remaining_requests

        # Cumulative counters
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0
        self.total_calls: int = 0
        self.peak_tokens_per_call: int = 0

        # Latest rate limit headers observed from API
        self.last_requests_remaining: Optional[int] = None
        self.last_requests_limit: Optional[int] = None
        self.last_requests_reset_time: Optional[str] = None
        self.last_tokens_remaining: Optional[int] = None
        self.last_tokens_limit: Optional[int] = None
        self.last_tokens_reset_time: Optional[str] = None

        # Pause state
        self.quota_paused: bool = False
        self.pause_reason: Optional[str] = None

    def update_from_headers(self, headers: Dict[str, Any]) -> None:
        """Parse rate limit headers from an HTTP response."""
        if not headers:
            return

        def _to_int(val):
            try:
                return int(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        # Case-insensitive lookup
        h_lower = {k.lower(): str(v) for k, v in headers.items()}

        rem_req = _to_int(h_lower.get("x-ratelimit-remaining-requests"))
        if rem_req is not None:
            self.last_requests_remaining = rem_req

        lim_req = _to_int(h_lower.get("x-ratelimit-limit-requests"))
        if lim_req is not None:
            self.last_requests_limit = lim_req

        reset_req = h_lower.get("x-ratelimit-reset-requests")
        if reset_req:
            self.last_requests_reset_time = reset_req

        rem_tok = _to_int(h_lower.get("x-ratelimit-remaining-tokens"))
        if rem_tok is not None:
            self.last_tokens_remaining = rem_tok

        lim_tok = _to_int(h_lower.get("x-ratelimit-limit-tokens"))
        if lim_tok is not None:
            self.last_tokens_limit = lim_tok

        reset_tok = h_lower.get("x-ratelimit-reset-tokens")
        if reset_tok:
            self.last_tokens_reset_time = reset_tok

    def record_call(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """Accumulate token counts from a completed call."""
        self.total_calls += 1
        in_t = input_tokens or 0
        out_t = output_tokens or 0
        tot_t = total_tokens if total_tokens is not None else (in_t + out_t)

        self.total_input_tokens += in_t
        self.total_output_tokens += out_t
        self.total_tokens += tot_t
        if tot_t > self.peak_tokens_per_call:
            self.peak_tokens_per_call = tot_t

    def check_quota(self, safety_enabled: bool = True) -> None:
        """Verify remaining quota before initiating an LLM call.

        Raises:
            QuotaPauseException: If known quota is at or below safety threshold.
        """
        if not safety_enabled:
            return

        import time

        def _parse_reset_seconds(reset_str: Optional[str]) -> float:
            if not reset_str:
                return 5.0
            s = str(reset_str).strip().lower()
            try:
                if s.endswith("ms"):
                    return max(0.5, float(s[:-2]) / 1000.0)
                if s.endswith("s") and not s.endswith("ms"):
                    if "m" in s:
                        parts = s.split("m")
                        mins = float(parts[0])
                        secs = float(parts[1].rstrip("s")) if parts[1].rstrip("s") else 0.0
                        return mins * 60.0 + secs
                    return max(1.0, float(s[:-1]))
                if s.endswith("m"):
                    return float(s[:-1]) * 60.0
                if s.endswith("h"):
                    return float(s[:-1]) * 3600.0
                return max(1.0, float(s))
            except Exception:
                return 5.0

        # 1. Check requests remaining
        if self.last_requests_remaining is not None:
            if self.last_requests_limit is not None:
                min_req = max(
                    int(self.last_requests_limit * (1.0 - self.request_safety_margin)),
                    self.min_remaining_requests,
                )
            else:
                min_req = self.min_remaining_requests

            if self.last_requests_remaining <= min_req:
                if self.last_requests_reset_time:
                    wait_sec = _parse_reset_seconds(self.last_requests_reset_time)
                    if wait_sec <= 90.0:
                        print(f"  [Rate Limit Backoff] Requests low ({self.last_requests_remaining}/{self.last_requests_limit}). Waiting {wait_sec + 1.0:.1f}s...")
                        time.sleep(wait_sec + 1.0)
                        self.last_requests_remaining = self.last_requests_limit
                    else:
                        self.quota_paused = True
                        self.pause_reason = (
                            f"Requests remaining ({self.last_requests_remaining}) is at or below "
                            f"safety threshold ({min_req}). Limit: {self.last_requests_limit}"
                        )
                        raise QuotaPauseException(self.pause_reason)
                else:
                    self.quota_paused = True
                    self.pause_reason = (
                        f"Requests remaining ({self.last_requests_remaining}) is at or below "
                        f"safety threshold ({min_req}). Limit: {self.last_requests_limit}"
                    )
                    raise QuotaPauseException(self.pause_reason)

        # 2. Check tokens remaining
        if self.last_tokens_remaining is not None:
            # If the limit is small (e.g. <= 30000), it is a Tokens-Per-Minute (TPM) window
            if self.last_tokens_limit is not None and self.last_tokens_limit <= 50000:
                min_tok = 500  # Smaller safety threshold for per-minute window
            elif self.last_tokens_limit is not None:
                min_tok = max(
                    int(self.last_tokens_limit * (1.0 - self.token_safety_margin)),
                    self.min_remaining_tokens,
                )
            else:
                min_tok = self.min_remaining_tokens

            if self.last_tokens_remaining <= min_tok:
                if self.last_tokens_reset_time:
                    wait_sec = _parse_reset_seconds(self.last_tokens_reset_time)
                    # If reset is short (<= 90 seconds, typical TPM rolling window), sleep rather than abort
                    if wait_sec <= 90.0:
                        print(f"  [Rate Limit Backoff] Per-minute tokens low ({self.last_tokens_remaining}/{self.last_tokens_limit}). Waiting {wait_sec + 1.5:.1f}s for TPM bucket refill...")
                        time.sleep(wait_sec + 1.5)
                        self.last_tokens_remaining = self.last_tokens_limit
                    else:
                        self.quota_paused = True
                        self.pause_reason = (
                            f"Tokens remaining ({self.last_tokens_remaining}) is at or below "
                            f"safety threshold ({min_tok}). Limit: {self.last_tokens_limit}"
                        )
                        raise QuotaPauseException(self.pause_reason)
                else:
                    self.quota_paused = True
                    self.pause_reason = (
                        f"Tokens remaining ({self.last_tokens_remaining}) is at or below "
                        f"safety threshold ({min_tok}). Limit: {self.last_tokens_limit}"
                    )
                    raise QuotaPauseException(self.pause_reason)

    def get_summary(self) -> Dict[str, Any]:
        """Return full usage and quota summary."""
        avg_tokens = (
            self.total_tokens / self.total_calls if self.total_calls > 0 else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "average_tokens_per_call": avg_tokens,
            "peak_tokens_per_call": self.peak_tokens_per_call,
            "quota_paused": self.quota_paused,
            "pause_reason": self.pause_reason,
            "last_requests_remaining": self.last_requests_remaining,
            "last_requests_limit": self.last_requests_limit,
            "last_tokens_remaining": self.last_tokens_remaining,
            "last_tokens_limit": self.last_tokens_limit,
        }
