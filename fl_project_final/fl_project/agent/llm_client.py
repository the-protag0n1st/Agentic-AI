"""LLM Client Abstraction Layer with Quota Tracking and Safe Quota Stopping.

Supports MockLLM for unit tests, GroqLLM for cloud production inference,
and OllamaLLM for local models. Decoupled from agentic reasoning logic.
"""

from abc import ABC, abstractmethod
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

from quota import QuotaPauseException, QuotaTracker


class LLMClient(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        """Generate text completion from prompt."""
        pass


class MockLLM(LLMClient):
    """Deterministic mock provider for unit testing without API keys or network calls."""

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "{}",
        side_effect: Optional[Callable[[str], str]] = None,
        quota_tracker: Optional[QuotaTracker] = None,
        mock_headers: Optional[Dict[str, str]] = None,
    ):
        self.responses = responses or {}
        self.default_response = default_response
        self.side_effect = side_effect
        self.call_history: List[Dict[str, Any]] = []
        self.last_call_metadata: Dict[str, Any] = {}
        self.quota_tracker = quota_tracker
        self.mock_headers = mock_headers

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        if self.quota_tracker:
            self.quota_tracker.check_quota(safety_enabled=True)

        t0 = time.time()
        call_record = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "json_mode": json_mode,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timestamp": time.time(),
        }
        self.call_history.append(call_record)

        if self.side_effect:
            resp = self.side_effect(prompt)
        else:
            resp = self.default_response
            for key, val in self.responses.items():
                if key.lower() in (prompt or "").lower() or key.lower() in (system_prompt or "").lower():
                    resp = val
                    break

        latency_ms = (time.time() - t0) * 1000.0
        in_tok = len(prompt.split())
        out_tok = len(resp.split())
        tot_tok = in_tok + out_tok

        headers = self.mock_headers or {}
        h_lower = {k.lower(): str(v) for k, v in headers.items()}

        def _to_int(val):
            try:
                return int(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        rem_req = _to_int(h_lower.get("x-ratelimit-remaining-requests"))
        lim_req = _to_int(h_lower.get("x-ratelimit-limit-requests"))
        reset_req = h_lower.get("x-ratelimit-reset-requests")
        rem_tok = _to_int(h_lower.get("x-ratelimit-remaining-tokens"))
        lim_tok = _to_int(h_lower.get("x-ratelimit-limit-tokens"))
        reset_tok = h_lower.get("x-ratelimit-reset-tokens")

        if self.quota_tracker:
            self.quota_tracker.update_from_headers(headers)
            self.quota_tracker.record_call(in_tok, out_tok, tot_tok)

        self.last_call_metadata = {
            "model": "mock",
            "retries": 0,
            "latency_ms": latency_ms,
            "status": "success",
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
            "request_id": f"mock-{int(time.time() * 1000)}",
            "requests_remaining": rem_req,
            "requests_limit": lim_req,
            "requests_reset_time": reset_req,
            "tokens_remaining": rem_tok,
            "tokens_limit": lim_tok,
            "tokens_reset_time": reset_tok,
        }
        return resp


class GroqLLM(LLMClient):
    """Groq Cloud LLM provider with bounded exponential backoff on rate limits.

    Ensures scientific rigor: NEVER switches to a different model automatically.
    Captures rate-limit headers and enforces safe quota margins.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 15,
        initial_backoff: float = 2.0,
        max_backoff: float = 60.0,
        quota_tracker: Optional[QuotaTracker] = None,
        quota_safe: bool = True,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or os.environ.get("FL_LLM_MODEL", "openai/gpt-oss-120b")
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.quota_tracker = quota_tracker
        self.quota_safe = quota_safe
        self._client = None
        self.last_call_metadata: Dict[str, Any] = {}

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                if not self.api_key:
                    raise ValueError(
                        "GROQ_API_KEY not set. Provide api_key or set GROQ_API_KEY in environment or .env."
                    )
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("groq package not installed. Run: pip install groq")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response_format = {"type": "json_object"} if json_mode else None

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        retries = 0
        backoff = self.initial_backoff
        start_time = time.time()

        while True:
            try:
                # Quota safety check before making outbound network call
                if self.quota_tracker and self.quota_safe:
                    self.quota_tracker.check_quota(safety_enabled=True)

                raw_resp = client.chat.completions.with_raw_response.create(**kwargs)
                completion = raw_resp.parse()
                headers = getattr(raw_resp, "headers", {}) or {}

                latency_ms = (time.time() - start_time) * 1000.0
                usage = getattr(completion, "usage", None)
                in_tok = getattr(usage, "prompt_tokens", None) if usage else None
                out_tok = getattr(usage, "completion_tokens", None) if usage else None
                tot_tok = getattr(usage, "total_tokens", None) if usage else None
                req_id = getattr(completion, "id", None)

                # Safe rate-limit header extraction
                h_lower = {k.lower(): str(v) for k, v in headers.items()}

                def _to_int(val):
                    try:
                        return int(val) if val is not None else None
                    except (ValueError, TypeError):
                        return None

                rem_req = _to_int(h_lower.get("x-ratelimit-remaining-requests"))
                lim_req = _to_int(h_lower.get("x-ratelimit-limit-requests"))
                reset_req = h_lower.get("x-ratelimit-reset-requests")
                rem_tok = _to_int(h_lower.get("x-ratelimit-remaining-tokens"))
                lim_tok = _to_int(h_lower.get("x-ratelimit-limit-tokens"))
                reset_tok = h_lower.get("x-ratelimit-reset-tokens")

                if self.quota_tracker:
                    self.quota_tracker.update_from_headers(headers)
                    self.quota_tracker.record_call(in_tok, out_tok, tot_tok)

                self.last_call_metadata = {
                    "model": self.model,
                    "retries": retries,
                    "latency_ms": latency_ms,
                    "status": "success",
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok,
                    "request_id": req_id,
                    "requests_remaining": rem_req,
                    "requests_limit": lim_req,
                    "requests_reset_time": reset_req,
                    "tokens_remaining": rem_tok,
                    "tokens_limit": lim_tok,
                    "tokens_reset_time": reset_tok,
                }
                return completion.choices[0].message.content or "{}"

            except QuotaPauseException:
                # Re-raise QuotaPauseException immediately without retrying
                raise

            except Exception as e:
                retries += 1
                latency_ms = (time.time() - start_time) * 1000.0
                if retries > self.max_retries:
                    self.last_call_metadata = {
                        "model": self.model,
                        "retries": retries - 1,
                        "latency_ms": latency_ms,
                        "status": "llm_error",
                        "error": str(e),
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "request_id": None,
                    }
                    raise RuntimeError(
                        f"Groq API call failed after {self.max_retries} retries on configured model '{self.model}': {e}"
                    )

                # Parse retry-after if present in exception text (e.g. "Please try again in 3.45s" or "2m4.4s")
                wait_time = backoff
                match = re.search(r"try again in (?:(\d+)m)?([0-9\.]+)s", str(e), re.IGNORECASE)
                if match:
                    try:
                        minutes = float(match.group(1)) if match.group(1) else 0.0
                        seconds = float(match.group(2)) if match.group(2) else 0.0
                        wait_time = minutes * 60.0 + seconds + 1.0
                    except Exception:
                        pass
                wait_time = min(wait_time, self.max_backoff)

                time.sleep(wait_time)
                backoff = min(backoff * 2.0, self.max_backoff)


class OllamaLLM(LLMClient):
    """Local Ollama LLM provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3:8b",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.last_call_metadata: Dict[str, Any] = {}

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        try:
            import urllib.request
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if system_prompt:
                payload["system"] = system_prompt
            if json_mode:
                payload["format"] = "json"

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            latency_ms = (time.time() - t0) * 1000.0
            self.last_call_metadata = {
                "model": self.model,
                "retries": 0,
                "latency_ms": latency_ms,
                "status": "success",
            }
            return result.get("response", "{}")
        except Exception as e:
            self.last_call_metadata = {
                "model": self.model,
                "retries": 0,
                "latency_ms": 0.0,
                "status": "llm_error",
                "error": str(e),
            }
            raise RuntimeError(f"Ollama API call failed: {e}")
