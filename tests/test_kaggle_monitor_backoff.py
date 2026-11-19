"""Public-API tests for the 429-aware log-stream reconnect pacing."""
from __future__ import annotations

import pytest

from cli.kaggle_monitor import ReconnectBackoff


class _FakeResponse:
    def __init__(self, retry_after: str | None = None):
        self.headers = {"Retry-After": retry_after} if retry_after else {}


class _FakeError(Exception):
    def __init__(self, message: str = "", *, status_code=None, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeResponse(retry_after)


def _backoff() -> ReconnectBackoff:
    return ReconnectBackoff(base_seconds=1.0, cap_seconds=60.0,
                            rate_limit_cap_seconds=300.0, jitter=0.0)


def test_is_rate_limited_detects_status_code():
    assert ReconnectBackoff.is_rate_limited(_FakeError(status_code=429))


def test_is_rate_limited_detects_message_text():
    assert ReconnectBackoff.is_rate_limited(
        _FakeError("Client Error: Too Many Requests for url"))


def test_is_rate_limited_false_for_transport_error():
    assert not ReconnectBackoff.is_rate_limited(_FakeError("connection reset"))


def test_delay_grows_exponentially_without_jitter():
    backoff = _backoff()
    error = _FakeError("connection reset")
    assert [backoff.delay(attempt, error) for attempt in (1, 2, 3, 4)] == [
        1.0, 2.0, 4.0, 8.0]


def test_delay_caps_plain_errors_at_cap_seconds():
    assert _backoff().delay(50, _FakeError("boom")) == pytest.approx(60.0)


def test_delay_uses_the_higher_rate_limit_ceiling_for_429():
    assert _backoff().delay(50, _FakeError(status_code=429)) == pytest.approx(300.0)


def test_delay_honours_retry_after_above_the_exponential_value():
    error = _FakeError(status_code=429, message="429", retry_after="120")
    assert _backoff().delay(1, error) == pytest.approx(120.0)


def test_delay_never_exceeds_the_rate_limit_ceiling():
    error = _FakeError(status_code=429, message="429", retry_after="9999")
    assert _backoff().delay(1, error) == pytest.approx(300.0)


def test_delay_without_error_still_backs_off():
    assert _backoff().delay(1) == pytest.approx(1.0)
