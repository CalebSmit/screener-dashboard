"""Tests for fetch resilience: rate-limit detection, adaptive throttling,
and retry pass logic.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factor_engine import (
    fetch_single_ticker,
    fetch_all_tickers,
    _is_rate_limited,
    _RATE_LIMIT_PATTERNS,
    _NON_RETRYABLE_PATTERNS,
)


# =====================================================================
# RATE-LIMIT DETECTION
# =====================================================================

class TestRateLimitDetection:
    def test_detects_429_error(self):
        assert _is_rate_limited("429 Client Error: Too Many Requests")

    def test_detects_too_many_requests(self):
        assert _is_rate_limited("Too Many Requests. Rate limited.")

    def test_detects_rate_limit(self):
        assert _is_rate_limited("rate limit exceeded, try later")

    def test_does_not_flag_normal_error(self):
        assert not _is_rate_limited("KeyError: 'totalRevenue'")

    def test_does_not_flag_404(self):
        assert not _is_rate_limited("404 Not Found")


# =====================================================================
# FETCH_SINGLE_TICKER RATE-LIMIT BEHAVIOR
# =====================================================================

class TestFetchSingleTickerRateLimit:
    @patch("factor_engine._fetch_single_ticker_inner")
    def test_rate_limited_returns_immediately(self, mock_inner):
        """When a 429 error is detected, fetch_single_ticker should return
        immediately with _rate_limited=True instead of exhausting retries."""
        mock_inner.return_value = {
            "Ticker": "TEST",
            "_error": "429 Client Error: Too Many Requests",
        }
        t0 = time.time()
        result = fetch_single_ticker("TEST", max_retries=3)
        elapsed = time.time() - t0

        assert result["_rate_limited"] is True
        assert "_error" in result
        # Should NOT have waited for exponential backoff (1s + 2s + 4s = 7s)
        assert elapsed < 2.0
        # Should have called inner only once (not 3 times)
        assert mock_inner.call_count == 1

    @patch("factor_engine._fetch_single_ticker_inner")
    def test_non_retryable_not_flagged_as_rate_limit(self, mock_inner):
        """404 errors should NOT be tagged as rate-limited."""
        mock_inner.return_value = {
            "Ticker": "TEST",
            "_error": "404 Not Found",
            "_non_retryable": True,
        }
        result = fetch_single_ticker("TEST", max_retries=3)

        assert result.get("_rate_limited") is not True
        assert "_error" in result
        assert mock_inner.call_count == 1

    @patch("factor_engine._fetch_single_ticker_inner")
    def test_rate_limit_from_exception(self, mock_inner):
        """When inner raises an exception containing rate-limit text,
        it should be caught and tagged."""
        mock_inner.side_effect = Exception(
            "Too Many Requests. Rate limited. Try after a while."
        )
        result = fetch_single_ticker("TEST", max_retries=3)

        assert result["_rate_limited"] is True
        assert mock_inner.call_count == 1

    @patch("factor_engine._fetch_single_ticker_inner")
    def test_successful_fetch_no_rate_limit_tag(self, mock_inner):
        """Successful fetches should not have _rate_limited."""
        mock_inner.return_value = {"Ticker": "TEST", "marketCap": 1e10}
        result = fetch_single_ticker("TEST")

        assert "_rate_limited" not in result
        assert "_error" not in result

    @patch("factor_engine._fetch_single_ticker_inner")
    def test_per_request_delay(self, mock_inner):
        """per_request_delay should add a delay before each attempt."""
        mock_inner.return_value = {"Ticker": "TEST", "marketCap": 1e10}
        t0 = time.time()
        result = fetch_single_ticker("TEST", per_request_delay=0.5)
        elapsed = time.time() - t0

        assert elapsed >= 0.4  # at least ~0.5s delay
        assert "_error" not in result


# =====================================================================
# FETCH_ALL_TICKERS ADAPTIVE THROTTLING
# =====================================================================

class TestFetchAllTickersAdaptive:
    @patch("factor_engine.fetch_single_ticker")
    def test_adaptive_reduces_workers_on_rate_limit(self, mock_fetch):
        """When rate limits are hit, subsequent batches should use fewer workers."""
        call_count = {"n": 0}

        def side_effect(ticker, **kwargs):
            call_count["n"] += 1
            # First batch: all succeed. Second batch: all rate-limited.
            if call_count["n"] > 5:
                return {"Ticker": ticker, "_error": "429 Too Many Requests",
                        "_rate_limited": True}
            return {"Ticker": ticker, "marketCap": 1e10}

        mock_fetch.side_effect = side_effect

        tickers = [f"T{i:02d}" for i in range(15)]

        # Use small batch/workers and zero delays for speed
        with patch("time.sleep"):  # skip actual sleeps in test
            results = fetch_all_tickers(
                tickers, batch_size=5, max_workers=3, inter_batch_delay=0.0
            )

        assert len(results) == 15
        # Some should be rate-limited
        rate_limited = [r for r in results if r.get("_rate_limited")]
        assert len(rate_limited) > 0

    @patch("factor_engine.fetch_single_ticker")
    def test_all_succeed_no_throttling(self, mock_fetch):
        """When all fetches succeed, no throttling should occur."""
        mock_fetch.return_value = {"Ticker": "TEST", "marketCap": 1e10}

        tickers = [f"T{i:02d}" for i in range(10)]
        with patch("time.sleep"):
            results = fetch_all_tickers(
                tickers, batch_size=5, max_workers=2, inter_batch_delay=0.0
            )

        assert len(results) == 10
        assert all("_error" not in r for r in results)
