"""
rate_limiter.py
---------------
Token bucket-based rate limiter for API endpoints.
Supports per-user and global rate limits with configurable windows.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, Tuple


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.
    Allows burst traffic while maintaining average rate limit.
    """

    capacity: int  # Maximum tokens (requests) in bucket
    refill_rate: float  # Tokens added per second
    tokens: float = field(init=False)  # Current tokens available
    last_refill: float = field(init=False)  # Last refill timestamp

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.time()

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from bucket.

        Returns:
            True if tokens available, False otherwise
        """
        self.refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_available(self) -> float:
        """Calculate seconds until next token is available."""
        self.refill()
        if self.tokens >= 1:
            return 0.0

        tokens_needed = 1 - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.
    Supports both per-user and global rate limits.
    """

    def __init__(
        self,
        requests_per_minute: int = 10,
        burst_size: Optional[int] = None,
        cleanup_interval: int = 3600,  # 1 hour
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Average requests allowed per minute per user
            burst_size: Maximum burst size (default: 2x requests_per_minute)
            cleanup_interval: Seconds between cleanup of stale buckets
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or (requests_per_minute * 2)
        self.refill_rate = requests_per_minute / 60.0  # Convert to per-second rate

        # Per-user rate limit tracking
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = Lock()

        # Cleanup tracking
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()

    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create token bucket for given key (user ID, IP, etc)."""
        with self.lock:
            if key not in self.buckets:
                self.buckets[key] = TokenBucket(
                    capacity=self.burst_size, refill_rate=self.refill_rate
                )
            return self.buckets[key]

    def check_rate_limit(self, key: str) -> Tuple[bool, Optional[float]]:
        """
        Check if request is allowed under rate limit.

        Args:
            key: Unique identifier (user_id, IP address, token hash, etc)

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
            - (True, None) if request allowed
            - (False, seconds) if rate limited
        """
        # Periodic cleanup of stale buckets
        self._cleanup_stale_buckets()

        bucket = self._get_bucket(key)

        if bucket.consume():
            return True, None
        else:
            retry_after = bucket.time_until_available()
            return False, retry_after

    def _cleanup_stale_buckets(self) -> None:
        """Remove buckets that haven't been used recently."""
        now = time.time()

        # Only cleanup periodically
        if now - self.last_cleanup < self.cleanup_interval:
            return

        with self.lock:
            # Remove buckets that are full and haven't been accessed in a while
            stale_keys = [
                key
                for key, bucket in self.buckets.items()
                if bucket.tokens >= bucket.capacity
                and (now - bucket.last_refill) > self.cleanup_interval
            ]

            for key in stale_keys:
                del self.buckets[key]

            self.last_cleanup = now

    def reset(self, key: str) -> None:
        """Reset rate limit for a specific key (e.g., for testing or admin override)."""
        with self.lock:
            if key in self.buckets:
                del self.buckets[key]

    def get_remaining(self, key: str) -> int:
        """Get number of requests remaining for key."""
        bucket = self._get_bucket(key)
        bucket.refill()
        return int(bucket.tokens)

    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                "total_users": len(self.buckets),
                "requests_per_minute": self.requests_per_minute,
                "burst_size": self.burst_size,
            }


class GlobalRateLimiter:
    """
    Global rate limiter for entire API.
    Useful for protecting against DDoS or limiting total API load.
    """

    def __init__(self, requests_per_minute: int = 1000):
        """
        Initialize global rate limiter.

        Args:
            requests_per_minute: Total requests allowed per minute globally
        """
        self.limiter = RateLimiter(requests_per_minute=requests_per_minute)
        self.global_key = "__global__"

    def check_rate_limit(self) -> Tuple[bool, Optional[float]]:
        """Check if request is allowed under global rate limit."""
        return self.limiter.check_rate_limit(self.global_key)

    def get_remaining(self) -> int:
        """Get remaining global requests."""
        return self.limiter.get_remaining(self.global_key)


# Create singleton instances for use across the app
# Configure these based on your needs
user_rate_limiter = RateLimiter(
    requests_per_minute=10,  # 10 requests per minute per user
    burst_size=20,  # Allow bursts up to 20 requests
)

global_rate_limiter = GlobalRateLimiter(
    requests_per_minute=1000  # 1000 requests per minute globally
)
