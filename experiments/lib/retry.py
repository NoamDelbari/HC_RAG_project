"""Retry logic for API calls with exponential backoff."""

import time
import logging

logger = logging.getLogger(__name__)


def call_with_retry(fn, *args, max_retries: int = 3, base_delay: float = 2.0, **kwargs):
    """Call fn with exponential backoff retry on rate limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = any(kw in err_msg for kw in ("rate", "429", "quota"))
            if is_rate_limit and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
