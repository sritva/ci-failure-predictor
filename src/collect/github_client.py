"""GitHub API client with authentication, rate-limit backoff, retry, and pagination."""

import os
import time
from typing import Any, Dict, Generator, Optional, Union
from dotenv import load_dotenv
import requests

load_dotenv()


class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""
    pass


class GitHubAuthError(GitHubAPIError):
    """Raised on authentication failures (HTTP 401/403)."""
    pass


class GitHubNotFoundError(GitHubAPIError):
    """Raised when a requested resource is not found (HTTP 404)."""
    pass


class GitHubClient:
    """Client for interacting with the GitHub REST API."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        timeout: int = 30,
        rate_limit_threshold: Optional[int] = None,
        backoff_delay: float = 1.5,
    ) -> None:
        """Initialize GitHub client.

        Args:
            token: GitHub personal access token. If None, loaded from GITHUB_TOKEN env var.
            timeout: Request timeout in seconds.
            rate_limit_threshold: Remaining requests threshold to trigger sleep until reset (defaults to 50 with token, 5 unauthenticated).
            backoff_delay: Backoff delay in seconds for retrying 5xx/connection errors.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.timeout = timeout
        if rate_limit_threshold is not None:
            self.rate_limit_threshold = rate_limit_threshold
        else:
            self.rate_limit_threshold = 50 if self.token else 5
        self.backoff_delay = backoff_delay
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ci-failure-predictor",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _resolve_url(self, url: str) -> str:
        """Ensure the URL is a full GitHub API URL."""
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return f"{self.BASE_URL.rstrip('/')}/{url.lstrip('/')}"

    def _check_rate_limit(self, response: requests.Response) -> None:
        """Check rate-limit headers and sleep if remaining quota is below threshold."""
        remaining_header = response.headers.get("X-RateLimit-Remaining")
        reset_header = response.headers.get("X-RateLimit-Reset")

        if remaining_header is not None and reset_header is not None:
            try:
                remaining = int(remaining_header)
                reset_timestamp = int(reset_header)
            except (ValueError, TypeError):
                return

            if remaining < self.rate_limit_threshold:
                now = int(time.time())
                sleep_seconds = max(0, reset_timestamp - now) + 5
                print(
                    f"[GitHubClient] Rate limit low ({remaining} remaining < {self.rate_limit_threshold}). "
                    f"Sleeping for {sleep_seconds}s until reset..."
                )
                time.sleep(sleep_seconds)

    def _handle_client_errors(self, response: requests.Response) -> None:
        """Inspect 4xx status codes and raise specific exceptions."""
        status_code = response.status_code
        if 400 <= status_code < 500:
            try:
                error_body = response.json()
                message = error_body.get("message", response.text)
            except Exception:
                message = response.text or "No error message provided"

            if status_code == 401:
                raise GitHubAuthError(
                    f"GitHub authentication failed (HTTP 401): {message}. "
                    "Verify GITHUB_TOKEN in your environment or .env file."
                )
            if status_code == 403:
                raise GitHubAuthError(
                    f"GitHub request forbidden or rate limit exceeded (HTTP 403): {message}"
                )
            if status_code == 404:
                raise GitHubNotFoundError(
                    f"GitHub resource not found (HTTP 404) at {response.url}: {message}"
                )
            raise GitHubAPIError(
                f"GitHub API client error (HTTP {status_code}) at {response.url}: {message}"
            )

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """Send a GET request with authentication, retries, and rate limit handling.

        Args:
            url: API endpoint (relative or absolute).
            params: Optional query parameters.

        Returns:
            requests.Response object.

        Raises:
            GitHubAuthError: On HTTP 401 or 403.
            GitHubNotFoundError: On HTTP 404.
            GitHubAPIError: On other 4xx errors or persistent 5xx/connection errors.
        """
        full_url = self._resolve_url(url)
        headers = self._get_headers()

        # Retry once on 5xx or connection errors
        for attempt in range(2):
            try:
                response = self.session.get(
                    full_url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )

                if response.status_code >= 500:
                    if attempt == 0:
                        print(
                            f"[GitHubClient] Server error (HTTP {response.status_code}) on {full_url}. "
                            f"Retrying once after {self.backoff_delay}s..."
                        )
                        time.sleep(self.backoff_delay)
                        continue
                    response.raise_for_status()

                self._check_rate_limit(response)
                self._handle_client_errors(response)
                return response

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt == 0:
                    print(
                        f"[GitHubClient] Connection error ({exc}) on {full_url}. "
                        f"Retrying once after {self.backoff_delay}s..."
                    )
                    time.sleep(self.backoff_delay)
                    continue
                raise GitHubAPIError(f"Request failed after retry on {full_url}: {exc}") from exc
            except requests.exceptions.HTTPError as exc:
                raise GitHubAPIError(f"HTTP error on {full_url}: {exc}") from exc

        raise GitHubAPIError(f"Request failed on {full_url}")

    def get_paginated(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        item_key: Optional[str] = None,
    ) -> Generator[Any, None, None]:
        """Generator that follows GitHub's Link header and yields items across all pages.

        Args:
            url: API endpoint to start from.
            params: Query parameters for initial request.
            item_key: Optional key to extract list from dictionary response.

        Yields:
            Individual items from list responses or from dictionary list fields.
        """
        current_url = url
        current_params = params

        while current_url:
            response = self.get(current_url, params=current_params)
            data = response.json()

            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                if item_key and item_key in data and isinstance(data[item_key], list):
                    for item in data[item_key]:
                        yield item
                else:
                    # Look for known list fields in GitHub API responses
                    list_found = False
                    for key in ("workflow_runs", "items", "pull_requests", "files"):
                        if key in data and isinstance(data[key], list):
                            for item in data[key]:
                                yield item
                            list_found = True
                            break
                    if not list_found:
                        yield data

            # Follow Link header's next relation
            next_url = response.links.get("next", {}).get("url")
            if next_url:
                current_url = next_url
                current_params = None  # Next URL already includes pagination query parameters
            else:
                current_url = None
