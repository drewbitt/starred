#!/usr/bin/env python
"""Fetch once and render with starred 4.3.0; GitHub Actions commits changed files.

Upstream swallows fetch failures and defaults to a 10s GraphQL timeout.
Fetch outside its CLI so errors propagate. Keep its Markdown renderer unchanged.
See maguowei/starred#99 (silent success) and #116 (GraphQL timeouts).
"""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import ClientConnectionError
from gql import Client
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.exceptions import (
    TransportClosed,
    TransportProtocolError,
    TransportServerError,
)
from starred.githubgql import GitHubGQL
import starred.starred as cli

MAX_ATTEMPTS = 8
DEADLINE = 25 * 60
REQUEST_TIMEOUT = 180
OUTPUTS = (("README.md", False), ("topics.md", True))
REPO_ENTRY = re.compile(r"^- \[[^\]]*\]\(https://github\.com/")


def is_transient(exc):
    if isinstance(exc, TransportServerError):
        return exc.code in {429, 500, 502, 503, 504}
    return isinstance(
        exc, (TimeoutError, ClientConnectionError, TransportClosed, TransportProtocolError)
    )


def execute_with_retry(client, execute, expires, *args, **kwargs):
    """Retry the individual query, not the recursive pagination above it."""
    for attempt in range(MAX_ATTEMPTS):
        remaining = expires - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("starred fetch exceeded its 25-minute deadline")
        client.execute_timeout = min(REQUEST_TIMEOUT, remaining)
        try:
            return execute(*args, **kwargs)
        except Exception as exc:
            if not is_transient(exc) or attempt == MAX_ATTEMPTS - 1:
                raise
            delay = min(
                random.uniform(5, min(90, 10 * 2**attempt)),
                max(0, expires - time.monotonic()),
            )
            print(
                f"Retry {attempt + 1}/{MAX_ATTEMPTS}: {type(exc).__name__}: "
                f"{exc}; waiting {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


class StarClient(GitHubGQL):
    """Keep upstream pagination, replacing its insecure/default client setup."""

    def __init__(self, token):
        self.client = Client(
            transport=AIOHTTPTransport(
                url=self.API_URL,
                headers={"Authorization": f"Bearer {token}"},
                ssl=True,
            ),
            fetch_schema_from_transport=False,
            execute_timeout=REQUEST_TIMEOUT,
        )


def fetch(username, token):
    gh = StarClient(token)
    execute = gh.client.execute
    expires = time.monotonic() + DEADLINE
    gh.client.execute = lambda *a, **kw: execute_with_retry(
        gh.client, execute, expires, *a, **kw
    )
    return gh.get_user_starred_by_username(username, topic_stargazer_count_limit=500)


def check_count(stars, previous=Path("README.md")):
    if not stars:
        raise ValueError("fetched 0 starred repositories")
    old_count = (
        sum(
            bool(REPO_ENTRY.match(line))
            for line in previous.read_text(encoding="utf-8").splitlines()
        )
        if previous.exists() else 0
    )
    if len(stars) < old_count * 0.9:
        raise ValueError(
            f"fetched {len(stars)} repositories; README already lists {old_count}; "
            "refusing a potentially truncated update"
        )


def render(username, token, stars, topic):
    snapshot = SimpleNamespace(get_user_starred_by_username=lambda *a, **kw: stars)
    output = StringIO()
    # The upstream renderer always fetches: inject the already-validated snapshot.
    with patch.object(cli, "GitHubGQL", lambda token: snapshot), redirect_stdout(output):
        cli.starred.main(
            args=["--username", username, "--token", token, "--sort"]
            + (["--topic"] if topic else []),
            standalone_mode=False,
        )
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    token = os.environ["GITHUB_TOKEN"]
    stars = fetch(args.username, token)
    check_count(stars)
    # Finish both renders before modifying either file.
    rendered = [(name, render(args.username, token, stars, topic)) for name, topic in OUTPUTS]
    for name, content in rendered:
        Path(name).write_text(content, encoding="utf-8")
    print(f"Rendered both files from {len(stars)} starred repositories", flush=True)


if __name__ == "__main__":
    main()
