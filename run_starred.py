#!/usr/bin/env python
"""Run `starred` with the safeguards it is missing upstream.

`starred` 4.3.0 (the current release) has two defects that make it a no-op on
accounts with a few thousand stars:

1. It builds its GraphQL client as
   `Client(transport=..., fetch_schema_from_transport=True)` and never sets
   `execute_timeout`, so `gql`'s 10s default applies. A full-schema
   introspection plus ~32 paged starred queries cannot fit in that budget.
   `gql` raises through `anyio.fail_after`, which produces a `TimeoutError`
   with empty `args`, so `str(e)` is the empty string.

2. `starred/starred.py` catches the fetch exception, prints `Error: {e}` and
   `return`s. The process exits 0. A workflow step therefore goes green while
   committing nothing.

This wrapper fixes both: a longer timeout, no introspection, per-page retries
with backoff for transient GitHub errors, and a non-zero exit for any failure.
The command line is passed straight through to `starred`.

See https://github.com/maguowei/starred/issues/99 (silent success) and
https://github.com/maguowei/starred/issues/116 (GraphQL gateway timeouts).
"""

import random
import re
import sys
import time

import gql.client

EXECUTE_TIMEOUT = 180  # seconds, per GraphQL request
MAX_ATTEMPTS = 8
BACKOFF_BASE = 10  # seconds; doubles per attempt up to BACKOFF_CAP
BACKOFF_CAP = 90
DEADLINE = 25 * 60  # seconds; give up rather than sit on the job timeout
_START = time.monotonic()

# GitHub's GraphQL endpoint answers transiently with these. Everything else
# (401, 403, 404, a real GraphQL query error) is a permanent failure.
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# gql formats a transport error as "502, message='Bad Gateway', url=..."
STATUS_RE = re.compile(r"^(\d{3}),")

# gql raises these for a truncated or non-JSON body, which GitHub does under
# load. They carry no status code, so they are matched by type name.
TRANSIENT_TYPES = {
    "TransportProtocolError",
    "TransportClosed",
}

TRANSIENT_MARKERS = (
    "Timeout",
    "Cannot connect",
    "Server disconnected",
    "Connection reset",
    "Not a JSON answer",
    "Server did not return a GraphQL result",
)


def _patch_gql_client():
    """Give every Client a usable timeout and skip the schema introspection."""
    original_init = gql.client.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("execute_timeout", EXECUTE_TIMEOUT)
        kwargs["fetch_schema_from_transport"] = False
        original_init(self, *args, **kwargs)

    gql.client.Client.__init__ = patched_init


def _is_transient(exc):
    message = str(exc).strip()
    status = STATUS_RE.match(message)
    if status:
        return int(status.group(1)) in TRANSIENT_STATUS
    if str(exc) == "":
        return True
    if type(exc).__name__ in TRANSIENT_TYPES:
        return True
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in TRANSIENT_MARKERS)


def _install_retrying_fetch():
    """Wrap the fetcher so each page retries, and every failure exits non-zero."""
    import starred.githubgql as githubgql

    original_fetch = githubgql.GitHubGQL.get_user_starred_by_username

    def fetch_with_retry(self, username, after="", topic_stargazer_count_limit=0):
        last = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                items = original_fetch(
                    self, username, after, topic_stargazer_count_limit
                )
            except Exception as exc:  # noqa: BLE001 - we re-raise or exit
                if not _is_transient(exc):
                    # Never fall through to starred's `except: print(); return`.
                    # That path is what turns a failure into a green CI step.
                    sys.exit(
                        "starred: GraphQL fetch failed: "
                        f"{type(exc).__name__}: {exc or '(empty message)'}"
                    )
                last = exc
                if attempt == MAX_ATTEMPTS:
                    break
                elapsed = time.monotonic() - _START
                if elapsed + BACKOFF_CAP > DEADLINE:
                    sys.exit(
                        "starred: giving up after "
                        f"{elapsed / 60:.0f} min, still failing with "
                        f"{type(exc).__name__}: {exc or '(empty message)'}"
                    )
                # exponential backoff with jitter, so a burst of 502s does not
                # turn into a synchronised retry storm
                ceiling = min(BACKOFF_CAP, BACKOFF_BASE * 2 ** (attempt - 1))
                wait = round(random.uniform(ceiling / 2, ceiling))
                print(
                    f"[retry {attempt}/{MAX_ATTEMPTS}] "
                    f"{type(exc).__name__}: {exc or '(empty message)'} "
                    f"- sleeping {wait}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
                continue

            if not items:
                sys.exit("starred: fetched 0 starred repositories")
            return items

        sys.exit(
            "starred: GraphQL fetch failed after "
            f"{MAX_ATTEMPTS} attempts: {type(last).__name__}: "
            f"{last or '(empty message)'}"
        )

    githubgql.GitHubGQL.get_user_starred_by_username = fetch_with_retry


def main():
    _patch_gql_client()
    _install_retrying_fetch()

    from starred.starred import starred

    starred(standalone_mode=True)


if __name__ == "__main__":
    main()
