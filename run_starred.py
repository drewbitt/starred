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

With `--both` it also renders every output file from a single fetch. The
starred-repository query is the slow and flaky part (~32 pages, and GitHub
answers a fair number of those with a 502), so fetching once instead of once
per file halves both the runtime and the exposure to that flakiness.

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
DEADLINE = 25 * 60  # seconds for the whole run, under the job's timeout
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

# One entry per generated file. `--both` renders all of them from one fetch.
OUTPUTS = (
    {
        "filename": "README.md",
        "message": (
            "awesome-stars category by language update by github actions "
            "cron, created by starred"
        ),
        "extra": (),
    },
    {
        "filename": "topics.md",
        "message": (
            "awesome-stars category by topic update by github actions "
            "cron, created by starred"
        ),
        "extra": ("--topic", "--topic_limit", "500"),
    },
)

# A repository line is "- [owner/name](https://github.com/owner/name) - desc";
# a table-of-contents line is "- [Category](#category)".
REPO_ENTRY_RE = re.compile(r"^- \[[^\]]*\]\(https://github\.com/")

# Generated markdown is compared against this file to catch a fetch that
# stopped paginating early. Missing file means a first run, so no floor.
FLOOR_FILE = "README.md"
FLOOR_RATIO = 0.9

_fetched = {}


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


def _count_repo_entries(path):
    """Number of repository lines already committed, or None on a first run."""
    try:
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if REPO_ENTRY_RE.match(line))
    except FileNotFoundError:
        return None


def _check_floor(count):
    """Refuse to overwrite the list with a visibly truncated one."""
    previous = _count_repo_entries(FLOOR_FILE)
    if previous is None:
        return
    if count < previous * FLOOR_RATIO:
        sys.exit(
            f"starred: fetched {count} repositories but {FLOOR_FILE} already "
            f"lists {previous}; refusing to write a truncated list"
        )
    print(
        f"starred: fetched {count} repositories "
        f"({FLOOR_FILE} lists {previous})",
        file=sys.stderr,
        flush=True,
    )


def _install_retrying_fetch():
    """Memoise the fetch, retry each page, and exit non-zero on any failure."""
    import starred.githubgql as githubgql

    original_fetch = githubgql.GitHubGQL.get_user_starred_by_username

    def fetch_with_retry(self, username, after="", topic_stargazer_count_limit=0):
        key = (username, after, topic_stargazer_count_limit)
        if key in _fetched:
            return _fetched[key]

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

            _fetched[key] = items
            # `after` is empty only on the outermost page, whose result is the
            # whole list -- the only place the floor check makes sense.
            if after == "":
                _check_floor(len(items))
            return items

        sys.exit(
            "starred: GraphQL fetch failed after "
            f"{MAX_ATTEMPTS} attempts: {type(last).__name__}: "
            f"{last or '(empty message)'}"
        )

    githubgql.GitHubGQL.get_user_starred_by_username = fetch_with_retry


def _prepare():
    _patch_gql_client()
    _install_retrying_fetch()


def run_one(argv):
    """One `starred` invocation, exactly as the CLI would have run it."""
    _prepare()
    from starred.starred import starred

    saved_stdout = sys.stdout
    try:
        starred.main(args=argv or None, standalone_mode=True)
    finally:
        sys.stdout = saved_stdout


def run_both(common_args):
    """Render every file in OUTPUTS from a single starred-repository fetch."""
    _prepare()
    from starred.starred import starred

    for spec in OUTPUTS:
        args = list(common_args) + ["--filename", spec["filename"]]
        args += ["--message", spec["message"]]
        args += list(spec["extra"])
        print(f"starred: writing {spec['filename']}", file=sys.stderr, flush=True)
        # starred() leaves sys.stdout pointing at the BytesIO it renders into,
        # so the next iteration would inherit a dead buffer.
        saved_stdout = sys.stdout
        try:
            starred.main(args=args, standalone_mode=False)
        finally:
            sys.stdout = saved_stdout


def main():
    if sys.argv[1:2] == ["--both"]:
        run_both(sys.argv[2:])
    else:
        run_one([])


if __name__ == "__main__":
    main()
