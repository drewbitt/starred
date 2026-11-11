import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gql.transport.exceptions import TransportProtocolError, TransportServerError
from starred.githubgql import Repository

import run_starred as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.stars = [
            Repository("owner/repo", "A <thing>", "Python",
                       "https://github.com/owner/repo", 1000, False, ["tools"]),
        ]

    def test_classifies_real_exception_types(self):
        cases = [
            (TimeoutError(), True),
            (TransportProtocolError("not JSON"), True),
            (TransportServerError("unavailable", code=502), True),
            (TransportServerError("unauthorized", code=401), False),
            (TransportServerError("forbidden", code=403), False),
            (ValueError(), False),
        ]
        for error, expected in cases:
            with self.subTest(error=repr(error)):
                self.assertEqual(runner.is_transient(error), expected)

    def retry(self, execute, expires=1000):
        return runner.execute_with_retry(SimpleNamespace(), execute, expires, "query")

    @patch.object(runner.time, "sleep")
    @patch.object(runner.time, "monotonic", return_value=0)
    def test_retry_then_success(self, clock, sleep):
        execute = Mock(side_effect=[TimeoutError(), {"data": "ok"}])
        self.assertEqual(self.retry(execute), {"data": "ok"})
        self.assertEqual(execute.call_count, 2)
        sleep.assert_called_once()

    @patch.object(runner.time, "sleep")
    @patch.object(runner.time, "monotonic", return_value=0)
    def test_retry_exhaustion_and_fatal_errors(self, clock, sleep):
        execute = Mock(side_effect=TimeoutError())
        with self.assertRaises(TimeoutError):
            self.retry(execute)
        self.assertEqual(execute.call_count, runner.MAX_ATTEMPTS)
        execute = Mock(side_effect=TransportServerError("forbidden", code=403))
        with self.assertRaises(TransportServerError):
            self.retry(execute)
        execute.assert_called_once()

    @patch.object(runner.time, "monotonic", return_value=1000)
    def test_deadline_prevents_next_request(self, clock):
        execute = Mock()
        with self.assertRaises(TimeoutError):
            self.retry(execute)
        execute.assert_not_called()

    @patch.object(runner.time, "sleep")
    def test_pagination_retries_only_the_failed_page(self, sleep):
        def page(cursor, more, name):
            return {"user": {"starredRepositories": {
                "pageInfo": {"endCursor": cursor, "hasNextPage": more},
                "nodes": [{
                    "nameWithOwner": name, "description": "", "url": "https://github.com/" + name,
                    "stargazerCount": 1, "isPrivate": False, "languages": {"edges": []},
                    "repositoryTopics": {"nodes": []},
                }],
            }}}

        client = SimpleNamespace(execute=Mock(side_effect=[
            page("next", True, "o/first"),
            TransportServerError("bad gateway", code=502),
            page("end", False, "o/second"),
        ]))
        execute = client.execute
        with patch.object(runner, "Client", return_value=client):
            stars = runner.fetch("user", "fake-token")
        self.assertEqual([star.name for star in stars], ["o/first", "o/second"])
        cursors = [call.kwargs["variable_values"]["after"] for call in execute.call_args_list]
        self.assertEqual(cursors, ["", "next", "next"])

    def test_client_security_and_timeouts(self):
        gh = runner.StarClient("fake-token")
        self.assertIs(gh.client.transport.ssl, True)
        self.assertFalse(gh.client.fetch_schema_from_transport)
        self.assertEqual(gh.client.execute_timeout, runner.REQUEST_TIMEOUT)

    def test_floor_counts_repos_not_toc(self):
        with TemporaryDirectory() as directory:
            previous = Path(directory) / "README.md"
            runner.check_count(self.stars, previous)
            previous.write_text("- [Python](#python)\n" + (
                "- [o/r](https://github.com/o/r) - desc\n" * 10
            ))
            runner.check_count(self.stars * 9, previous)
            with self.assertRaisesRegex(ValueError, "truncated"):
                runner.check_count(self.stars * 8, previous)
            with self.assertRaisesRegex(ValueError, "0 starred"):
                runner.check_count([], previous)

    def test_render_preserves_categories_escaping_and_footer(self):
        original = runner.cli.GitHubGQL
        for topic in (False, True):
            output = runner.render("user", "fake-token", self.stars, topic)
            self.assertIn("- [owner/repo](https://github.com/owner/repo) - A &lt;thing&gt;", output)
            self.assertIn("## tools" if topic else "## Python", output)
            self.assertIn("## License", output)
            self.assertIs(runner.cli.GitHubGQL, original)

    @patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token"})
    @patch("sys.argv", ["run_starred.py", "--username", "user"])
    def test_main_fetches_once_before_writing(self):
        with TemporaryDirectory() as directory:
            old = Path.cwd()
            os.chdir(directory)
            try:
                with patch.object(runner, "fetch", return_value=self.stars) as fetch:
                    runner.main()
                    fetch.assert_called_once_with("user", "fake-token")
                for name, _ in runner.OUTPUTS:
                    self.assertTrue(Path(name).exists())
                with patch.object(runner, "fetch", side_effect=RuntimeError("failure")):
                    before = Path("README.md").read_bytes()
                    with self.assertRaises(RuntimeError):
                        runner.main()
                    self.assertEqual(Path("README.md").read_bytes(), before)
                with patch.object(runner, "fetch", return_value=self.stars), patch.object(
                    runner, "render", side_effect=["partial render", RuntimeError("render failed")]
                ):
                    with self.assertRaises(RuntimeError):
                        runner.main()
                    self.assertEqual(Path("README.md").read_bytes(), before)
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
