"""Portable CI checker regression tests; fixtures live only in temporary Git repos."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "ci_source", Path(__file__).resolve().parents[1] / "scripts" / "ci_source.py",
)
assert SPEC and SPEC.loader
ci_source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci_source)


class SourceChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="crucible ci source ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git("init", "-q")

    def git(self, *args: str) -> bytes:
        # Do not let a caller's temporary index or Git directory leak into fixtures.
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"}}
        return subprocess.run(["git", "-C", str(self.root), *args], check=True,
                              capture_output=True, env=environment).stdout

    def write(self, name: str, data: str | bytes, *, tracked: bool = True) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        if tracked:
            self.git("add", "--", name)

    def check(self, **kwargs):
        # The checker intentionally inherits Git's index choice in production.
        # Unit fixtures must select their own default index instead.
        from unittest.mock import patch
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"}}
        with patch.dict(os.environ, environment, clear=True):
            return ci_source.check_repository(self.root, **kwargs)

    def assert_problem(self, text: str) -> None:
        errors, _ = self.check()
        self.assertTrue(any(text in error for error in errors), errors)

    def test_valid_files_are_checked_without_imports_or_writes(self):
        self.write("scripts/unusual name ' quote.py", "import dependency_not_installed\n")
        self.write("evidence.json", '{"measurement": [1, null, true]}')
        self.write("README.md", "[data](evidence.json#measurement)\n")
        self.write("figures/plot.png", b"\x89PNG\r\n\x1a\n")
        self.write("untracked-broken.py", "syntax is (broken", tracked=False)
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        errors, counts = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(counts, {"tracked": 4, "python": 1, "json": 1, "markdown": 1, "local_links": 1})
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(list(self.root.rglob("__pycache__")))

    def test_python_invalid_syntax_is_not_imported(self):
        self.write("broken.py", "def missing_colon()\n    pass\n")
        self.assert_problem("Python syntax")

    def test_json_rejects_truncation_nonfinite_and_duplicate_keys(self):
        for text in ['{"a":', '{"a": NaN}', '{"a": Infinity}', '{"a": 1, "a": 2}']:
            with self.subTest(text=text):
                self.write("broken.json", text)
                self.assert_problem("JSON:")

    def test_tracked_file_must_exist(self):
        self.write("missing.py", "pass\n")
        (self.root / "missing.py").unlink()
        self.assert_problem("tracked file is missing")

    def test_local_links_must_be_tracked_even_when_the_file_exists(self):
        self.write("README.md", "[missing](missing.md)\n")
        self.write("missing.md", "present but untracked", tracked=False)
        self.assert_problem("does not name a tracked file")

    def test_markdown_inline_reference_image_and_escaped_paths(self):
        self.write("docs/a (b).md", "content")
        self.write("figures/plot.png", b"\x89PNG")
        self.write("docs/links.md", """[balanced](a%20(b).md#heading)
[escaped](a\\ (b).md)
[angle](<a (b).md> "title")
![image](../figures/plot.png)
[reference][doc]
[doc]: <a (b).md> 'title'
""")
        errors, counts = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(counts["local_links"], 5)

    def test_markdown_ignores_code_remote_urls_and_anchors(self):
        self.write("README.md", """[remote](https://invalid.example.test/never-requested)
[mail](mailto:someone@example.test) [anchor](#not-validated)
[remote-relative](//invalid.example.test/path)
`[example](missing.md)`
```markdown
[example](missing.md)
```
~~~
[example](missing.md)
~~~
    [example](missing.md)
""")
        errors, counts = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(counts["local_links"], 0)

    def test_reference_definition_is_checked(self):
        self.write("README.md", "[doc]: missing.md\n")
        self.assert_problem("does not name a tracked file")

    def test_local_links_cannot_escape_repository(self):
        for target in ["../outside.md", "%2e%2e/outside.md", "/tmp/outside.md"]:
            with self.subTest(target=target):
                self.write("README.md", f"[bad]({target})")
                errors, _ = self.check()
                self.assertEqual(len(errors), 1, errors)

    def test_invalid_local_link_reports_failure_without_crashing(self):
        # NUL cannot occur in a tracked filename, so it fails before path resolution.
        self.write("README.md", "[bad](file%00.md)\n")
        self.assert_problem("does not name a tracked file")

    def test_conflicts_fail_but_heading_underlines_are_valid(self):
        self.write("README.md", "A heading\n=======\n")
        self.assertEqual(self.check()[0], [])
        self.write("broken.rs", "<" * 7 + " HEAD\nleft\n" + "=" * 7 + "\nright\n" + ">" * 7 + " branch\n")
        self.assert_problem("unresolved merge-conflict marker")

    def test_machine_paths_fail_in_scripts_but_historical_docs_are_allowed(self):
        machine_path = "/".join(["", "home", "fixture-user", "model"])
        self.write("docs/report.md", f"Historical measurement used `{machine_path}`.\n")
        self.write("docs/evidence.json", '{"historical_path": "' + machine_path + '"}')
        self.assertEqual(self.check()[0], [])
        self.write("run.py", f"MODEL = {machine_path!r}\n")
        self.assert_problem("machine-specific user-home path")

    def test_orphan_conflict_separator_fails_outside_markdown(self):
        self.write("config.toml", "=" * 7 + "\n")
        self.assert_problem("unresolved merge-conflict marker")

    def test_generic_placeholder_paths_are_allowed(self):
        path = "/".join(["", "home", "user", "export", "120m"])
        self.write("fixture.rs", f'assert!(validate_model_id("{path}").is_err());\n')
        self.assertEqual(self.check()[0], [])

    def test_machine_paths_are_checked_in_workflow_configuration(self):
        for prefix in [["", "mnt", "c", "Users"], ["C:", "Users"]]:
            with self.subTest(prefix=prefix):
                path = "/".join(prefix + ["fixture-user", "model"])
                self.write(".github/workflows/check.yml", f"env:\n  MODEL: {path}\n")
                self.assert_problem("machine-specific user-home path")

    def test_machine_home_roots_and_escaped_windows_literals_are_rejected(self):
        paths = [
            "/".join(["", "home", "fixture-user"]),
            "/".join(["", "mnt", "c", "Users", "fixture-user"]),
            "\\".join(["C:", "Users", "fixture-user"]),
            "\\".join(["C:", "Users", "fixture user", "model"]),
        ]
        for path in paths:
            with self.subTest(path=path):
                # repr models an escaped Python literal, including doubled slashes.
                self.write("run.py", f"MODEL = {path!r}\n")
                self.assert_problem("machine-specific user-home path")

    def test_generated_model_and_profiler_artifacts_fail(self):
        for name in ["scripts/__pycache__/tool.pyc", "export/model.safetensors", "baseline-src/file.rs", "runs/capture.nsys-rep", "engine/target/.rustc_info.json"]:
            with self.subTest(name=name):
                self.write(name, "junk")
                self.assert_problem("tracked generated/model/profiler artifact")

    def test_large_artifact_fails_without_reading_it(self):
        self.write("accidental.data", b"")
        with (self.root / "accidental.data").open("wb") as stream:
            stream.truncate(ci_source.MAX_TRACKED_BYTES + 1)
        self.assert_problem("10 MiB artifact limit")

    @unittest.skipIf(os.name == "nt", "Windows disallows newline filenames")
    def test_filename_newlines_are_not_split(self):
        self.write("scripts/line\nbreak.py", "pass\n")
        errors, counts = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(counts["python"], 1)

    def test_clean_option_rejects_changes_and_untracked_files(self):
        self.write("good.py", "pass\n")
        self.git("-c", "user.name=CI test", "-c", "user.email=ci@example.test", "commit", "-qm", "fixture")
        self.assertEqual(self.check(check_clean=True)[0], [])
        self.write("untracked.py", "pass\n", tracked=False)
        self.assertEqual(self.check()[0], [])
        errors, _ = self.check(check_clean=True)
        self.assertTrue(any("not clean" in error for error in errors), errors)
        (self.root / "untracked.py").unlink()
        self.write("good.py", "print('changed')\n", tracked=False)
        self.assertEqual(self.check()[0], [])
        self.assertTrue(any("not clean" in error for error in self.check(check_clean=True)[0]))

    @unittest.skipIf(os.name == "nt", "Windows symlinks need additional privileges")
    def test_symlink_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory(prefix="crucible ci external ") as outside:
            target = Path(outside) / "outside.md"
            target.write_text("outside")
            (self.root / "escape.md").symlink_to(target)
            self.git("add", "--", "escape.md")
            self.write("README.md", "[outside](escape.md)\n")
            self.assert_problem("tracked symlink leaves the repository")
            self.assert_problem("local link target is missing or leaves the repository")

    @unittest.skipIf(os.name == "nt", "Windows symlinks need additional privileges")
    def test_link_to_cyclic_symlink_reports_failure_without_crashing(self):
        (self.root / "cycle.md").symlink_to("cycle.md")
        self.git("add", "--", "cycle.md")
        self.write("README.md", "[cycle](cycle.md)\n")
        self.assert_problem("local link target is invalid or cannot be resolved")


if __name__ == "__main__":
    unittest.main()
