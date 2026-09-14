"""GPU CI orchestration contracts that require neither CUDA nor model assets."""
import importlib.util
import io
import json
import os
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("ci_gpu", Path(__file__).resolve().parents[1] / "scripts/ci_gpu.py")
ci = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ci
SPEC.loader.exec_module(ci)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.assets = {key: f"/configured assets/{key}; literal" for key in ci.ASSETS}

    def plan(self, mode):
        return ci.suite_plan(mode, self.assets, "/build/release/llm-engine", "/scratch")

    def test_smoke_is_model_only_and_uses_bounded_fuzz(self):
        steps = self.plan("smoke")
        self.assertEqual(len(steps), 10)
        self.assertTrue(all(step.server is None for step in steps))
        packed = next(step for step in steps if "gpu-packed-prefill-check" in step.argv)
        self.assertEqual(packed.argv[-2:], ("--fuzz", "4"))
        self.assertIn(self.assets[ci.ASSETS[0]], packed.argv)
        self.assertTrue(all(ci.ASSETS[1] not in str(step.argv) for step in steps))

    def test_full_reference_precedes_default_service_and_requires_sdks(self):
        steps = self.plan("full")
        service = [step for step in steps if step.server]
        self.assertEqual([step.server for step in service], ["reference"] + ["default"] * 5 + ["pressure"])
        self.assertIn("--record-reference", service[0].argv)
        self.assertIn("--reference", service[1].argv)
        self.assertEqual(next(step for step in service if "test_openai.py" in " ".join(step.argv)).argv[-1], self.assets[ci.ASSETS[3]])
        self.assertEqual(next(step for step in service if "test_anthropic.py" in " ".join(step.argv)).argv[-1], self.assets[ci.ASSETS[4]])
        ce = [step for step in steps if step.ce_context is not None]
        self.assertEqual([step.ce_context for step in ce], [32, 32, 256, 256, 941, 941])
        self.assertEqual([step.env["CRUCIBLE_PREFILL_ATTN"] for step in ce], ["reference", "exact-q4-k64"] * 3)
        self.assertTrue(all("--locked" in step.argv for step in steps if step.argv[0] == "cargo"))

    def test_sanitizer_is_explicit_and_errors_fail(self):
        steps = self.plan("sanitizer")
        self.assertEqual(len(steps), 3)
        for step in steps[1:]:
            self.assertEqual(step.argv[:5], ("compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99"))
        self.assertEqual(steps[-1].env, {"CRUCIBLE_PREFILL_ATTN": "exact-q4-k64"})

    def test_list_commands_never_runs_preflight_or_processes(self):
        with patch.object(ci, "preflight") as preflight, patch.object(ci, "execute") as execute:
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(ci.main(["--mode", "full", "--list-commands"]), 0)
            self.assertIn("Plan only", output.getvalue())
            preflight.assert_not_called()
            execute.assert_not_called()

    def test_runtime_defaults_ignore_inherited_experiments(self):
        env = ci.runtime_env({"PATH": "tools", "CRUCIBLE_PREFILL_ATTN": "q1-k16",
                              "CRUCIBLE_BATCHED_PREFILL": "0", "CRUCIBLE_MODEL_PATH": "private"})
        self.assertFalse(any(key.startswith("CRUCIBLE_") for key in env))
        self.assertEqual(env["PATH"], "tools")
        self.assertEqual(ci.server_env("default", env), env)
        self.assertEqual(ci.server_env("reference", env)["CRUCIBLE_BATCHED_PREFILL"], "0")

    def test_asset_redaction_handles_nested_paths(self):
        self.assertEqual(ci.redact("/private/model /private", {"MODEL": "/private/model", "BASE": "/private"}), "<MODEL> <BASE>")


class PreflightTests(unittest.TestCase):
    def test_missing_assets_fail_without_any_execution(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(ci, "execute") as execute:
            with redirect_stderr(io.StringIO()) as error:
                self.assertEqual(ci.main([]), 1)
            self.assertIn("Set CRUCIBLE_MODEL_PATH", error.getvalue())
            execute.assert_not_called()

    def test_missing_full_sdk_is_not_silently_skipped(self):
        source = {key: str(Path(tempfile.gettempdir()) / key) for key in ci.ASSETS[:-1]}
        with self.assertRaisesRegex(ci.CheckFailure, "CRUCIBLE_ANTHROPIC_PYTHON"):
            ci.required_assets("full", source)

    def test_relative_assets_rejected(self):
        with self.assertRaisesRegex(ci.CheckFailure, "absolute"):
            ci.required_assets("smoke", {ci.ASSETS[0]: "relative/model"})

    def test_sanitizer_rejects_wsl_before_accessing_assets(self):
        with (patch.object(ci.platform, "system", return_value="Linux"),
              patch.object(ci.platform, "release", return_value="6.6-microsoft-standard-WSL2"),
              self.assertRaisesRegex(ci.CheckFailure, "native Linux")):
            ci.preflight("sanitizer", {}, {}, Path(tempfile.gettempdir()))

    def test_missing_tool_fails_before_cuda_initialization(self):
        with patch.object(ci.platform, "system", return_value="Linux"), patch.object(ci, "validate_assets"), patch.object(ci.shutil, "which", return_value=None), patch.object(ci, "cuda_preflight") as cuda:
            with self.assertRaisesRegex(ci.CheckFailure, "nvidia-smi"):
                ci.preflight("smoke", {}, {}, Path(tempfile.gettempdir()))
            cuda.assert_not_called()

    def test_wrong_model_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text(json.dumps({"n_layer": 1}))
            (path / "model.safetensors").write_bytes(b"placeholder")
            with self.assertRaisesRegex(ci.CheckFailure, "120M model geometry"):
                ci.validate_assets("smoke", {ci.ASSETS[0]: str(path)})

    def test_full_rejects_directory_name_incompatible_with_tui_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint"
            path.mkdir()
            (path / "config.json").write_text(json.dumps({
                "n_layer": 12, "n_head": 12, "n_kv_head": 3, "n_embd": 768,
                "block_size": 1024, "vocab_size": 50304,
            }))
            (path / "model.safetensors").write_bytes(b"placeholder")
            with self.assertRaisesRegex(ci.CheckFailure, "directory name"):
                ci.validate_assets("full", {ci.ASSETS[0]: str(path)})


@unittest.skipUnless(os.name == "posix", "GPU subprocess groups require Linux")
class ProcessTests(unittest.TestCase):
    def test_server_rejects_port_with_live_listener(self):
        with socket.socket() as listener, tempfile.TemporaryDirectory() as folder:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with patch.object(ci.subprocess, "Popen") as spawn, self.assertRaises(OSError):
                with ci.server("default", {}, "unused", port, {}, Path(folder)):
                    self.fail("An occupied port must never reach server readiness.")
            spawn.assert_not_called()

    def test_server_reuses_port_after_prior_connections_close(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                accepted, _ = listener.accept()
                with accepted:
                    # Active close on the server puts its local port in
                    # TIME_WAIT, as can happen after the real TUI suite.
                    accepted.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
        assets = {key: "/configured/" + key for key in ci.ASSETS}
        with (tempfile.TemporaryDirectory() as folder,
              patch.object(ci.subprocess, "Popen", side_effect=RuntimeError("spawn reached")),
              self.assertRaisesRegex(RuntimeError, "spawn reached")):
            with ci.server("default", assets, "unused", port, {}, Path(folder)):
                self.fail("The mocked spawn must stop execution.")

    def test_failed_command_redacts_output_and_fails(self):
        with (redirect_stdout(io.StringIO()) as output,
              self.assertRaisesRegex(ci.CheckFailure, "exit code 7")):
            ci.run_command((sys.executable, "-c", "print('/private/asset'); raise SystemExit(7)"),
                           dict(os.environ), {"MODEL": "/private/asset"}, 5)
        self.assertIn("<MODEL>", output.getvalue())
        self.assertNotIn("/private/asset", output.getvalue())

    def test_timeout_terminates_command(self):
        with (redirect_stdout(io.StringIO()),
              self.assertRaisesRegex(ci.CheckFailure, "timeout")):
            ci.run_command((sys.executable, "-c", "import time; time.sleep(30)"),
                           dict(os.environ), {}, 0.1)


class EvaluationTests(unittest.TestCase):
    def metric(self, ce="3.319376", ppl="27.6431", count=31):
        return f"int8    cross-entropy {ce}   perplexity {ppl}   weights 118 MB   {count} positions in 0.1s\n"

    def test_metric_requires_finite_scored_evaluation(self):
        self.assertEqual(ci.parse_ce(self.metric()), (3.319376, 27.6431, 31))
        for text in ("no metric", self.metric(count=0), self.metric(ce="NaN"), self.metric(ppl="inf"), self.metric() * 2):
            with self.subTest(text=text), self.assertRaises(ci.CheckFailure):
                ci.parse_ce(text)

    def test_changed_reference_metric_fails_the_suite(self):
        steps = [ci.Step("reference", ("unused",), ce_context=32), ci.Step("exact", ("unused",), ce_context=32)]
        with (patch.object(ci, "run_command", side_effect=[self.metric(), self.metric(ce="3.319377")]),
              redirect_stdout(io.StringIO()),
              self.assertRaisesRegex(ci.CheckFailure, "CE mismatch")):
            ci.execute(steps, {}, {}, "unused", Path(tempfile.gettempdir()), 18081)


if __name__ == "__main__":
    unittest.main()
