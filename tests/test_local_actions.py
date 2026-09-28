"""Behavioral tests for output isolation, fail-closed inputs and recovery."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import local_actions as a


class LocalActionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.env = patch.dict(
            os.environ,
            {
                "LOCAL_ACTIONS_ROOT": str(self.base),
                "LOCAL_ACTIONS_PYTHON": sys.executable,
                "GITHUB_RUN_ID": "123456",
                "GITHUB_RUN_ATTEMPT": "1",
                "LOCAL_ACTIONS_RUN_KEY": "123456-1",
                "LOCAL_ACTIONS_INPUTS": "{}",
                "GITHUB_OUTPUT": str(self.base / "outputs.txt"),
                "GITHUB_STEP_SUMMARY": str(self.base / "summary.md"),
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_rejects_path_and_ref_injection(self):
        for text in ["../outside", "1-1/../../x", "C:\\other", "1-1\nvalue=x"]:
            with self.assertRaises(ValueError):
                a.run_path(text)
        for values in [
            {"epochs": "1.5"},
            {"seed": -1},
            {"instances": "u2_t20;whoami"},
            {"instances": "u2_t20,u2_t20"},
            {"source_ref": "--upload-pack=cmd"},
            {"source_ref": "master\nmalicious"},
            {"batch_id": "../x"},
            {"task": "shell"},
            {"method": "unknown"},
        ]:
            with self.assertRaises(ValueError):
                a.validate_inputs(values)
        self.assertEqual(
            a.validate_inputs({"source_ref": "claude/example"})["source_ref"],
            "claude/example",
        )

    def test_run_isolation_and_republish_does_not_reinitialize(self):
        a.init()
        first = a.run_path("123456-1")
        state = a.read_json(first / "manifest.json")
        state["status"] = "failed"
        a.write_json(first / "manifest.json", state)
        original = (first / "manifest.json").read_bytes()
        os.environ["LOCAL_ACTIONS_INPUTS"] = json.dumps(
            {"task": "republish", "republish_run": "123456-1"}
        )
        a.init()
        a.execute("does-not-exist")
        self.assertEqual(original, (first / "manifest.json").read_bytes())
        os.environ["LOCAL_ACTIONS_INPUTS"] = "{}"
        os.environ["GITHUB_RUN_ATTEMPT"] = "2"
        a.init()
        self.assertTrue(a.run_path("123456-2").exists())

    def test_small_result_policy_keeps_checkpoints_and_logs_out_of_git(self):
        self.assertTrue(a.small_result("results/figures/chart.pdf", 123))
        self.assertFalse(a.small_result("results/training_logs/train.csv", 123))
        self.assertFalse(a.small_result("models/checkpoint.pth", 123))
        self.assertFalse(a.small_result("results/huge.csv", a.MAX_GIT_FILE + 1))
        self.assertFalse(a.small_result("results/.env", 123))

    def test_failed_run_is_bundleable_without_source(self):
        a.init()
        directory = a.run_path("123456-1")
        (directory / "logs/failure.log").write_text(
            "intentional failure", encoding="utf-8"
        )
        (directory / "results/report.json").write_text("{}", encoding="utf-8")
        a.finalize()
        a.bundle()
        stage = self.base / "publications/123456-1/artifact"
        self.assertEqual(a.read_json(stage / "manifest.json")["status"], "interrupted")
        self.assertTrue((stage / "logs/failure.log").exists())
        self.assertFalse((stage / "source").exists())

    def test_child_failure_propagates_and_log_is_retained(self):
        a.init()
        directory = a.run_path("123456-1")
        commands = a.Commands(directory, self.base, 10)
        with self.assertRaises(subprocess.CalledProcessError) as context:
            commands.run(
                [
                    sys.executable,
                    "-c",
                    'print("failure evidence"); raise SystemExit(17)',
                ]
            )
        self.assertEqual(context.exception.returncode, 17)
        self.assertIn(
            "failure evidence", next((directory / "logs").glob("*.log")).read_text()
        )

    def test_timeout_stops_child(self):
        a.init()
        commands = a.Commands(a.run_path("123456-1"), self.base, 0.4)
        with self.assertRaises(subprocess.TimeoutExpired):
            commands.run([sys.executable, "-c", "import time; time.sleep(20)"])
        self.assertIsNotNone(commands.process.poll())

    def test_rejects_incomplete_or_incompatible_batch(self):
        a.init()
        directory = a.run_path("123456-1")
        state = a.read_json(directory / "manifest.json")
        state.update(
            batch_id="a" * 24, source_sha="b" * 40, batch_spec={"source_sha": "b" * 40}
        )
        batch = self.base / "batches" / ("a" * 24) / "batch.json"
        a.write_json(batch, {"spec": {"source_sha": "c" * 40}, "runs": {}})
        with self.assertRaisesRegex(ValueError, "code SHA"):
            a.collect_inputs(self.base, directory, state)
        a.write_json(batch, {"spec": state["batch_spec"], "runs": {"full": "123456-1"}})
        a.write_json(directory / "manifest.json", dict(state, status="failed"))
        with self.assertRaisesRegex(ValueError, "Invalid completed"):
            a.collect_inputs(self.base, directory, state)

    def test_checkpoint_preflight_fails_without_running_random_model(self):
        a.init()
        directory = a.run_path("123456-1")
        state = a.read_json(directory / "manifest.json")
        state["inputs"]["data_source"] = "historical"
        commands = a.Commands(directory, self.base, 10)
        with self.assertRaisesRegex(ValueError, "trained checkpoints"):
            a.scalability(commands, self.base / "historical", directory, state)
        self.assertEqual(commands.number, 0)

    def test_successful_batch_collects_only_indexed_run(self):
        a.init()
        old = a.run_path("123456-1")
        spec = {"source_sha": "b" * 40, "seed": 0, "epochs": 5000}
        state = a.read_json(old / "manifest.json")
        state.update(
            status="success", source_sha="b" * 40, batch_id="a" * 24, batch_spec=spec
        )
        state["inputs"]["task"] = "train"
        a.write_json(old / "manifest.json", state)
        (old / "results/training_logs").mkdir()
        (old / "results/training_logs/selected.csv").write_text(
            "selected", encoding="utf-8"
        )
        (old / "results/rules_results.csv").write_text("rules", encoding="utf-8")
        a.update_batch(old, state, "full")
        target = self.base / "collection"
        a.collect_inputs(self.base, target, state)
        self.assertEqual(
            (target / "inputs/results/training_logs/selected.csv").read_text(),
            "selected",
        )
        self.assertEqual(state["input_runs"], {"full": "123456-1"})

    def test_publication_creates_results_only_branch_and_can_republish(self):
        remote = self.base / "remote.git"
        subprocess.run(
            ["git", "init", "--bare", str(remote)], check=True, capture_output=True
        )
        a.init()
        directory = a.run_path("123456-1")
        (directory / "results/metrics.csv").write_text("score\n1\n", encoding="utf-8")
        (directory / "models/checkpoint.pth").write_bytes(b"model")
        (directory / "logs/console.txt").write_text("log", encoding="utf-8")
        a.finalize()
        os.environ.update(
            GH_TOKEN="test-only",
            LOCAL_ACTIONS_ARTIFACT_URL="https://example.invalid/artifact",
            LOCAL_ACTIONS_ARTIFACT_STATUS="success",
        )
        original_run = subprocess.run

        def local_git(argv, **kwargs):
            if argv[:4] == ["git", "remote", "add", "origin"]:
                argv = argv[:4] + [str(remote)]
            return original_run(argv, **kwargs)

        with patch.object(a.subprocess, "run", side_effect=local_git):
            a.bundle()
            a.publish()
            os.environ["GITHUB_RUN_ATTEMPT"] = "2"
            a.bundle()
            a.publish()
        names = subprocess.check_output(
            [
                "git",
                "--git-dir",
                str(remote),
                "ls-tree",
                "-r",
                "--name-only",
                "results",
            ],
            text=True,
        )
        self.assertIn("runs/123456-1/results/metrics.csv", names)
        self.assertNotIn("checkpoint.pth", names)
        self.assertNotIn("console.txt", names)
        index = json.loads(
            subprocess.check_output(
                ["git", "--git-dir", str(remote), "show", "results:index.json"],
                text=True,
            )
        )
        self.assertEqual(list(index), ["123456-1"])
        self.assertEqual(index["123456-1"]["publication_run"], "123456-2")


if __name__ == "__main__":
    unittest.main()
