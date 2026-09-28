"""Manual self-hosted experiments. Controller uses only the Python standard library.

Project code is executed in child processes; their scientific behavior is unchanged.
Persistent outputs are deliberately outside the checkout that Actions cleans.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import ctypes
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = "Linshan-Ding/HGNN_PPO_USV_Scheduling"
METHODS = (
    "full",
    "no_hgnn",
    "shared_encoder",
    "no_reward_norm",
    "A2C",
    "DQN",
    "DDQN",
    "REINFORCE",
)
TASKS = ("smoke", "train", "scalability", "report", "republish")
SMALL_EXTENSIONS = {".csv", ".json", ".md", ".txt", ".pdf", ".png"}
MAX_GIT_FILE = 10 * 1024 * 1024
DEFAULT_ROOT = r"D:\GitHubActions\HGNN_PPO_USV_Scheduling"
DEFAULT_PYTHON = r"E:\anaconda3\envs\python3.13\python.exe"


def now():
    return datetime.now(timezone.utc).isoformat()


def root():
    return Path(os.environ.get("LOCAL_ACTIONS_ROOT", DEFAULT_ROOT)).resolve()


def python():
    return os.environ.get("LOCAL_ACTIONS_PYTHON", DEFAULT_PYTHON)


def inside(parent: Path, path: Path) -> Path:
    parent, path = parent.resolve(), path.resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError(f"Path must remain strictly inside {parent}: {path}")
    return path


def safe_key(value, pattern=r"[0-9]+-[0-9]+"):
    if not re.fullmatch(pattern, str(value)):
        raise ValueError(f"Invalid identifier: {value!r}")
    return str(value)


def run_path(key):
    return inside(root(), root() / "runs" / safe_key(key))


def run_key():
    return safe_key(os.environ["LOCAL_ACTIONS_RUN_KEY"])


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_output(name, value):
    if "\n" in str(value) or "\r" in str(value):
        raise ValueError("Multiline workflow output rejected")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def validate_inputs(raw):
    d = {
        "task": "smoke",
        "source_ref": "master",
        "method": "full",
        "epochs": 5000,
        "seed": 0,
        "instances": "",
        "data_source": "batch",
        "batch_id": "",
        "republish_run": "",
    }
    d.update(raw)
    if d["task"] not in TASKS or d["method"] not in METHODS:
        raise ValueError("Unknown task or training method")
    if d["data_source"] not in ("batch", "historical"):
        raise ValueError("Unknown data source")
    for name, low, high in [("epochs", 1, 1000000), ("seed", 0, 2147483647)]:
        if not re.fullmatch(r"[0-9]+", str(d[name])):
            raise ValueError(f"{name} must be an integer")
        d[name] = int(d[name])
        if not low <= d[name] <= high:
            raise ValueError(f"{name} must be in [{low}, {high}]")
    ref = d["source_ref"]
    if not isinstance(ref, str) or len(ref) > 240 or not ref:
        raise ValueError("Invalid source_ref")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        checked = subprocess.run(
            ["git", "check-ref-format", "--branch", ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if checked.returncode or ref.startswith(("-", "refs/pull/")):
            raise ValueError("source_ref must be a branch name or full commit SHA")
    ids = [i.strip() for i in str(d["instances"]).split(",") if i.strip()]
    if any(not re.fullmatch(r"u[0-9]+_t[0-9]+", i) for i in ids) or len(ids) != len(
        set(ids)
    ):
        raise ValueError("Invalid or duplicate instance IDs")
    d["instances"] = ",".join(sorted(ids))
    if d["batch_id"]:
        safe_key(d["batch_id"], r"[0-9a-f]{24}")
    if d["task"] == "republish":
        safe_key(d["republish_run"])
    return d


def init():
    inputs = validate_inputs(json.loads(os.environ.get("LOCAL_ACTIONS_INPUTS", "{}")))
    if inputs["task"] == "republish":
        key = inputs["republish_run"]
        manifest = read_json(run_path(key) / "manifest.json")
        if manifest["repository"] != REPOSITORY:
            raise ValueError("Run belongs to another repository")
    else:
        key = safe_key(
            f"{os.environ['GITHUB_RUN_ID']}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
        )
        target = run_path(key)
        target.mkdir(parents=True, exist_ok=False)
        for name in ("results", "models", "logs", "data"):
            (target / name).mkdir()
        write_json(
            target / "manifest.json",
            {
                "repository": REPOSITORY,
                "run_key": key,
                "status": "initialized",
                "created_at": now(),
                "inputs": inputs,
                "controller_sha": os.environ.get("GITHUB_SHA"),
                "workflow_url": f"https://github.com/{REPOSITORY}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                "notes": [],
            },
        )
    set_output("run_key", key)
    print(f"Run: {key}")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def files_under(folder):
    folder = Path(folder)
    if folder.is_symlink() or (hasattr(folder, "is_junction") and folder.is_junction()):
        raise ValueError(f"Linked output directories are not allowed: {folder}")
    if not folder.exists():
        return
    for path in sorted(folder.rglob("*")):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"Linked output paths are not allowed: {path}")
        if path.is_file():
            inside(folder, path)
            yield path


def copy_tree(source, destination):
    for src in files_under(source):
        dest = inside(Path(destination), Path(destination) / src.relative_to(source))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def batch_spec(source, sha, inputs):
    manifest = source / "data/public/manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    requested = set(filter(None, inputs["instances"].split(",")))
    available = {row["instance_id"] for row in rows}
    if requested - available:
        raise ValueError(f"Unknown instances: {sorted(requested - available)}")
    selected = [r for r in rows if not requested or r["instance_id"] in requested]
    digest = hashlib.sha256()
    for row in selected:
        data_file = inside(
            source / "data/public", source / "data/public" / row["filename"]
        )
        digest.update(row["instance_id"].encode())
        digest.update(bytes.fromhex(file_hash(data_file)))
    return {
        "source_sha": sha,
        "seed": inputs["seed"],
        "epochs": inputs["epochs"],
        "instances": [r["instance_id"] for r in selected],
        "data_sha256": digest.hexdigest(),
        "hidden_dim": 256,
        "hgnn_layers": 3,
        "n_heads": 4,
        "n_trajectories": 8,
    }


def batch_key(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]


class Commands:
    def __init__(self, directory, source, seconds):
        self.directory, self.source = directory, source
        self.deadline = time.monotonic() + seconds
        self.process = None
        self.number = 0

    def terminate(self):
        if self.process is not None and self.process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def run(self, argv, *, optional=False):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Task runtime limit reached")
        self.number += 1
        log = self.directory / "logs" / f"{self.number:02d}-{Path(argv[0]).stem}.log"
        env = dict(
            os.environ,
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            PYTHONUNBUFFERED="1",
            MPLBACKEND="Agg",
        )
        for name in list(env):
            if "TOKEN" in name.upper() or name.startswith("GIT_CONFIG_"):
                env.pop(name, None)
        print(
            "[Command] " + subprocess.list2cmdline([str(a) for a in argv]), flush=True
        )
        with log.open("w", encoding="utf-8") as output:
            self.process = subprocess.Popen(
                [str(a) for a in argv],
                cwd=self.source,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0,
                start_new_session=os.name != "nt",
            )

            def stream():
                for line in self.process.stdout:
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)

            reader = threading.Thread(target=stream, daemon=True)
            reader.start()
            try:
                code = self.process.wait(timeout=remaining)
            except BaseException:
                self.terminate()
                reader.join(timeout=10)
                raise
            reader.join()
        self.process = None
        if code and not optional:
            raise subprocess.CalledProcessError(code, argv)
        return code


def train_command(method, inputs, results, models, *, smoke=False):
    args = [python(), "-u"]
    if method in METHODS[:4]:
        args += ["multi_train.py", "--variant", method]
    else:
        args += ["-m", "drl_baselines.multi_run", "--algorithm", method]
    args += [
        "--result-dir",
        str(results),
        "--model-dir",
        str(models),
        "--max-epochs",
        str(inputs["epochs"]),
        "--seed",
        str(inputs["seed"]),
        "--hidden-dim",
        "32" if smoke else "256",
        "--hgnn-layers",
        "1" if smoke else "3",
        "--n-heads",
        "4",
        "--n-trajectories",
        "2" if smoke else "8",
        "--no-visdom",
    ]
    if inputs["instances"]:
        args += ["--instances", inputs["instances"]]
    return args


def note(manifest, text):
    print("[Note] " + text, flush=True)
    manifest["notes"].append(text)


def preflight(commands, need_gpu):
    if shutil.disk_usage(root()).free < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free space is required in the output drive")
    script = (
        "import json,sys,platform,importlib.metadata as m; "
        "import numpy,pandas,matplotlib,torch,visdom,pytest; "
        "d={'python':sys.version,'os':platform.platform(),'packages':{n:m.version(n) for n in "
        "['numpy','pandas','matplotlib','torch','visdom','pytest']},'cuda_available':torch.cuda.is_available(),"
        "'cuda_version':torch.version.cuda}; "
        "d['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None; "
        "d['cuda_probe']=(torch.ones(8,device='cuda')+1).sum().item() if torch.cuda.is_available() else None; "
        "print(json.dumps(d)); "
        + (
            "assert d['cuda_available'], 'CUDA is required for this training task'; "
            if need_gpu
            else ""
        )
        + 'open(sys.argv[1],"w",encoding="utf-8").write(json.dumps(d,indent=2))'
    )
    commands.run([python(), "-c", script, str(commands.directory / "environment.json")])


def update_batch(directory, manifest, slot):
    batch_id = manifest["batch_id"]
    path = inside(root(), root() / "batches" / batch_id / "batch.json")
    batch = (
        read_json(path)
        if path.exists()
        else {"batch_id": batch_id, "spec": manifest["batch_spec"], "runs": {}}
    )
    if batch["spec"] != manifest["batch_spec"]:
        raise ValueError("Batch configuration mismatch")
    batch["runs"][slot] = manifest["run_key"]
    batch["updated_at"] = now()
    write_json(path, batch)


def collect_inputs(source, directory, manifest):
    inputs = manifest["inputs"]
    destination = directory / "inputs"
    if inputs["data_source"] == "historical":
        copy_tree(source / "results", destination / "results")
        copy_tree(source / "models", destination / "models")
        note(
            manifest,
            "Historical inputs come only from the selected source commit; no new experiment outputs were merged.",
        )
        return destination
    selected = inputs["batch_id"] or manifest["batch_id"]
    safe_key(selected, r"[0-9a-f]{24}")
    path = inside(root(), root() / "batches" / selected / "batch.json")
    if not path.is_file():
        raise ValueError(
            f"Batch {selected} not found. Run train first, or choose historical inputs."
        )
    batch = read_json(path)
    if batch["spec"]["source_sha"] != manifest["source_sha"]:
        raise ValueError(
            "Batch code SHA differs from source_ref. Select the recorded code SHA."
        )
    # Explicit batch selection uses its recorded settings, not unrelated form defaults.
    manifest["batch_id"], manifest["batch_spec"] = selected, batch["spec"]
    manifest["input_runs"] = dict(batch["runs"])
    for method in (*METHODS, "scalability"):
        key = batch["runs"].get(method)
        if not key:
            continue
        previous = run_path(key)
        record = read_json(previous / "manifest.json")
        if (
            record["status"] != "success"
            or record.get("batch_id") != selected
            or record["inputs"]["task"] == "smoke"
        ):
            raise ValueError(f"Invalid completed batch member: {key}")
        if record["batch_spec"] != batch["spec"]:
            raise ValueError(f"Incompatible batch member: {key}")
        if method != "scalability":
            copy_tree(
                previous / "results/training_logs",
                destination / "results/training_logs",
            )
            copy_tree(previous / "models", destination / "models")
            rules = previous / "results/rules_results.csv"
            target = destination / "results/rules_results.csv"
            if rules.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rules, target)
        else:
            copy_tree(previous / "results", destination / "results")
    return destination


def report(commands, source, directory, manifest):
    data = collect_inputs(source, directory, manifest)
    logs, models = data / "results/training_logs", data / "models"
    rules = data / "results/rules_results.csv"
    commands.run(
        [
            python(),
            "-u",
            "extract_results.py",
            "--log-dir",
            logs,
            "--rules-csv",
            rules,
            "--out-dir",
            directory / "results",
        ]
    )
    if (data / "results/scalability_summary.csv").exists():
        shutil.copy2(
            data / "results/scalability_summary.csv",
            directory / "results/scalability_summary.csv",
        )
    # Only skip known missing inputs. An unexpected plot failure fails the task.
    methods = set()
    for log in files_under(logs):
        if log.suffix == ".csv" and log.name != "summary.csv":
            with log.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("protocol") == "round_robin":
                        methods.add((row.get("algorithm"), row.get("variant")))
                        break
    expected = {("PPO", m) for m in METHODS[:4]} | {
        (m, "baseline") for m in METHODS[4:]
    }
    missing = sorted(expected - methods)
    if missing:
        note(manifest, "Partial report; missing methods: " + repr(missing))
    seed = (
        manifest["batch_spec"]["seed"]
        if manifest["inputs"]["data_source"] == "batch"
        else manifest["inputs"]["seed"]
    )
    figures = [
        "training_curves",
        "convergence_all25",
        "ablation_curves",
        "gap_heatmap",
        "decision_time_heatmap",
        "drl_gap_violin",
        "gap_by_tasks",
        "dumbbell",
        "gap_ecdf",
        "scalability",
        "gantt",
    ]
    for figure in figures:
        reason = None
        if figure == "ablation_curves" and any(
            ("PPO", m) not in methods for m in METHODS[:4]
        ):
            reason = "not all ablation methods are present"
        if figure == "drl_gap_violin" and any(
            (m, "baseline") not in methods for m in METHODS[4:]
        ):
            reason = "not all DRL baseline methods are present"
        if (
            figure == "scalability"
            and not (directory / "results/scalability_summary.csv").exists()
        ):
            reason = "scalability results are missing"
        if figure == "gantt" and not (models / f"best_u6_t60_seed{seed}.pth").is_file():
            reason = "u6_t60 checkpoint is missing"
        if reason:
            note(manifest, f"Skipped {figure}: {reason}")
            continue
        commands.run(
            [
                python(),
                "-u",
                "analyze_training_logs.py",
                figure,
                "--log-dir",
                logs,
                "--model-dir",
                models,
                "--output-dir",
                directory / "results/figures",
                "--main-results-csv",
                directory / "results/main_results.csv",
                "--drl-results-csv",
                directory / "results/drl_results.csv",
                "--decision-time-csv",
                directory / "results/decision_time_grid.csv",
                "--scalability-csv",
                directory / "results/scalability_summary.csv",
                "--seed",
                str(seed),
            ]
        )


def scalability(commands, source, directory, manifest):
    data = collect_inputs(source, directory, manifest)
    models = data / "models"
    seed = (
        manifest["batch_spec"]["seed"]
        if manifest["inputs"]["data_source"] == "batch"
        else manifest["inputs"]["seed"]
    )
    names = [f"best_u10_t100_seed{seed}.pth"] + [
        f"best_{m}_u10_t100_seed{seed}.pth" for m in METHODS[4:]
    ]
    missing = [n for n in names if not (models / n).is_file()]
    if missing:
        raise ValueError(
            "Formal scalability requires trained checkpoints: " + ", ".join(missing)
        )
    commands.run(
        [
            python(),
            "-u",
            "scalability_experiment.py",
            "--seeds",
            str(seed),
            "--model-dir",
            models,
            "--data-dir",
            directory / "data/scalability",
            "--result-dir",
            directory / "results",
        ]
    )


def smoke(commands, directory, manifest):
    commands.run([python(), "-m", "pytest", "tests", "-q"])
    config = dict(
        manifest["inputs"], epochs=9, seed=0, instances="u2_t20,u2_t40,u2_t60"
    )
    commands.run(
        train_command(
            "full",
            config,
            directory / "results/full",
            directory / "models/full",
            smoke=True,
        )
    )
    for method in METHODS[1:]:
        config.update(epochs=2, instances="u2_t20,u2_t40")
        commands.run(
            train_command(
                method,
                config,
                directory / f"results/{method}",
                directory / f"models/{method}",
                smoke=True,
            )
        )
    commands.run(
        [
            python(),
            "-u",
            "scalability_experiment.py",
            "--smoke",
            "--hidden-dim",
            "16",
            "--hgnn-layers",
            "1",
            "--data-dir",
            directory / "data/scalability",
            "--result-dir",
            directory / "results/scalability",
        ]
    )
    note(
        manifest,
        "Smoke outputs are validation only; excluded from scientific experiment batches.",
    )


def execute(source):
    directory = run_path(run_key())
    manifest = read_json(directory / "manifest.json")
    if os.environ.get("LOCAL_ACTIONS_REPUBLISH") == "1" or manifest["status"] not in (
        "initialized",
    ):
        # A republish task refers to an existing run, never re-executes its code.
        print("Existing run selected; computation is not repeated.")
        return
    source = Path(source).resolve()
    inputs = manifest["inputs"]
    seconds = {
        "train": 72 * 3600,
        "smoke": 30 * 60,
        "report": 2 * 3600,
        "scalability": 2 * 3600,
    }[inputs["task"]]
    commands = Commands(directory, source, seconds)
    previous_handlers = {}

    def interrupted(signum, frame):
        commands.terminate()
        raise KeyboardInterrupt(f"Signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, interrupted)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        manifest["status"], manifest["started_at"] = "running", now()
        manifest["source_sha"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        manifest["batch_spec"] = batch_spec(source, manifest["source_sha"], inputs)
        manifest["batch_id"] = (
            batch_key(manifest["batch_spec"]) if inputs["task"] != "smoke" else None
        )
        write_json(directory / "manifest.json", manifest)
        preflight(commands, inputs["task"] in ("train", "smoke"))
        if inputs["task"] == "smoke":
            smoke(commands, directory, manifest)
        elif inputs["task"] == "train":
            commands.run(
                train_command(
                    inputs["method"],
                    inputs,
                    directory / "results",
                    directory / "models",
                )
            )
        elif inputs["task"] == "report":
            report(commands, source, directory, manifest)
        else:
            scalability(commands, source, directory, manifest)
        manifest["status"], manifest["exit_code"] = "success", 0
    except BaseException as exc:
        manifest["status"] = (
            "timeout"
            if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))
            else "cancelled"
            if isinstance(exc, KeyboardInterrupt)
            else "failed"
        )
        manifest["exit_code"] = getattr(exc, "returncode", 1) or 1
        manifest["error"] = str(exc)
        (directory / "logs/controller-error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        commands.terminate()
        manifest["finished_at"] = now()
        write_json(directory / "manifest.json", manifest)
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        summary(directory, manifest)
    if manifest["status"] == "success":
        if inputs["task"] == "train":
            update_batch(directory, manifest, inputs["method"])
        elif inputs["task"] == "scalability" and inputs["data_source"] == "batch":
            update_batch(directory, manifest, "scalability")


def summary(directory, manifest):
    lines = [
        "# Local experiment",
        "",
        f"- Run: `{manifest['run_key']}`",
        f"- Status: **{manifest['status']}**",
        f"- Task: `{manifest['inputs']['task']}`",
        f"- Method: `{manifest['inputs']['method']}`",
        f"- Code: `{manifest.get('source_sha', 'not checked out')}`",
        f"- Batch: `{manifest.get('batch_id') or 'none (smoke)'}`",
        f"- [Actions run]({manifest['workflow_url']})",
        "",
    ]
    if manifest.get("error"):
        lines += ["## Error", "", "```text", manifest["error"], "```", ""]
    lines += ["- " + n for n in manifest.get("notes", [])]
    text = "\n".join(lines) + "\n"
    (directory / "README.md").write_text(text, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)


def finalize():
    directory = run_path(run_key())
    manifest = read_json(directory / "manifest.json")
    if (
        manifest["status"] == "success"
        and os.environ.get("LOCAL_ACTIONS_COMPUTE_OUTCOME") == "failure"
    ):
        manifest["status"] = "failed"
        manifest["error"] = (
            "Compute post-processing failed; inspect the Actions job log."
        )
        write_json(directory / "manifest.json", manifest)
    if manifest["status"] in ("initialized", "running"):
        manifest["status"] = "interrupted"
        manifest["finished_at"] = now()
        manifest["error"] = (
            "Checkout, preflight or compute did not finish; see the Actions job log."
        )
        write_json(directory / "manifest.json", manifest)
    summary(directory, manifest)


def bundle():
    directory = run_path(run_key())
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] in ("initialized", "running"):
        manifest["status"] = "interrupted"
        manifest["finished_at"] = now()
        write_json(directory / "manifest.json", manifest)
    summary(directory, manifest)
    publication = safe_key(
        f"{os.environ['GITHUB_RUN_ID']}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    )
    stage = inside(root(), root() / "publications" / publication / "artifact")
    stage.mkdir(parents=True, exist_ok=False)
    inventory = []
    for folder in ("results", "models", "logs", "data"):
        for source in files_under(directory / folder):
            rel = source.relative_to(directory)
            target = inside(stage, stage / rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            inventory.append(
                {
                    "path": rel.as_posix(),
                    "size": source.stat().st_size,
                    "sha256": file_hash(source),
                }
            )
    for name in ("manifest.json", "environment.json", "README.md"):
        if (directory / name).exists():
            shutil.copy2(directory / name, stage / name)
    write_json(stage / "inventory.json", inventory)
    set_output("artifact_path", stage)


def small_result(relative, size):
    path = Path(relative)
    return (
        size <= MAX_GIT_FILE
        and path.suffix.lower() in SMALL_EXTENSIONS
        and path.parts[0] == "results"
        and "training_logs" not in path.parts
    )


def git_env(token):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    # Environment configuration keeps the temporary token out of command lines and disk.
    for key in list(env):
        if key.startswith("GIT_CONFIG_"):
            env.pop(key)
    env.update(
        GIT_CONFIG_COUNT="2",
        GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
        GIT_CONFIG_VALUE_0="AUTHORIZATION: basic "
        + base64.b64encode(("x-access-token:" + token).encode()).decode(),
        GIT_CONFIG_KEY_1="credential.helper",
        GIT_CONFIG_VALUE_1="",
    )
    return env


def publish():
    key = run_key()
    directory = run_path(key)
    manifest = read_json(directory / "manifest.json")
    publication = safe_key(
        f"{os.environ['GITHUB_RUN_ID']}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    )
    stage = inside(root(), root() / "publications" / publication / "artifact")
    checkout = inside(root(), root() / "publications" / publication / "git")
    checkout.mkdir(parents=True, exist_ok=False)
    env = git_env(os.environ["GH_TOKEN"])

    def git(*args, check=True):
        p = subprocess.run(
            ["git", *args],
            cwd=checkout,
            env=env,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        if p.returncode and check:
            # Git output never contains the environment-only authorization header.
            raise RuntimeError(
                p.stderr.strip() or p.stdout.strip() or f"git {args[0]} failed"
            )
        return p

    git("init")
    git("remote", "add", "origin", f"https://github.com/{REPOSITORY}.git")
    remote = git("ls-remote", "--heads", "origin", "refs/heads/results")
    if remote.stdout.strip():
        git("fetch", "--depth", "1", "origin", "refs/heads/results")
        git("checkout", "-b", "results", "FETCH_HEAD")
    else:
        git("checkout", "--orphan", "results")
    destination = checkout / "runs" / key
    destination.mkdir(parents=True, exist_ok=True)
    for item in read_json(stage / "inventory.json"):
        if small_result(item["path"], item["size"]):
            target = inside(destination, destination / item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inside(stage, stage / item["path"]), target)
    for name in ("manifest.json", "environment.json", "README.md", "inventory.json"):
        if (stage / name).exists():
            shutil.copy2(stage / name, destination / name)
    artifact_url = os.environ.get("LOCAL_ACTIONS_ARTIFACT_URL", "")
    entry = {
        "run_key": key,
        "status": manifest["status"],
        "task": manifest["inputs"]["task"],
        "method": manifest["inputs"]["method"],
        "source_sha": manifest.get("source_sha"),
        "batch_id": manifest.get("batch_id"),
        "workflow_url": manifest["workflow_url"],
        "artifact_url": artifact_url,
        "artifact_status": os.environ.get("LOCAL_ACTIONS_ARTIFACT_STATUS", "unknown"),
        "publication_run": publication,
        "published_at": now(),
    }
    write_json(destination / "publication.json", entry)
    index_file = checkout / "index.json"
    index = read_json(index_file) if index_file.exists() else {}
    index[key] = entry
    write_json(index_file, index)
    lines = [
        "# 本机实验结果",
        "",
        "完整输出在各次 Actions 附件中保留 14 天；模型和完整训练日志不进入 Git 历史。",
        "",
        "| 运行 | 状态 | 任务/方法 | 代码 | 批次 | 完整输出 |",
        "|---|---|---|---|---|---|",
    ]
    for item in sorted(index.values(), key=lambda x: x["published_at"], reverse=True):
        artifact = (
            f"[附件]({item['artifact_url']})"
            if item.get("artifact_url")
            else "上传失败，可补传"
        )
        lines.append(
            f"| [{item['run_key']}](runs/{item['run_key']}/README.md) | {item['status']} | "
            f"{item['task']}/{item['method']} | `{(item.get('source_sha') or '')[:12]}` | "
            f"`{item.get('batch_id') or ''}` | {artifact} |"
        )
    (checkout / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    git("config", "user.name", "github-actions[bot]")
    git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    git("add", "--", "runs/" + key, "index.json", "README.md")
    git("commit", "-m", f"Record local experiment {key} ({manifest['status']})")
    for attempt in range(3):
        pushed = git("push", "origin", "HEAD:refs/heads/results", check=False)
        if pushed.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError(
                "Results push failed; local output is intact. Use republish. "
                + pushed.stderr
            )
        git("fetch", "origin", "refs/heads/results")
        git("rebase", "FETCH_HEAD")
    url = f"https://github.com/{REPOSITORY}/tree/results/runs/{key}"
    print(f"Results published: {url}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"\n[结果分支]({url})\n\n[完整输出附件]({artifact_url})\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["init", "execute", "finalize", "bundle", "publish"]
    )
    parser.add_argument("--source")
    args = parser.parse_args()
    if args.command == "execute":
        execute(args.source)
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
