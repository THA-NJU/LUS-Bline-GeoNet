#!/usr/bin/env python3
"""Plan and run test-only lesion evaluation for all 36 models and two tracks.

This is a process-level orchestrator: every task uses its original evaluator
and Python environment.  ``plan`` never executes a model.  ``run`` consumes the
reviewed immutable command list and supports restart by completed summary file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}
TRACKS = {512: "Size_512", 224: "Size_224_filtered"}


@dataclass(frozen=True)
class Spec:
    family: str
    model: str
    weight_family: str
    weight_dir: str
    backend: str


SPECS = [
    *(Spec("CNN", name, "CNN", name, "monai") for name in
      ("monai_unet", "monai_vnet", "monai_attention_unet", "monai_unetplusplus")),
    Spec("CNN", "mednext", "CNN", "mednext", "additional"),
    *(Spec("CNN", name, "CNN", name, "autodl") for name in ("nnunet_2d", "aau_net")),
    Spec("CNN", "Resnet34_fpn", "CNN", "Resnet34_fpn", "resnet_fpn"),
    Spec("CNN", "Resnet34_DLV3", "CNN", "Resnet34_DLV3", "resnet_dlv3"),
    *(Spec("Transformer", name, "Transformer", name, "autodl") for name in
      ("swinunet", "nnformer_2d")),
    Spec("Transformer", "pvtb2_emcad", "Transformer", "pvtb2_emcad", "additional"),
    *(Spec("Transformer", name, "Transformer", name, "autodl") for name in
      ("sepnet", "nulite")),
    *(Spec("Mamba", name, "Mamba", name, "mamba") for name in
      ("mamba_unet", "vm_unet_v2", "nnmamba_2d", "swin_umamba")),
    Spec("other", "rolling_unet", "other", "rolling_unet", "additional"),
    *(Spec("other", name, "other", name, "autodl") for name in
      ("ukan", "xlstm_unet_bot")),
    Spec("S2DENet", "s2denet", "S2DENet", "s2denet", "s2denet"),
    *(Spec("RWKV", name, "RWKV", name, "rwkv") for name in ("u_rwkv", "rwkv_unet")),
    *(Spec("SAM", name, "SAM", name, "sam") for name in ("sam2", "medsam", "samus")),
    Spec("SAM", "sam3", "sam3", "", "sam3"),
    Spec("Transformer", "USFM", "Transformer", "USFM", "usfm"),
    *(Spec("YOLO", name.replace("-", "_"), "Yolo", name, "yolo") for name in
      ("yolo11s-seg", "yolo11m-seg", "yolo11l-seg")),
    Spec("YOLO", "yolo26s_sem", "Yolo", "yolo26s_sem", "yolo26s"),
    *(Spec("YOLO", name.replace("-", "_"), "Yolo", name, "yolo") for name in
      ("yolo26m-sem", "yolo26l-sem")),
    Spec("Transformer", "segformer_b2", "Transformer", "Segformer", "segformer"),
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def png_identifiers(directory: Path) -> list[str]:
    values = sorted(
        path.stem for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate PNG identifiers in {directory}")
    return values


def identifier_sha256(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def candidates(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path.resolve() for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
        and "size224_full_metrics" not in path.parts
        and "last" not in path.name.lower()
    )


def select_checkpoint(directory: Path, explicit: str | None = None):
    found = candidates(directory)
    if explicit:
        selected = Path(explicit).expanduser().resolve()
        if not selected.is_file() or selected.suffix.lower() not in CHECKPOINT_SUFFIXES:
            return None, "INVALID_CHECKPOINT_OVERRIDE", found
        return selected, "READY", found
    if not found:
        return None, "MISSING_CHECKPOINT", []
    exact_names = ("best_model.pth", "best.pt", "best.pth", "best_model.pt")
    exact = [path for path in found if path.name.lower() in exact_names]
    if len(exact) == 1:
        return exact[0], "READY", found
    best = [path for path in found if "best" in path.name.lower()]
    if len(best) == 1:
        return best[0], "READY", found
    if len(found) == 1:
        return found[0], "READY", found
    return None, "AMBIGUOUS_CHECKPOINT", found


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def python_for(config, backend):
    mapping = config.get("python_by_backend", {})
    return str(mapping.get(backend, mapping.get("default", sys.executable)))


def command_for(spec, size, checkpoint, repo, data_root, weights_root,
                output_root, config, gpu):
    py = python_for(config, spec.backend)
    data512 = data_root / TRACKS[512]
    data224 = data_root / TRACKS[224]
    data = data_root / TRACKS[size]
    april = Path(config.get("april_root", repo / "third_party/april_medseg"))
    # Lock execution to the exact checkpoint (and hash) recorded by ``plan``.
    # Family evaluators read this process-local override instead of searching
    # their directory again at run time.
    environment_assignments = [f"LUSBENCH_EXACT_CHECKPOINT={checkpoint}"]
    if config.get("legacy_unnormalized_input", True):
        environment_assignments.append("LUSBENCH_LEGACY_UNNORMALIZED_INPUT=1")
    base = ["/usr/bin/env", *environment_assignments, py]
    if spec.backend == "monai":
        return base + [str(repo / "evaluation/families/evaluate_monai_family.py"),
            "--weights-root", str(weights_root), "--output-root", str(output_root / "CNN"),
            "--data-root-512", str(data512), "--data-root-224", str(data224),
            "--sizes", str(size), "--models", spec.model, "--splits", "test",
            "--gpu", str(gpu), "--no-auto-batch", "--fixed-batch", "1"]
    if spec.backend == "additional":
        script = ("evaluate_mednext_pvtb2_rolling.py" if size == 512
                  else "evaluate_mednext_pvtb2_rolling_224.py")
        return base + [str(repo / "evaluation/families" / script),
            "--april-root", str(april), "--weights-root", str(weights_root),
            "--output-root", str(output_root / spec.family), "--data-root", str(data),
            "--models", spec.model, "--splits", "test", "--gpu", str(gpu),
            "--no-auto-batch", "--fixed-batch", "1"]
    if spec.backend == "autodl":
        return base + [str(repo / "evaluation/families/evaluate_autodl_easy_models.py"),
            "--training-script", str(repo / "experiments/april/train_general.py"),
            "--weights-root", str(weights_root), "--output-root", str(output_root),
            "--data-root-512", str(data512), "--data-root-224", str(data224),
            "--sizes", str(size), "--families", spec.family, "--models", spec.model,
            "--splits", "test", "--gpu", str(gpu), "--no-auto-batch"]
    if spec.backend in ("mamba", "rwkv"):
        script = "evaluate_mamba_family.py" if spec.backend == "mamba" else "evaluate_rwkv_family.py"
        return base + [str(repo / "evaluation/families" / script),
            "--april-root", str(april), "--weights-root", str(weights_root),
            "--output-root", str(output_root / spec.family),
            "--data-root-512", str(data512), "--data-root-224", str(data224),
            "--sizes", str(size), "--models", spec.model, "--splits", "test",
            "--gpu", str(gpu), "--no-auto-batch", "--fixed-batch", "1"]
    if spec.backend == "segformer":
        return base + [str(repo / "evaluation/native/evaluate_segformer_b2.py"),
            "--checkpoint", str(checkpoint), "--size", str(size),
            "--data-root", str(data), "--output-dir",
            str(output_root / "Transformer" / str(size) / "segformer_b2"),
            "--splits", "test", "--local-files-only"]
    if spec.backend == "yolo26s":
        return base + [str(repo / "evaluation/native/evaluate_yolo26s_sem.py"),
            "--checkpoint", str(checkpoint), "--size", str(size),
            "--data-root", str(data), "--output-dir",
            str(output_root / "YOLO" / str(size) / "yolo26s_sem"),
            "--splits", "test", "--device", str(gpu)]
    if spec.backend == "yolo":
        cli_model = spec.weight_dir
        return base + [str(repo / "experiments/yolo/train_family.py"),
            "--eval-only", "--model", cli_model, "--size", str(size),
            "--weights", str(checkpoint), "--data-root", str(data),
            "--evaluation-root", str(output_root / "YOLO"),
            "--run-root", str(output_root / "_runs/yolo"),
            "--cache-root", str(output_root / "_cache/yolo"),
            "--device", str(gpu), "--skip-figures"]
    if spec.backend == "s2denet":
        source = config.get("s2denet_source_root")
        if not source:
            return None
        return base + [str(repo / "experiments/s2denet/train.py"), "--eval-only",
            "--checkpoint", str(checkpoint), "--size", str(size),
            "--data-root", str(data), "--repo-root", str(source),
            "--output-root", str(output_root / "S2DENet"), "--device", str(gpu)]
    if spec.backend == "usfm":
        return base + [str(repo / "experiments/usfm/evaluate_zeroshot.py"),
            "--checkpoint", str(checkpoint), "--sizes", str(size),
            "--data-root-512", str(data512), "--data-root-224", str(data224),
            "--output-root", str(output_root / "Transformer" / "USFM"),
            "--split", "test", "--device", f"cuda:{gpu}", "--skip-count-check"]
    if spec.backend.startswith("resnet_"):
        evaluator_model = "fpn_resnet34" if spec.backend.endswith("fpn") else "deeplabv3plus_resnet34"
        return base + [str(repo / "evaluation/legacy_resnet.py"),
            "--checkpoint", str(checkpoint), "--models", evaluator_model,
            "--sizes", str(size), "--data-root-512", str(data512),
            "--data-root-224", str(data224), "--output-root", str(output_root / "CNN"),
            "--split", "test", "--device", f"cuda:{gpu}", "--skip-count-check"]
    if spec.backend == "sam":
        extra = config.get("sam_extra_args", {}).get(f"{size}/{spec.model}")
        if not extra:
            return None
        forbidden = {"--data-root", "--resume", "--min-mask-pixels",
                     "--output-dir", "--eval-only"}
        collisions = sorted(forbidden.intersection(map(str, extra)))
        if collisions:
            raise ValueError(
                f"sam_extra_args for {size}/{spec.model} contains controlled flags: {collisions}"
            )
        # Size-224 checkpoints were trained through the adapter, but the
        # architecture and preprocessing still live in the corresponding base
        # script. The rich evaluator must import that base script directly.
        training = repo / "experiments/foundation" / f"train_{spec.model}_512.py"
        # Evaluate the paired test images with min_mask_pixels=0.
        forwarded = ["--data-root", str(data), "--resume", str(checkpoint),
                     "--min-mask-pixels", "0"]
        return base + [str(repo / "evaluation/foundation/evaluate_sam_checkpoints.py"),
            "--model", spec.model, "--benchmark-script", str(training),
            "--output-dir", str(output_root / "SAM" / str(size) / spec.model),
            "--gpu", str(gpu), "--", *forwarded, *map(str, extra)]
    if spec.backend == "sam3":
        source, base_checkpoint = config.get("sam3_source_root"), config.get("sam3_base_checkpoint")
        if not source or not base_checkpoint:
            return None
        return base + [str(repo / "evaluation/foundation/evaluate_sam3_official.py"),
            "--sam3-root", str(source), "--base-checkpoint", str(base_checkpoint),
            "--finetuned-checkpoint", str(checkpoint), "--data-root", str(data),
            "--output-root", str(output_root / "SAM"), "--size", str(size),
            "--splits", "test", "--gpu", str(gpu)]
    raise KeyError(spec.backend)


def result_path(output_root: Path, spec: Spec, size: int):
    model = spec.model
    if spec.backend.startswith("resnet_"):
        model = "fpn_resnet34" if spec.backend.endswith("fpn") else "deeplabv3plus_resnet34"
    if spec.backend == "usfm":
        return output_root / "Transformer/USFM" / str(size) / "lesion/test_summary.json"
    return output_root / spec.family / str(size) / model / "lesion/test_summary.json"


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def collect_completed(payload):
    """Write one compact row per threshold for every completed task."""
    rows = []
    status_rows = []
    for task in payload["tasks"]:
        result = Path(task["result"])
        completed = result.is_file()
        status_rows.append({
            "order": task["order"], "size": task["size"],
            "family": task["family"], "model": task["model"],
            "plan_status": task["status"], "result_complete": int(completed),
            "result": str(result),
        })
        if not completed:
            continue
        summary = json.loads(result.read_text(encoding="utf-8"))
        for metric in summary.get("metrics", []):
            rows.append({
                "order": task["order"], "size": task["size"],
                "family": task["family"], "model": task["model"],
                "images": summary.get("images"),
                "empty_prediction_count": summary.get("empty_prediction_count"),
                "empty_prediction_rate": summary.get("empty_prediction_rate"),
                **{key: value for key, value in metric.items() if key not in {"model", "size", "froc"}},
            })
    root = Path(payload["output_root"])
    write_csv(root / "lesion_suite_status.csv", status_rows)
    if rows:
        write_csv(root / "lesion_metrics_all.csv", rows)


def plan(args):
    repo = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    weights_root = args.weights_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    config = load_config(args.env_config)
    cleanup_manifest = data_root / "gt_component_cleanup.json"
    if (data_root / ".incomplete").exists():
        raise RuntimeError(f"Derived GT root is incomplete: {data_root}")
    if not cleanup_manifest.is_file():
        raise FileNotFoundError(
            f"Missing GT cleanup manifest; run preprocess_gt_components.py first: {cleanup_manifest}"
        )
    cleanup = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
    if cleanup.get("connectivity") != 8 or cleanup.get("remove_rule") != "component_area < 5":
        raise ValueError(f"Unexpected GT cleanup protocol in {cleanup_manifest}")
    datasets = {}
    for size, track in TRACKS.items():
        image_dir = data_root / track / "test/images"
        mask_dir = data_root / track / "test/masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing {image_dir} or {mask_dir}")
        image_ids, mask_ids = png_identifiers(image_dir), png_identifiers(mask_dir)
        if image_ids != mask_ids:
            raise ValueError(f"Test image/mask identifiers differ for Size_{size}")
        datasets[str(size)] = {
            "track": track, "test_images": len(image_ids),
            "identifier_sha256": identifier_sha256(image_ids),
        }
    rows = []
    for size in (512, 224):
        for order, spec in enumerate(SPECS, 1):
            weight_dir = weights_root / spec.weight_family
            if spec.weight_dir:
                weight_dir = weight_dir / str(size) / spec.weight_dir
            else:
                weight_dir = weight_dir / str(size)
            override = config.get("checkpoint_by_task", {}).get(f"{size}/{spec.model}")
            checkpoint, status, found = select_checkpoint(weight_dir, override)
            command = None
            if checkpoint is not None:
                command = command_for(spec, size, checkpoint, repo, data_root,
                                      weights_root, output_root, config, args.gpu)
                if command is None:
                    status = "NEEDS_ENV_CONFIG"
                elif not Path(python_for(config, spec.backend)).expanduser().is_file():
                    status = "MISSING_PYTHON"
            rows.append({
                "order": order, "size": size, "family": spec.family,
                "model": spec.model, "backend": spec.backend,
                "weight_dir": str(weight_dir), "status": status,
                "checkpoint_override": override,
                "checkpoint": None if checkpoint is None else str(checkpoint),
                "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
                "candidate_count": len(found),
                "candidates": [str(path) for path in found],
                "command": command,
                "result": str(result_path(output_root, spec, size)),
            })
    payload = {
        "schema": "lusbench-lesion-suite-plan-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo), "data_root": str(data_root),
        "weights_root": str(weights_root), "output_root": str(output_root),
        "datasets": datasets,
        "dataset_manifest": str(cleanup_manifest),
        "dataset_manifest_sha256": sha256(cleanup_manifest),
        "tasks": rows,
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.plan.with_suffix(".csv"), [
        {**row, "candidates": " | ".join(row["candidates"]),
         "command": "" if row["command"] is None else shlex.join(row["command"])}
        for row in rows
    ])
    counts = {status: sum(row["status"] == status for row in rows)
              for status in sorted({row["status"] for row in rows})}
    print(json.dumps({"plan": str(args.plan), "tasks": len(rows), "status": counts}, indent=2))


def run(args):
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest_path = payload.get("dataset_manifest")
    expected_manifest_hash = payload.get("dataset_manifest_sha256")
    if manifest_path and expected_manifest_hash:
        current_manifest_hash = sha256(Path(manifest_path))
        if current_manifest_hash != expected_manifest_hash:
            raise RuntimeError(
                "Dataset cleanup manifest changed after plan creation; build and review a new plan"
            )
    data_root = Path(payload["data_root"])
    for size, track in TRACKS.items():
        planned = payload["datasets"][str(size)]
        image_ids = png_identifiers(data_root / track / "test/images")
        mask_ids = png_identifiers(data_root / track / "test/masks")
        if (image_ids != mask_ids
                or identifier_sha256(image_ids) != planned["identifier_sha256"]):
            raise RuntimeError(
                f"Size_{size} test identifiers changed after plan creation; build a new plan"
            )
    tasks = payload["tasks"]
    blocked = [row for row in tasks if row["status"] != "READY"]
    if blocked and not args.allow_partial_plan:
        raise RuntimeError(
            f"Plan has {len(blocked)} unresolved tasks; review {args.plan.with_suffix('.csv')} "
            "or pass --allow-partial-plan to run READY tasks only"
        )
    log_dir = Path(payload["output_root"]) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for row in tasks:
        if row["status"] != "READY":
            continue
        if args.sizes and row["size"] not in args.sizes:
            continue
        if args.models and row["model"] not in args.models:
            continue
        result = Path(row["result"])
        provenance = result.parent / "task_provenance.json"
        if result.is_file() and not args.rerun:
            prior = (json.loads(provenance.read_text(encoding="utf-8"))
                     if provenance.is_file() else {})
            if (prior.get("checkpoint_sha256") == row["checkpoint_sha256"]
                    and prior.get("data_root") == payload["data_root"]
                    and prior.get("dataset_manifest_sha256")
                    == payload.get("dataset_manifest_sha256")):
                print(f"[SKIP COMPLETE] {row['size']}/{row['model']}")
                continue
            failures.append({"size": row["size"], "model": row["model"],
                             "returncode": None, "log": None,
                             "error": "Existing result has absent/mismatched provenance; use --rerun"})
            print(f"[STALE RESULT] {row['size']}/{row['model']}; use --rerun")
            continue
        actual_checkpoint_hash = sha256(Path(row["checkpoint"]))
        if actual_checkpoint_hash != row["checkpoint_sha256"]:
            failures.append({"size": row["size"], "model": row["model"],
                             "returncode": None, "log": None,
                             "error": "Checkpoint changed after plan creation"})
            print(f"[CHECKPOINT CHANGED] {row['size']}/{row['model']}")
            continue
        command = row["command"]
        print(f"[RUN] {row['size']}/{row['model']}\n{shlex.join(command)}", flush=True)
        log_path = log_dir / f"{row['size']}_{row['model']}.log"
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            payload["repo_root"]
            if not existing_pythonpath
            else payload["repo_root"] + os.pathsep + existing_pythonpath
        )
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=payload["repo_root"],
                                       env=environment, stdout=log,
                                       stderr=subprocess.STDOUT)
        if completed.returncode or not result.is_file():
            failures.append({"size": row["size"], "model": row["model"],
                             "returncode": completed.returncode, "log": str(log_path)})
            print(f"[FAILED] see {log_path}")
        else:
            provenance.write_text(json.dumps({
                "schema": "lusbench-lesion-task-provenance-v1",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "size": row["size"], "family": row["family"], "model": row["model"],
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": actual_checkpoint_hash,
                "data_root": payload["data_root"],
                "dataset_manifest_sha256": payload.get("dataset_manifest_sha256"),
                "command": command,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[COMPLETE] {result}")
    failure_path = log_dir / "failures.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    collect_completed(payload)
    return 2 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True,
                   help="Derived root produced by preprocess_gt_components.py")
    p.add_argument("--weights-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--env-config", type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--plan", type=Path, required=True)
    r = commands.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--sizes", type=int, nargs="+", choices=(224, 512))
    r.add_argument("--models", nargs="+")
    r.add_argument("--allow-partial-plan", action="store_true")
    r.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    return plan(args) if args.command == "plan" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
