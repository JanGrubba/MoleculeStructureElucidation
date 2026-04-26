from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from nmrsolver import config as C
except Exception:
    try:
        from . import config as C
    except Exception:
        import config as C


def _default_storage(study_name: str) -> str:
    db_path = Path(C.LOG_DIR) / f"optuna_{study_name}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def _parse_gpu_ids(raw_gpu_ids: str | None) -> list[str]:
    if raw_gpu_ids is None or not raw_gpu_ids.strip():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            return [gpu_id.strip() for gpu_id in visible.split(",") if gpu_id.strip()]
        gpu_count = 0
        try:
            import torch

            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 0
        return [str(i) for i in range(gpu_count)]
    return [gpu_id.strip() for gpu_id in raw_gpu_ids.split(",") if gpu_id.strip()]


def _split_trials(total_trials: int, worker_count: int) -> list[int]:
    base = total_trials // worker_count
    remainder = total_trials % worker_count
    return [base + (1 if idx < remainder else 0) for idx in range(worker_count)]


def _build_worker_command(
    args: argparse.Namespace,
    worker_trials: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nmrsolver.train",
        args.mode,
        "--optuna-trials",
        str(worker_trials),
        "--optuna-study-name",
        args.study_name,
        "--optuna-storage",
        args.storage,
    ]

    if args.db is not None:
        command.extend(["--db", args.db])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.lr is not None:
        command.extend(["--lr", str(args.lr)])
    if args.test_fraction is not None:
        command.extend(["--test-fraction", str(args.test_fraction)])
    if args.optuna_timeout is not None:
        command.extend(["--optuna-timeout", str(args.optuna_timeout)])
    if args.restrict_carbons_to_peaks:
        command.append("--restrict-carbons-to-peaks")

    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one Optuna worker per GPU against shared storage.",
    )
    parser.add_argument(
        "mode",
        choices=["inverse", "forward", "both"],
        help="Which training mode each Optuna worker should run",
    )
    parser.add_argument(
        "--study-name",
        required=True,
        help="Shared Optuna study name used by all workers",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        required=True,
        help="Total number of trials across all workers",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Shared Optuna storage URL. Defaults to a sqlite DB under logs/.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids to dedicate to workers, e.g. '0,1,2,3'",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker count. Defaults to number of selected GPUs.",
    )
    parser.add_argument("--db", type=str, default=C.DB_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--optuna-timeout", type=int, default=None)
    parser.add_argument(
        "--restrict-carbons-to-peaks",
        action="store_true",
    )
    args = parser.parse_args()

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("No GPUs selected or visible for Optuna workers.")

    worker_count = args.workers or len(gpu_ids)
    if worker_count < 1:
        raise ValueError("Worker count must be at least 1.")
    if worker_count > len(gpu_ids):
        raise ValueError(
            f"Requested {worker_count} workers but only {len(gpu_ids)} GPU ids were provided."
        )

    args.storage = args.storage or _default_storage(args.study_name)
    per_worker_trials = _split_trials(args.optuna_trials, worker_count)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    launcher_log_dir = Path(C.LOG_DIR) / f"optuna_workers_{args.study_name}_{timestamp}"
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, str, subprocess.Popen[bytes], object]] = []
    print(f"[Launcher] Study: {args.study_name}")
    print(f"[Launcher] Storage: {args.storage}")
    print(f"[Launcher] GPUs: {gpu_ids[:worker_count]}")
    print(f"[Launcher] Trial split: {per_worker_trials}")
    print(f"[Launcher] Logs: {launcher_log_dir}")

    # Initialize Optuna storage and create the study from the launcher process
    # to avoid concurrent Alembic/DB migrations when multiple workers start.
    try:
        import optuna

        print("[Launcher] Initializing Optuna storage and study (parent process)...")
        optuna.create_study(
            study_name=args.study_name, storage=args.storage, load_if_exists=True
        )
        print("[Launcher] Optuna study ready")
    except Exception as e:
        # Log and continue; workers will still attempt to use the storage.
        print(f"[Launcher] Warning: failed to pre-initialize Optuna study: {e}")

    try:
        for worker_idx in range(worker_count):
            worker_trials = per_worker_trials[worker_idx]
            if worker_trials <= 0:
                continue

            gpu_id = gpu_ids[worker_idx]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONUNBUFFERED"] = "1"
            # Ensure child processes use UTF-8 for IO to avoid UnicodeEncodeError
            env["PYTHONIOENCODING"] = "utf-8"
            # Also enable the UTF-8 mode for interpreters that support it
            env["PYTHONUTF8"] = "1"
            command = _build_worker_command(args, worker_trials)
            log_path = launcher_log_dir / f"worker_{worker_idx:02d}_gpu_{gpu_id}.log"
            log_handle = log_path.open("wb")
            process = subprocess.Popen(
                command,
                cwd=C.PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((worker_idx, gpu_id, process, log_handle))
            print(
                f"[Launcher] Started worker {worker_idx} on GPU {gpu_id} "
                f"for {worker_trials} trial(s) -> {log_path}"
            )

        failed = False
        for worker_idx, gpu_id, process, log_handle in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code != 0:
                failed = True
                print(
                    f"[Launcher] Worker {worker_idx} on GPU {gpu_id} failed with exit code {return_code}"
                )
            else:
                print(
                    f"[Launcher] Worker {worker_idx} on GPU {gpu_id} finished successfully"
                )

        if failed:
            raise SystemExit(1)
    except KeyboardInterrupt:
        print("[Launcher] Interrupted; terminating workers...")
        for _, _, process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
            log_handle.close()
        raise


if __name__ == "__main__":
    main()
