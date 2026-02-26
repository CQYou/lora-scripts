import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import toml

from mikazuki.log import log


MIXED_RESOLUTION_RESUME_SENTINEL = "__MIXED_AUTO_RESUME__"
DATASET_DIR_KEYS = ("train_data_dir", "reg_data_dir")


def _resolve_local_path(path_value: str, repo_root: Path) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _clear_dataset_npz_cache_by_config(config: dict, repo_root: Path):
    total_removed = 0
    for key in DATASET_DIR_KEYS:
        value = str(config.get(key, "") or "").strip()
        if not value:
            continue
        local_dir = _resolve_local_path(value, repo_root)
        if not local_dir.exists() or not local_dir.is_dir():
            continue

        removed = 0
        for npz_file in local_dir.rglob("*.npz"):
            try:
                npz_file.unlink()
                removed += 1
            except Exception as e:
                raise RuntimeError(f"删除缓存失败: {npz_file} ({e})") from e

        total_removed += removed
        log.info(f"[staged-resolution] cache reset: {key}, removed={removed}, dir={local_dir}")

    log.info(f"[staged-resolution] cache reset finished, total npz removed={total_removed}")


def _find_latest_state_dir(config: dict, repo_root: Path) -> Optional[Path]:
    output_dir = _resolve_local_path(str(config.get("output_dir", "./output") or "./output"), repo_root)
    if not output_dir.exists() or not output_dir.is_dir():
        return None

    output_name = str(config.get("output_name", "") or "").strip()
    candidates = []
    for entry in output_dir.glob("*-state"):
        if not entry.is_dir():
            continue
        if output_name and not entry.name.startswith(f"{output_name}-"):
            continue
        state_file = entry / "train_state.json"
        step_num = -1
        epoch_num = -1
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                step_num = int(data.get("current_step", -1))
                epoch_num = int(data.get("current_epoch", -1))
            except Exception:
                pass
        if step_num < 0 or epoch_num < 0:
            # fallback for legacy folders without readable train_state
            match = re.search(r"-(\d+)-state$", entry.name)
            epoch_num = int(match.group(1)) if match else -1
        try:
            mtime = entry.stat().st_mtime
        except Exception:
            mtime = 0
        candidates.append((step_num, epoch_num, mtime, entry))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


def _load_toml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return toml.load(f)


def _write_toml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(data, f)


def _build_phase_command(trainer_file: str, toml_file: str, cpu_threads: int, launch_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        str(cpu_threads),
        "--quiet",
        *launch_args,
        trainer_file,
        "--config_file",
        toml_file,
    ]


def _run_mixed_plan(plan: dict) -> int:
    trainer_file = str(plan["trainer_file"])
    cpu_threads = int(plan.get("cpu_threads", 2) or 2)
    launch_args = [str(x) for x in plan.get("launch_args", [])]
    phases = list(plan.get("phases", []))
    repo_root = Path(plan.get("repo_root", ".")).resolve()

    if not phases:
        log.error("[staged-resolution] no phase in plan")
        return 2

    auto_resume_state_dir = ""
    prev_resolution = None

    for phase in phases:
        phase_index = int(phase["phase_index"])
        phase_toml = Path(str(phase["toml_path"])).resolve()
        if not phase_toml.exists():
            log.error(f"[staged-resolution] phase toml not found: {phase_toml}")
            return 2

        phase_config = _load_toml(phase_toml)
        phase_resolution = str(phase.get("resolution", phase_config.get("resolution", "")) or "").strip()

        if phase.get("clear_cache_before_start", False) and prev_resolution is not None and phase_resolution != prev_resolution:
            log.info(
                f"[staged-resolution] phase {phase_index}: resolution switched "
                f"{prev_resolution} -> {phase_resolution}, reset dataset npz cache"
            )
            _clear_dataset_npz_cache_by_config(phase_config, repo_root)

        resume_value = str(phase_config.get("resume", "") or "").strip()
        if resume_value == MIXED_RESOLUTION_RESUME_SENTINEL:
            if not auto_resume_state_dir:
                log.error(
                    f"[staged-resolution] phase {phase_index}: resume sentinel found but previous phase state is missing"
                )
                return 2
            phase_config["resume"] = auto_resume_state_dir
            phase_config["resume_epoch_offset"] = 1
            _write_toml(phase_toml, phase_config)
            log.info(f"[staged-resolution] phase {phase_index}: auto resume from {auto_resume_state_dir}")

        log.info(
            f"[staged-resolution] phase {phase_index}/{len(phases)} start: "
            f"res={phase.get('resolution')} ratio_percent={phase.get('ratio_percent')} "
            f"batch={phase.get('batch_size')} "
            f"save_every_n_epochs={phase.get('save_every_n_epochs')} "
            f"sample_every_n_epochs={phase.get('sample_every_n_epochs')} "
            f"epochs={phase.get('epochs')} "
            f"phase_steps={phase.get('phase_steps')} target_max_steps={phase.get('target_max_train_steps')} "
            f"target_epoch_end={phase.get('target_epoch_end')}"
        )
        log.info(
            f"[staged-resolution] phase {phase_index} formulas: "
            f"raw='{phase.get('raw_epochs_formula')}', actual='{phase.get('actual_epochs_formula')}'"
        )
        cmd = _build_phase_command(trainer_file, str(phase_toml), cpu_threads, launch_args)
        proc = subprocess.Popen(cmd, env=os.environ.copy())
        return_code = proc.wait()
        if return_code != 0:
            log.error(f"[staged-resolution] phase {phase_index} failed with code={return_code}")
            return return_code

        prev_resolution = phase_resolution
        is_last_phase = phase_index >= len(phases)
        if is_last_phase:
            log.info(f"[staged-resolution] phase {phase_index} finished (last phase)")
            continue

        phase_config_after = _load_toml(phase_toml)
        latest_state_dir = _find_latest_state_dir(phase_config_after, repo_root)
        if latest_state_dir is None:
            log.error(
                "[staged-resolution] cannot find latest state after phase "
                f"{phase_index}. 请确认 save_state 已启用并且 save_every_n_epochs 配置正确。"
            )
            return 2

        auto_resume_state_dir = str(latest_state_dir)
        log.info(
            f"[staged-resolution] phase {phase_index} finished, "
            f"resume state={auto_resume_state_dir}, next phase will apply resume_epoch_offset=1"
        )

    log.info("[staged-resolution] all phases finished")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Staged-resolution phase runner")
    parser.add_argument("--plan-file", required=True, help="Path to staged-resolution plan json")
    args = parser.parse_args()

    plan_file = Path(args.plan_file).resolve()
    if not plan_file.exists():
        log.error(f"[staged-resolution] plan file not found: {plan_file}")
        sys.exit(2)

    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"[staged-resolution] failed to parse plan file: {plan_file} ({e})")
        sys.exit(2)

    try:
        code = _run_mixed_plan(plan)
    except Exception as e:
        log.error(f"[staged-resolution] runner fatal error: {e}")
        code = 2
    sys.exit(int(code))


if __name__ == "__main__":
    main()
