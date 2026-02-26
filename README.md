
# SD-Trainer
LoRA-scripts (a.k.a SD-Trainer)

LoRA & Dreambooth training GUI & scripts preset & one-key training environment for [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts.git).

## Recent Updates / 更新说明

### 1) Resume 流程修复（可从中断步数继续）

- Resume now restores model/optimizer/scheduler/dataloader/random states from `*-state` folders correctly.
- To continue from an interrupted run (for example continue from `6/1800`), set `resume` to a state directory like:
  - `./output/<your_output_name>/<your_output_name>-000070-state`
- The state directory must contain `train_state.json`.
- `network_weights` only loads LoRA weights and starts a new schedule. It is not equal to full state resume.

Recommended resume usage:

1. Keep `train_data_dir`, `resolution`, `train_batch_size`, and `gradient_accumulation_steps` the same as the original run.
2. Set `resume` to the desired `*-state` directory.
3. Start training; progress should continue from recorded step/epoch.

### 2) 阶段分辨率训练（Staged Resolution Training）

- Added staged resolution training mode (512 -> 768 -> 1024).
- Base training resolution is fixed to `1024,1024` in this mode.
- The trainer auto-calculates phase batch size and equivalent epochs from the 1024-base setting.
- Target is equivalent pixel-level training budget (total compute budget stays close), while often improving detail consolidation.
- Phase ratios are configurable (0% to 100% each, sum <= 100%).

Core formula:

- `phase_batch = floor(base_batch * (1024*1024)/(phase_res*phase_res))`
- `raw_phase_epoch = ceil(base_epoch * phase_ratio * (phase_batch/base_batch))`
- `actual_phase_epoch = ceil_to_multiple(raw_phase_epoch, phase_save/sample alignment rule)`

### 3) TensorBoard 记录优化

- Run naming is improved for readability: `YYYY-MM-DD + model + _n`.
- Resume training merges into the same TensorBoard run.
- Runs that produce no checkpoint are removed from records.
- Logging continuity is improved for resumed training scenarios.

### 4) Torch 2.10 + Blackwell 显卡建议

- For Torch 2.10 on Blackwell GPUs, this repo prefers SDPA path and avoids xformers path.
- This reduces VRAM usage and improves stability on new architecture cards.
- Verified practical batch size guidance on RTX 5090:
  - Linux: up to batch size `24` (typical)
  - Windows: recommend `22` for stability margin


## Usage

### Required Dependencies

- Git
- NVIDIA driver and CUDA-capable GPU
- Internet access for dependency download
- `iperf3` (optional but recommended for mesh bandwidth checks in cluster compatibility test)

> Python installation is not required manually. Installer uses embedded Python 3.10 and creates `venv`.


## ✨ SD-Trainer GUI

### Windows

#### Installation

Run `install.ps1` to install embedded Python + create `venv` + install dependencies.

If you are in mainland China, use `install-cn.ps1`.

#### Start GUI

```powershell
.\run_gui.ps1
```

Then open `http://127.0.0.1:28000`.

### Linux

#### Installation

```bash
bash install.bash
```

#### Start GUI

```bash
bash run_gui.sh
```

Then open `http://127.0.0.1:28000`.

## Legacy training with scripts

### Windows

- Edit `train.ps1`, then run `./train.ps1`
- Edit `train_by_toml.ps1`, then run `./train_by_toml.ps1`

### Linux

- Edit `train.sh`, then run `bash train.sh`
- Edit `train_by_toml.sh`, then run `bash train_by_toml.sh`

`train*.sh` / `train*.ps1` use the project `venv` Python directly. Manual activation is not required.

## Cluster compatibility check (single + multi-node + mesh iperf3)

Unified checker script:

- `cluster_compat_check.py` (core)
- `cluster_compat_check.sh` (Linux launcher, runs in project `venv`)
- `cluster_compat_check.ps1` (Windows launcher, runs in project `venv`)

### Interactive full flow (recommended)

Linux:

```bash
bash cluster_compat_check.sh
```

Windows:

```powershell
.\cluster_compat_check.ps1
```

Flow:

1. Run environment check (Python/driver/torch/NCCL availability/network brief). No `nvcc` check.
2. Run single-node NCCL compatibility check.
3. Ask whether to continue multi-node compatibility check.
4. If `host` role is selected, input cluster size and test parameters; host starts waiting for workers.
5. Workers input host IP/hostname/domain and connect, then enter waiting state.
6. Host confirms and types `start`; NCCL distributed test starts.
7. Output NCCL compatibility result table.
8. Run `iperf3` pairwise mesh tests and output a bandwidth table.

### Manual mode examples

Env check only:

```bash
bash cluster_compat_check.sh --mode check-env
```

Single-node NCCL only:

```bash
bash cluster_compat_check.sh --mode single
```

Host mode:

```bash
bash cluster_compat_check.sh --mode host --cluster-size 2 --master-addr 192.168.50.219 --master-port 29500 --control-port 29610
```

Worker mode:

```bash
bash cluster_compat_check.sh --mode worker --host 192.168.50.219 --control-port 29610
```

All compatibility checks are now unified in `cluster_compat_check.py`.

## TensorBoard

Windows helper script:

```powershell
.\tensorboard.ps1
```

Starts TensorBoard at `http://127.0.0.1:6006` by default.

## Program arguments

| Parameter Name                | Type  | Default Value | Description                                      |
|-------------------------------|-------|---------------|--------------------------------------------------|
| `--host`                      | str   | "0.0.0.0"     | Hostname for the server                          |
| `--port`                      | int   | 28000         | Port to run the server                           |
| `--listen`                    | bool  | false         | Enable listening mode for the server             |
| `--skip-prepare-environment`  | bool  | false         | Skip the environment preparation step            |
| `--disable-tensorboard`       | bool  | false         | Disable TensorBoard                              |
| `--disable-tageditor`         | bool  | false         | Disable tag editor                               |
| `--tensorboard-host`          | str   | "0.0.0.0"     | Host to run TensorBoard                          |
| `--tensorboard-port`          | int   | 6006          | Port to run TensorBoard                          |
| `--localization`              | str   |               | Localization settings for the interface          |
