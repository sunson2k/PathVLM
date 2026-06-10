# PathVLM Agent Guide

This repository is a PyTorch project for predicting gene expression from spatial transcriptomics inputs. Use this file as the first project-specific orientation when making code changes.

## Working Defaults

- Work from the repository root: `C:\Users\sunso\github\PathVLM`.
- Prefer the repo virtual environment for runtime checks:
  `.\.venv\Scripts\python.exe`.
- Do not assume `python` on `PATH` is the correct interpreter.
- Treat `configs\run_config.json` as the active runtime config.
- Keep user-facing switches config-driven when practical, especially model choices, ablations, data targets, paths, and training knobs.
- Use `rg`/`rg --files` for repo search.

## Project Map

- `configs\run_config.json`: active local runtime settings.
- `src\config.py`: config loading, path normalization, and config-derived behavior.
- `src\data_loaders.py`: datasets, feature loading, StandardScaler fitting, and tensor conversion.
- `src\models.py`: image, visual, and multimodal model definitions.
- `src\training.py`: device resolution, training loop, checkpoints, history, and logging.
- `src\evaluation.py`: metrics and prediction outputs.
- `src\visualization.py`: loss plots and report images.
- `src\results_discovery.py`: result-folder and history discovery.
- `scripts\01_prepare_data.py`: split creation and alignment checks.
- `scripts\02_train_image.py`: image model training.
- `scripts\03_train_visual.py`: visual embedding model training.
- `scripts\04_train_multimodal.py`: multimodal model training.
- `scripts\05_summarize_results.py`: `results\summary.md` plus loss-curve PNGs.
- `scripts\run_pipeline.py`: end-to-end orchestration.

## Common Commands

```powershell
.\.venv\Scripts\python.exe -m compileall scripts src
.\.venv\Scripts\python.exe scripts\05_summarize_results.py
.\.venv\Scripts\python.exe scripts\05_summarize_results.py --y-min 0.3 --y-max 0.4
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

Use `py -m py_compile <file>` only for quick syntax checks when the venv is not needed.

## Configuration Notes

- `data_root` and `project_root` may be relative paths; code should resolve them from the project root, not the shell's current working directory.
- `expr_target` selects the expression folder, currently `8n` or `spcs`.
- `model.resnet_freeze_mode` choices are `none`, `early`, and `all`.
- `model.dnn_normalization` choices are `batchnorm`, `layernorm`, and `none`.
- `model.multimodal_model` choices are `concat`, `cross_attention`, and `gmu`.
- The three training entrypoints should share behavior through `src\` helpers rather than duplicating logic.

## Model And Training Expectations

- The shared DNN head is implemented through `MultiLayerDNN`.
- Image mode uses a ResNet backbone with a configurable freeze mode.
- Visual mode consumes 1024-dimensional visual embeddings.
- Multimodal mode consumes 1536-dimensional concatenated visual and text embeddings.
- GMU fusion should preserve the configured tanh projection plus sigmoid gate behavior.
- Training status should remain concise: one clear status line per epoch is preferred.
- Saved `history.json` should describe the current run, not silently append stale history from an older run.

## Data Loading Notes

- Fit scalers on training data only.
- Avoid per-row CSV reads or `iterrows()` in performance-sensitive loading paths.
- When converting NumPy arrays to tensors, prefer writable `float32` arrays before `torch.from_numpy(...)` to avoid non-writable array warnings.
- If multimodal or visual startup is slow, inspect scaler fitting and grouped CSV loading before assuming model training is slow.

## Windows And CUDA Notes

- CUDA checks should use the venv Python.
- `WinError 1455` during CUDA DLL load, especially involving `cufft64_10.dll`, usually indicates Windows commit/pagefile pressure during DLL loading rather than a normal model VRAM out-of-memory error.
- Useful mitigations to consider before changing architecture: pagefile size, `CUDA_MODULE_LOADING=LAZY`, lower `training.batch_size`, and lower `training.num_workers`.
- If `uv run` hits cache permission errors under `C:\Users\sunso\AppData\Local\uv\cache`, use the repo venv directly.

## Result Reporting

- Summary outputs live under `results\`:
  - `summary.md`
  - `loss_comparison.png`
  - `loss_per_model.png`
- `scripts\05_summarize_results.py` discovers result folders automatically and excludes `results\old`.
- Use `--y-min` and `--y-max` when fixed y-axis bounds are needed for comparison plots.
- Local Markdown image links may not render inside Codex chat; use absolute image paths when showing PNGs in the conversation.

## Change Discipline

- Keep edits scoped to the requested behavior.
- Prefer shared code in `src\` over parallel edits in `02_train_image.py`, `03_train_visual.py`, and `04_train_multimodal.py`.
- Do not preserve deprecated knobs just for backward compatibility unless the user asks for it.
- For repo-specific questions, inspect the current files, generated artifacts, or command output instead of answering from convention.
