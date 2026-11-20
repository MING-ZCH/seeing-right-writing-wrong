# Reproducing the tables and figures

This page maps every table and figure of the paper to the script, command and inputs that produce it, and documents the CPU check that regenerates them from the shipped per-item outputs. It covers the main-paper tables and figures (Tab. 1-2, Fig. 1-6) and 15 appendix tables (Tab. S1-S15). The repository also contains the per-item model outputs (compressed, unused fields removed), the aggregated result files (`outputs/bundle/`), the synthetic stimulus sets, and the copies of the tables and figure data the paper was compiled from (`reference/`).

Table and figure labels below follow the paper version these files were built for; other versions (e.g. arXiv) may number them differently. The LaTeX label in the second column is the stable identifier.

`docs/MANIFEST.md` / `docs/MANIFEST.json` list every file with its sha256 and map each table and figure to its script, command and inputs.

## Quick start (CPU, no GPU or network needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
source env.example.sh
python tools/unpack_outputs.py      # decompress outputs/**/*.gz next to the .gz files
python tools/regen_tables.py        # runs every generator below, then tools/check_tables.py
```

`regen_tables.py` writes the tables and figures to `regenerated/` (logs in `regenerated/logs/`) and `check_tables.py` writes `regenerated/check_report.json`. A single item: `python tools/regen_tables.py --only rigor_e60`; `--list` prints all commands. All commands are run from the repository root.

Some generators rewrite their bundle file in `outputs/bundle/` (see below); `regen_tables.py` saves the shipped copies to `regenerated/bundle_shipped/` first. Run `git checkout -- outputs/bundle` to restore the shipped files afterwards.

## Paper tables and figures

| Paper | Label | Generator (run from the repository root) | Writes | Per-item inputs | Shipped |
|---|---|---|---|---|---|
| Tab. 1 | `tab:behavioral_main` | hand-built | - | bundle only | bundle; hand-maintained table; every number is read from outputs/bundle/numbers.json (build_bundle.load_main) |
| Tab. 2 | `tab:profile` | `python scripts/paper_tools/make_e27_tables.py` | `regenerated/tables/tab2_model_profile.tex` | bundle only | yes |
| Fig. 1 | `fig:teaser` | `python scripts/make_figures_v3.py` | `regenerated/figures/fig1_teaser.pdf` | 19 files (e2b, e2d, e4, items, legacy_v2) | yes |
| Fig. 2 | `fig:error_onset` | `python scripts/make_figures_v3.py` | `regenerated/figures/fig2_error_onset.pdf` | 19 files (e2b, e2d, e4, items, legacy_v2) | yes |
| Fig. 3 | `fig:cues` | `python scripts/make_fig_cues.py` | `regenerated/figures/fig_cues.pdf` | bundle only | yes |
| Fig. 4 | `fig:swaps` | `python scripts/make_figures_v3.py` | `regenerated/figures/fig3_swaps.pdf` | 19 files (e2b, e2d, e4, items, legacy_v2) | yes |
| Fig. 5 | `fig:coco_examples` | `python scripts/e62_fig.py` | `regenerated/figures/supp_e62_examples.pdf` | 1 files (e62) | yes; draws 5 COCO-composite twin pairs; the composites are not shipped (COCO images are third-party): regenerate them with scripts/e62_stimuli.py from COCO val2017, then run the command |
| Fig. 6 | `fig:two_channel` | `python scripts/rigor_e32.py --only fig` | `regenerated/figures/fig_two_channel.pdf` | bundle only | yes |
| Tab. S1 | `tab:supp_evidence` | hand-built | - | - | -; evidence map written by hand (text only, no computed numbers) |
| Tab. S2 | `tab:supp_tasks` | `python scripts/paper_tools/make_supp_task_table.py` | `regenerated/tables/tab_s_tasks.tex` | bundle only | yes; paper table = generator output + hand-added pooled-row columns Fil+t-Direct and Fil+t-Fil from `outputs/bundle/numbers.json` |
| Tab. S3 | `tab:supp_e54a` | `python scripts/rigor_e54.py --only a` | `regenerated/tables/supp_e54a.tex` | 9 files (e10, e2d, e45, e54) | yes |
| Tab. S4 | `tab:supp_e57` | `python scripts/rigor_e57.py` | `regenerated/tables/supp_e57.tex` | 37 files (e10, e17, e28, e2b, e2d, e4, e45, e57) | yes |
| Tab. S5 | `tab:supp_e59` | `python scripts/rigor_e59.py` | `regenerated/tables/supp_e59.tex` | 43 files (e52, e59) | yes |
| Tab. S6 | `tab:supp_e60` | `python scripts/rigor_e60.py` | `regenerated/tables/supp_e60.tex` | 43 files (e3x, e4, e60, e6x, legacy_v2) | yes |
| Tab. S7 | `tab:supp_e62` | `python scripts/rigor_e62.py` | `regenerated/tables/supp_e62.tex` | 14 files (e62) | yes |
| Tab. S8 | `tab:supp_e61` | `python scripts/rigor_e61.py` | `regenerated/tables/supp_e61.tex` | 14 files (e61, e9h) | yes; paper table = generator output + hand-added edits (Gemini-4-Argon gray-image cells printed as '--'; math-mode minus signs) from `outputs/bundle/e61_closed_fixed_rcut.json` |
| Tab. S9 | `tab:supp_e50` | `python scripts/e50_closed_image_blind.py analyze` | `regenerated/tables/supp_e50.tex` | 23 files (e38, e45, e50, e9h) | yes; paper table = generator output + hand-added columns / footer from `outputs/bundle/e50_closed_image_blind.json` |
| Tab. S10 | `tab:supp_e45` | `python scripts/e45_uniform_two_channel.py analyze` | `regenerated/tables/supp_e45.tex` | 14 files (e35, e38, e45) | yes; paper table = generator output + hand-added footer, half-up rounded cells and the ceiling-model row from `outputs/bundle/e45_uniform_two_channel.json` |
| Tab. S11 | `tab:supp_e35` | `python scripts/e35_natural_two_channel.py analyze` | `regenerated/tables/supp_e35.tex` | 40 files (e10, e1x, e2b, e2d, e35, e4) | yes; paper table = generator output + hand-added footer (incl. the synthetic fit of Tab. S10) from `outputs/bundle/e35_natural_two_channel.json`, `outputs/bundle/e45_uniform_two_channel.json` |
| Tab. S12 | `tab:supp_e58b` | `python scripts/rigor_e58.py --only b` | `regenerated/tables/supp_e58b.tex` | 11 files (e1x, e4, e45) | yes |
| Tab. S13 | `tab:supp_e31` | `python scripts/rigor_e31.py --table-only` | `regenerated/tables/supp_e31.tex` | bundle only | yes; paper table = generator output + hand-added height-AME row from `outputs/bundle/e31_cue_ladder.json` |
| Tab. S14 | `tab:supp_e43` | `python scripts/rigor_e43.py --table-only` | `regenerated/tables/supp_e43.tex` | bundle only | yes; paper table = generator output + hand-added height-AME row from `outputs/bundle/e43_big_open.json` |
| Tab. S15 | `tab:supp_e48` | `python scripts/rigor_e48.py --table-only` | `regenerated/tables/supp_e48.tex` | bundle only | yes; paper table = generator output + hand-added height-AME row and footer from `outputs/bundle/e48_reasoning_open.json` |

Tab. S13-S15 are built from bundle files; the commands that rebuild those bundle files from per-item outputs (about 5-7 min each on 4 CPU threads) are not run by default:

- `python scripts/rigor_e31.py`
- `python scripts/rigor_e43.py`
- `python scripts/rigor_e48.py`

Several appendix tables in the paper are condensed versions of the generated ones (columns or rows deleted to fit the page, cells not edited), so `check_tables.py` checks that every number printed in the paper table occurs in the regenerated table; byte-identical tables are reported as such. Figures are compared through the `*.data.json` file each figure script writes next to the PDF (the plotted values).

The following paper tables are generator output plus rows / footers added by hand after an audit; the shipped generators do not write these parts yet, so their numbers are looked up in the output file listed (second level of `check_tables.py`):

- Tab. S2 (`tab:supp_tasks`): generator output + hand-added pooled-row columns Fil+t-Direct and Fil+t-Fil from `outputs/bundle/numbers.json`
- Tab. S8 (`tab:supp_e61`): generator output + hand-added edits (Gemini-4-Argon gray-image cells printed as '--'; math-mode minus signs) from `outputs/bundle/e61_closed_fixed_rcut.json`
- Tab. S9 (`tab:supp_e50`): generator output + hand-added columns / footer from `outputs/bundle/e50_closed_image_blind.json`
- Tab. S10 (`tab:supp_e45`): generator output + hand-added footer, half-up rounded cells and the ceiling-model row from `outputs/bundle/e45_uniform_two_channel.json`
- Tab. S11 (`tab:supp_e35`): generator output + hand-added footer (incl. the synthetic fit of Tab. S10) from `outputs/bundle/e35_natural_two_channel.json`, `outputs/bundle/e45_uniform_two_channel.json`
- Tab. S13 (`tab:supp_e31`): generator output + hand-added height-AME row from `outputs/bundle/e31_cue_ladder.json`
- Tab. S14 (`tab:supp_e43`): generator output + hand-added height-AME row from `outputs/bundle/e43_big_open.json`
- Tab. S15 (`tab:supp_e48`): generator output + hand-added height-AME row and footer from `outputs/bundle/e48_reasoning_open.json`

### Result of this check

Run on CPU (Python 3.11.6, 4 threads), about 165 s in total:

Columns: status of `check_tables.py` against the paper copy; whether the regenerated file is identical to the output of the original scripts on the original (unstripped) outputs; paper numbers found.

| Paper | Generator | vs. paper copy | vs. original scripts | Paper numbers found |
|---|---|---|---|---|
| Tab. 1 | `hand-built` | hand-built; all numbers found in bundle | - | 93 numbers |
| Tab. 2 | `python scripts/paper_tools/make_e27_tables.py` | all paper numbers in regenerated table or bundle | identical | 98/111 in regenerated table; rest in bundle |
| Fig. 1 | `python scripts/make_figures_v3.py` | data.json identical (plotted values; provenance paths differ) | identical | - |
| Fig. 2 | `python scripts/make_figures_v3.py` | data.json identical | identical | - |
| Fig. 3 | `python scripts/make_fig_cues.py` | data.json identical | identical | - |
| Fig. 4 | `python scripts/make_figures_v3.py` | data.json identical | identical | - |
| Fig. 5 | `python scripts/e62_fig.py` | not regenerated (draws 5 COCO-composite twin pairs; the composites are not shipped (COCO images are third-party): regenerate them with scripts/e62_stimuli.py from COCO val2017, then run the command) | not regenerated | - |
| Fig. 6 | `python scripts/rigor_e32.py --only fig` | data.json identical | identical | - |
| Tab. S1 | `hand-built` | hand-built text table (numbers not checked) | - | 96 numbers |
| Tab. S2 | `python scripts/paper_tools/make_supp_task_table.py` | all paper numbers in regenerated table or bundle | identical | 401/526 in regenerated table; rest in bundle |
| Tab. S3 | `python scripts/rigor_e54.py --only a` | all paper numbers reproduced | identical | 161/161 in regenerated table |
| Tab. S4 | `python scripts/rigor_e57.py` | all paper numbers reproduced | identical | 205/205 in regenerated table |
| Tab. S5 | `python scripts/rigor_e59.py` | all paper numbers reproduced | identical | 170/170 in regenerated table |
| Tab. S6 | `python scripts/rigor_e60.py` | identical | identical | 111/111 in regenerated table |
| Tab. S7 | `python scripts/rigor_e62.py` | identical | identical | 174/174 in regenerated table |
| Tab. S8 | `python scripts/rigor_e61.py` | identical | identical | 134/134 in regenerated table |
| Tab. S9 | `python scripts/e50_closed_image_blind.py analyze` | all paper numbers in regenerated table or bundle | identical | 209/227 in regenerated table; rest in bundle |
| Tab. S10 | `python scripts/e45_uniform_two_channel.py analyze` | all paper numbers in regenerated table or bundle | identical | 477/486 in regenerated table; rest in bundle |
| Tab. S11 | `python scripts/e35_natural_two_channel.py analyze` | all paper numbers in regenerated table or bundle | identical | 249/272 in regenerated table; rest in bundle |
| Tab. S12 | `python scripts/rigor_e58.py --only b` | all paper numbers reproduced | identical | 282/282 in regenerated table |
| Tab. S13 | `python scripts/rigor_e31.py --table-only` | all paper numbers in regenerated table or bundle | identical | 59/61 in regenerated table; rest in bundle |
| Tab. S14 | `python scripts/rigor_e43.py --table-only` | all paper numbers in regenerated table or bundle | identical | 61/63 in regenerated table; rest in bundle |
| Tab. S15 | `python scripts/rigor_e48.py --table-only` | all paper numbers in regenerated table or bundle | identical | 218/222 in regenerated table; rest in bundle |

Some generators also rewrite their bundle file in `outputs/bundle/`; compared with the shipped copy: `e35_natural_two_channel.json` equal up to float noise / NaN; `e45_uniform_two_channel.json` equal up to float noise / NaN; `e48_reasoning_open.json` JSON-equal; `e50_closed_image_blind.json` differs; `e54_decomp.json` JSON-equal; `e58_leverage.json` equal up to float noise / NaN; `e61_closed_fixed_rcut.json` differs. The shipped copies are the files of the original runs (`regen_tables.py` keeps them in `regenerated/bundle_shipped/`). In `e50_closed_image_blind.json` and `e61_closed_fixed_rcut.json` only the keys of the tally of failed API calls differ (error messages of the API service are replaced by short tags in the shipped outputs); other differences are floats in the last digit or NaN fields.

## Layout

| Path | Content |
|---|---|
| `scripts/` | experiment, judge and analysis scripts (`vtax_paths.py`: all locations, derived from `VTAX_ROOT`) (152 files, 2.4 MB) |
| `scripts/legacy_v2/` | protocol module of the original runs (`common.py`: data loading, prompts, filler, top-20 readout) and the generators of `outputs/legacy_v2/` (4 files, 0.0 MB) |
| `scripts/paper_tools/` | generators of Tab. 2 and Tab. S2 (3 files, 0.0 MB) |
| `scripts/api_client.py` | minimal client for an OpenAI-compatible chat-completions endpoint (closed models, LLM judges) |
| `outputs/bundle/` | aggregated results (JSON) read by the table / figure scripts (184 files, 16.6 MB) |
| `outputs/<exp>/` | per-item outputs (`.jsonl.gz` / `.json.gz`), one directory per experiment (216 files, 57.0 MB) |
| `outputs/legacy_v2/` | per-item outputs of the original protocol runs (main pool, five models) (9 files, 2.1 MB) |
| `data/items/` | synthetic sets: occlusion twins, difficulty-calibrated hard set, cue-factorial set (JSONL + PNG) (1215 files, 6.1 MB) |
| `data/coco_composites/` | COCO val2017 composite twins: pairs, paste parameters, items, sha256 of every image (no images) (4 files, 1.9 MB) |
| `data/benchmark_ids/` | item ids of the public benchmarks used (no benchmark content) (3 files, 0.1 MB) |
| `reference/` | the table files and figure data the paper was compiled from (22 files, 0.1 MB) |
| `tools/` | `unpack_outputs.py`, `regen_tables.py`, `check_tables.py`, `targets.json` (4 files, 0.0 MB) |
| `requirements/` | pinned environments (7 files, 0.0 MB) |

Paths stored in the released outputs that pointed to the original storage are replaced by placeholders (`<DATA_ROOT>/datasets/...`, `<VTAX_ROOT>/outputs/...`); `vtax_paths.resolve` maps them back. The CPU analysis does not read these paths. Fields of the per-item records that none of the shipped analyses reads were removed (the readouts, rationales, judge labels and ids that the tables use are kept).

## Environments

| File | Used for |
|---|---|
| `requirements/requirements_cpu_analysis.txt` | CPU regeneration of all tables / figures (versions used for the check above; Python 3.11) |
| `requirements/requirements_main.txt` (+ `requirements_main_vllm.txt`: vLLM 0.8.4) | Qwen2.5-VL-3B/7B, InternVL3-8B, Phi-3.5-Vision, Molmo-7B-D: PyTorch 2.6.0, Transformers 4.51.3 |
| `requirements/requirements_qwen3vl.txt` (+ `requirements_qwen3vl_vllm.txt`: vLLM 0.11.0) | Qwen3-VL (2B-32B, Instruct / Thinking), InternVL3.5-8B/38B: PyTorch 2.8.0, Transformers 4.57.1 |
| `requirements/requirements_legacy_v2.txt` | original protocol runs (`scripts/legacy_v2/`) |

`scripts/setup_envs.sh main|qwen3vl` creates the two GPU environments. GPU runs used bfloat16, greedy decoding and the vLLM engine settings in `scripts/vx.py` (`make_llm`): the V0 engine with a logits processor for the exact full-vocabulary option readout at `Answer: (`; Qwen3-VL, InternVL3.5 and Molmo on the V1 engine with the top-20 readout. Models are loaded from `$VTAX_DATA_ROOT/models/<Hugging Face repo name>` (e.g. `Qwen2.5-VL-7B-Instruct`, `InternVL3-8B`, `Phi-3.5-vision-instruct`, `Molmo-7B-D-0924`, `Qwen3-VL-8B-Instruct`, `InternVL3_5-8B`, `MiMo-VL-7B-RL`, `GLM-4.1V-9B-Thinking`).

## How the GPU and API experiments were run

The per-item outputs in `outputs/` were produced by the runner scripts below (GPU: vLLM, one or two GPUs per model; API: `scripts/api_client.py`). They need the model checkpoints and benchmark data under `$VTAX_DATA_ROOT` and are not needed for the CPU regeneration. Re-running a runner overwrites the shipped outputs of that experiment.

| Outputs | Experiment | Runner / judge scripts |
|---|---|---|
| `outputs/e1x/` | readouts of all conditions on the 2,850-item binary spatial pool (open models, exact full-vocabulary readout) | `e1x.py` |
| `outputs/e2b/` | commit trajectories: forced readout after each rationale sentence | `e2b.py`, `e2b_prefix_trajectory.py` |
| `outputs/e2d/` | occlusion twins and cue-factorial set: Direct / CoT readouts and identity-swapped twins | `e2d.py`, `e2d_think.py` |
| `outputs/e3x/` | fixed-rationale twin image swaps (stored rationale R, R with the conclusion removed) | `e3x.py` |
| `outputs/e4/` | Qwen3-VL-8B / InternVL3.5-8B main-pool generations (vLLM V1) | `e4.py` |
| `outputs/e6x/` | held-out pool (BLINK, 3DSRBench) under the frozen protocol | `e6x.py` |
| `outputs/e10/` | cue-factorial occlusion set: Direct / CoT readouts, written relations | `e10_state.py`, `e13_cue_analysis.py`, `e16_cue_judge.py` |
| `outputs/e28/` | sentence-level pipeline (commit sentences) on the cue set | `e28_judge.py`, `e28_run.py` |
| `outputs/e35/` | natural depth questions: two-channel test | `e35_natural_two_channel.py` |
| `outputs/e38/` | closed models: written relation and answer | `e38_closed_judge.py` |
| `outputs/e45/` | two-channel analysis under one conclusion-removal protocol (two text-only judges) | `e45_analyze.py`, `e45_uniform_two_channel.py` |
| `outputs/e50/` | closed models under the real, a gray and the twin image | `e50_analyze.py`, `e50_closed_image_blind.py`, `e50_e38_gemini_fix.py`, `e50_judge.py` |
| `outputs/e52/` | attention knockouts (full rationale) | `e52_knockout.py` |
| `outputs/e54/` | decomposition of CoT losses and rescues (judge labels) | `e54_judge.py` |
| `outputs/e57/` | sentence-level pipeline on Qwen3-VL-32B | `e57_judge.py`, `e57_probe.py`, `e57_run.py` |
| `outputs/e59/` | image knockouts after writing, conclusion removed | `e59_knockout.py` |
| `outputs/e60/` | fixed rationale, mirrored natural image (CV-Bench 2D Relation left/right) | `e60_items.py`, `e60_judge.py`, `e60_run.py` |
| `outputs/e61/` | closed models with their own conclusion-removed rationale held fixed | `e61_fixed_rcut.py` |
| `outputs/e62/` | natural-object occlusion twins from COCO val2017 composites | `e62_fig.py`, `e62_judge.py`, `e62_run.py`, `e62_stimuli.py` |
| `outputs/e9h/` | closed models on the difficulty-calibrated occlusion set | `e9x_api.py` |

Closed models and LLM judges were called through an OpenAI-compatible chat-completions endpoint: set `OPENAI_COMPAT_BASE_URL` (e.g. `https://<host>/v1`) and `OPENAI_COMPAT_API_KEY`; `api_client.chat` sends `reasoning_effort` (`thinking_config.thinking_level` for Gemini models) and records the reply, the reasoning text and the token usage. The LLM judges of rationale sentences were Claude Opus 5.5 (`claude-opus-5-5`) and GPT-6-Astra (`gpt-6-astra`) through such an endpoint; judge prompts are in the `*judge*.py` scripts and every judge reply used in the paper is in the shipped outputs (`outputs/*/judge_*.jsonl.gz`). The six closed models evaluated in the paper are referred to by their API model ids in the outputs (`claude-opus-5-5`, `gpt-6-astra`, `gemini-4-argon`, `qwen3.8-max-0902`, `hy4`, `deepseek-v4.1-flash`).

## Data

- Synthetic sets (`data/items/<set>/`: items JSONL, generation report, `images_sha256.json`). Each set was regenerated with its generator while building this release and compared with the files of the original runs (byte-identical images and item files; Python 3.11, numpy 2.2.6, Pillow 11.2.1):

  | Set | Images | Shipped | Command |
  |---|---|---|---|
  | occlusion twins (`occlusion/`) | 400 | PNG images | `python scripts/e2_occlusion_set.py --out-dir data/items/occlusion` |
  | cue-factorial occlusion set (`occ_cues/`) | 800 | PNG images | `python scripts/e12_cue_set.py --out-dir data/items/occ_cues --n 400` |
  | difficulty-calibrated hard occlusion set (`occlusion_hard/`) | 288 | sha256 only (noise-textured PNGs, 150 MB; regenerate with the command) | `python scripts/e2_occlusion_hard.py --out-dir data/items/occlusion_hard --n 144 --levels L3,L3b,L4` |

- COCO composites (Exp-62, 466 images): not shipped, because they contain COCO pixels. `scripts/e62_stimuli.py select|download|build|sheets|finalize` rebuilds them from COCO val2017 (images and `instances_val2017.json` under `$VTAX_DATA_ROOT/datasets/coco2017_val/`); `data/coco_composites/` has the COCO ids and paste parameters (`pairs.jsonl`, `candidates.json`), the questions (`items_final.jsonl`) and the sha256 of every composite (`manifest.json`). Fig. 5 needs these images.
- Public benchmarks: only the ids of the items that occur in the shipped outputs are included (`data/benchmark_ids/`: 3DSRBench 500 ids, BLINK 267 ids, CV-Bench 2638 ids), no benchmark content. Download the benchmarks from their public releases into `$VTAX_DATA_ROOT/datasets/` (CV-Bench test split; BLINK val `Spatial_Relation` / `Relative_Depth`; 3DSRBench) for the GPU / API runs; `scripts/build_items.py` builds the item files (set `CVBENCH_HF_ID` to the Hugging Face id of CV-Bench). VSI-Bench and MindCube-Tiny are used only by an appendix table that is not reproduced here (`scripts/e51_*.py`, `VSIBENCH_HF_ID`).

Tab. 1 and Tab. S1 are typed by hand: Tab. 1 from `outputs/bundle/numbers.json` (written by `scripts/build_bundle.py`), Tab. S1 is a text-only evidence map. Fig. 1-6 are regenerated as PDF; their plotted values are compared through `*.data.json`.
