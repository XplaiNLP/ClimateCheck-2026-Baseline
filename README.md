# ClimateCheck-2026-Baseline
Implementation for the ClimateCheck 2026 Baseline

Repo status: WIP

More Information on the Shared Task:
- https://nfdi4ds.github.io/nslp2026/docs/climatecheck_shared_task.html
- https://www.codabench.org/competitions/12213

# Installation
Run `python -m pip install -r requirements.txt` in a new python env

# Run Inference

`full_inference.py` contains a full script version which produces a submission file for both tasks.

# Fine-tune Qwen3-8B

1. run `wget https://huggingface.co/datasets/rabuahmad/climatecheck/resolve/main/data/train-00000-of-00001.parquet` in the current dir
2. Run `ft_prep.py` to produce a `cc_messages.jsonl` file
3. Run `ft_qwen3.py` for fine tuning