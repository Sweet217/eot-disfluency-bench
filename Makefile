PY := .venv/bin/python

.PHONY: all venv data extract train bench report test clean

all: data train bench report

venv: .venv/.installed
.venv/.installed: requirements.txt
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	touch $@

# Synthetic TTS set (macOS `say`) + EdAcc (all 16 shards, <= 60 turns per conversation)
data: venv
	$(PY) scripts/prepare_data.py

# Real-time pass (streaming ASR schedule, measured compute; test sets single-stream),
# then the instant-feature ablation pass. Cached in data/cache_v2/.
extract: venv
	$(PY) scripts/extract.py --pass rt
	$(PY) scripts/extract.py --pass instant

train: extract
	$(PY) scripts/train.py

bench: train
	$(PY) scripts/bench.py

report: venv
	$(PY) scripts/report.py

test: venv
	$(PY) -m pytest

clean:
	rm -rf data/cache_v2 data/models
