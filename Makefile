PY := .venv/bin/python

.PHONY: all venv data extract train bench report test clean

all: data train bench report

venv: .venv/.installed
.venv/.installed: requirements.txt
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	touch $@

# Synthetic TTS set (macOS `say`) + modest EdAcc subset (one parquet row group per shard)
data: venv
	$(PY) scripts/prepare_data.py

# VADs, Smart Turn, ASR and prosody features, cached per item
extract: venv
	$(PY) scripts/extract.py

train: extract
	$(PY) scripts/train.py

bench: train
	$(PY) scripts/bench.py

report: venv
	$(PY) scripts/report.py

test: venv
	$(PY) -m pytest

clean:
	rm -rf data/cache data/models
