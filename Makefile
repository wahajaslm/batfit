PY=python3

.PHONY: setup train build run test all

setup:
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) scripts/train_lora.py

build:
	ollama create batfit -f ollama/Modelfile

run:
	bash tools/run_ollama_test.sh

test: train build run

all: test
