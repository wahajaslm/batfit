#!/usr/bin/env python3
"""CPU/MPS-friendly LoRA fine-tune pipeline for BatFit data (directory/manifest aware).

Workflow:
1. Resolve all runtime knobs from env vars (model ids, batch sizes, paths, etc.).
2. Parse the manifest to discover which JSONL files feed training/validation and normalize them.
3. Wrap each example in a chat-style prompt so tokenization mirrors inference usage.
4. Tokenize prompts + targets, producing padded LM samples with causal labels.
5. Load the base model, inject LoRA adapters, and configure Trainer hyperparameters.
6. Train, then persist only the LoRA adapter weights plus the tokenizer to OUTPUT_DIR.
"""

import contextlib
import datetime
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib.machinery
import sys
import types

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Run `python3 -m pip install -r requirements.txt` "
        "or `make setup` before executing this script."
    ) from exc

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
try:
    import mlflow
except Exception:
    mlflow = None


def _ensure_bitsandbytes_stub() -> None:
    """PEFT imports bitsandbytes even when 8-bit features are unused."""
    try:
        import bitsandbytes  # noqa: F401
        return
    except Exception:
        stub = types.ModuleType("bitsandbytes")
        stub.__version__ = "0.0.0-stub"
        stub.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", loader=None)
        sys.modules["bitsandbytes"] = stub


_ensure_bitsandbytes_stub()

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@contextlib.contextmanager
def mlflow_run(params: Dict[str, str]):
    """Context manager that logs params/metrics if MLflow is enabled."""
    if not MLFLOW_ENABLED or mlflow is None:
        yield None
        return
    try:
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        if MLFLOW_EXPERIMENT:
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run_name = MLFLOW_RUN_NAME or f"batfit-{Path(BASE_MODEL).name}-{datetime.datetime.utcnow():%Y%m%d-%H%M%S}"
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.log_params(params)
            yield active_run
    except Exception as exc:  # pragma: no cover - logging fallback
        warnings.warn(f"Disabling MLflow logging due to error: {exc}")
        yield None


def _log_mlflow_metrics(metrics: Dict[str, float]) -> None:
    if not metrics or mlflow is None:
        return
    try:
        mlflow.log_metrics(metrics)
    except Exception as exc:  # pragma: no cover - logging fallback
        warnings.warn(f"Failed to log MLflow metrics: {exc}")


def _log_mlflow_artifact(path: Path, artifact_path: Optional[str] = None) -> None:
    if mlflow is None:
        return
    try:
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        elif path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:  # pragma: no cover - logging fallback
        warnings.warn(f"Failed to log MLflow artifact {path}: {exc}")

# ------------------ Config (env) ------------------
# Every knob can be overridden via BATFIT_* env vars so CI, notebooks, and local runs match.
BASE_MODEL = os.getenv("BATFIT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_DIR = Path(os.getenv("BATFIT_DATA_DIR", "data"))
MANIFEST = Path(os.getenv("BATFIT_MANIFEST", DATA_DIR / "manifest.yaml"))
SPLIT_SEED = int(os.getenv("BATFIT_SPLIT_SEED", "42"))

# Output path doubles as the PEFT adapter + tokenizer save location.
OUTPUT_DIR = Path(os.getenv("BATFIT_OUTPUT_DIR", "fine-tune/outputs/batfit"))
MAX_SEQ_LEN = int(os.getenv("BATFIT_MAX_LEN", "768"))
MICRO_BATCH = int(os.getenv("BATFIT_BATCH", "1"))
GRAD_ACCUM = int(os.getenv("BATFIT_GRAD_ACCUM", "2"))
EPOCHS = float(os.getenv("BATFIT_EPOCHS", "2"))
LEARNING_RATE = float(os.getenv("BATFIT_LR", "1.5e-4"))
LORA_R = int(os.getenv("BATFIT_LORA_R", "16"))
LORA_ALPHA = int(os.getenv("BATFIT_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("BATFIT_LORA_DROPOUT", "0.05"))
TARGET_MODULES = os.getenv(
    "BATFIT_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj"
).split(",")

SYSTEM_PROMPT = os.getenv("BATFIT_SYSTEM_PROMPT", "").strip()
MLFLOW_ENABLED = os.getenv("BATFIT_ENABLE_MLFLOW", "1").lower() not in {"0", "false", "no"}
MLFLOW_TRACKING_URI = os.getenv("BATFIT_MLFLOW_TRACKING_URI", "").strip()
MLFLOW_EXPERIMENT = os.getenv("BATFIT_MLFLOW_EXPERIMENT", "batfit-ft").strip()
MLFLOW_RUN_NAME = os.getenv("BATFIT_MLFLOW_RUN_NAME", "").strip()


def resolve_system_prompt() -> str:
    """Load the system prompt using env/file fallback logic so notebooks can reuse it."""
    sys_prompt_file = DATA_DIR / "common" / "prompts" / "system.txt"
    return SYSTEM_PROMPT or _read_text_if_exists(sys_prompt_file)


# ------------------ Data loading ------------------
def _read_text_if_exists(path: Path) -> str:
    """Return file text if it exists, otherwise an empty string to keep callers simple."""
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _normalize_example(example: dict) -> dict:
    """Normalize different JSONL formats into {input_text,target_text}.

    Args:
        example: Raw row from various BatFit data exports.
    Returns:
        Dict with unified keys so downstream prompt construction is predictable.
    """
    if "messages" in example:
        parts = []
        for msg in example["messages"]:
            role = msg.get("role", "user")
            # Preserve chat role markup so downstream prompt builder can skip re-wrapping.
            parts.append(f"<|{role}|>\n{msg.get('content','')}\n")
        input_text = "".join(parts) + "<|assistant|>\n"
        target_text = json.dumps(example.get("output_json", {}), ensure_ascii=False)
        return {"input_text": input_text, "target_text": target_text}

    user_text = example.get("input") or example.get("text") or ""
    target_text = example.get("output") or ""
    return {"input_text": user_text, "target_text": target_text}


def _weighted_concat(datasets_with_weights: List[Tuple[Dataset, float]]) -> Dataset:
    """Bootstrap/duplicate datasets to honor manifest weights.

    Args:
        datasets_with_weights: List of (dataset, weight) pairs from the manifest.
    Returns:
        A single dataset representing the weighted mix.
    Raises:
        ValueError: If the manifest lists no datasets.
    """
    expanded: List[Dataset] = []
    for dataset, weight in datasets_with_weights:
        if weight == 1.0:
            expanded.append(dataset)
            continue
        length = len(dataset)
        target_size = max(1, int(math.ceil(length * weight)))
        indices = [random.randrange(length) for _ in range(target_size)]
        expanded.append(dataset.select(indices))
    if not expanded:
        raise ValueError("No datasets found in manifest mix section")
    return concatenate_datasets(expanded) if len(expanded) > 1 else expanded[0]


def load_from_manifest() -> Tuple[Dataset, Optional[Dataset]]:
    """Read manifest-defined dataset splits and fall back to random split if needed.

    Returns:
        Tuple of (train_dataset, val_dataset or None).
    """
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    # Manifest guards which JSONL files enter training/validation plus optional weights.
    manifest = yaml.safe_load(MANIFEST.read_text())
    mix_files = manifest.get("mix", [])
    val_files = manifest.get("val", [])

    train_parts: List[Tuple[Dataset, float]] = []
    for entry in mix_files:
        dataset_path = DATA_DIR / entry["file"]
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {dataset_path}")
        dataset = load_dataset("json", data_files=str(dataset_path))["train"]
        # Normalize each JSON row into the unified user/assistant format.
        dataset = dataset.map(_normalize_example)
        weight = float(entry.get("weight", 1.0))
        train_parts.append((dataset, weight))
    train_dataset = _weighted_concat(train_parts)

    val_dataset = None
    if val_files:
        val_parts = []
        for entry in val_files:
            dataset_path = DATA_DIR / entry["file"]
            if not dataset_path.exists():
                continue
            dataset = load_dataset("json", data_files=str(dataset_path))["train"]
            val_parts.append(dataset.map(_normalize_example))
        if val_parts:
            val_dataset = (
                concatenate_datasets(val_parts) if len(val_parts) > 1 else val_parts[0]
            )

    if val_dataset is None:
        # If no explicit validation set is listed, carve one out deterministically.
        if len(train_dataset) <= 1:
            return train_dataset, None
        split: DatasetDict = train_dataset.train_test_split(
            test_size=max(1, int(0.1 * len(train_dataset))),
            seed=SPLIT_SEED,
            shuffle=True,
        )
        return split["train"], split["test"]

    return train_dataset, val_dataset


# ------------------ Prompt building ------------------
def build_prompt(input_text: str, tokenizer: AutoTokenizer, system_prompt_text: str) -> Dict:
    """Wrap raw input in a chat template so the model sees system/user/assistant roles.

    Args:
        input_text: Either plain user text or a preformatted chat string.
        tokenizer: Loaded tokenizer to format the chat template.
        system_prompt_text: Optional override for the system instruction.
    Returns:
        Tokenizer output dict ready to be extended with labels.
    """
    if "<|assistant|>" in input_text:
        prompt = input_text
    else:
        # Fall back to the default schema description if the user does not pass BATFIT_SYSTEM_PROMPT.
        system_text = system_prompt_text or (
            "Output ONLY one JSON object with keys context and interaction. "
            "context should contain any inferred fit fields "
            "(pitch_type, preferred_pickup, target_weight_lb, dominant_shot_area, "
            "durability_bias, profile_pref, player_level, pickup_goal, etc.), "
            "and interaction must include asked, skipped, why_next_question. "
            "Never mention geometry or accessories."
        )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": ""},
        ]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            prompt = "".join(
                f"<|{m['role']}|>\n{m['content']}\n" for m in messages
            )

    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    return tokenized


def prepare_tokenizer(base_model: str = BASE_MODEL) -> AutoTokenizer:
    """Load and pad the tokenizer once so scripts/notebooks share the exact settings."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_splits(
    train_raw: Dataset,
    val_raw: Optional[Dataset],
    tokenizer: AutoTokenizer,
    system_prompt_text: str,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Apply prompt building + LM formatting to raw datasets."""

    def _map_build(example: dict) -> dict:
        prompt_tokens = build_prompt(example["input_text"], tokenizer, system_prompt_text)
        prompt_text = tokenizer.decode(
            prompt_tokens["input_ids"], skip_special_tokens=False
        )
        full = tokenizer(
            prompt_text + example["target_text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        full["labels"] = full["input_ids"].copy()
        return full

    train_tokenized = train_raw.map(_map_build, remove_columns=train_raw.column_names)
    val_tokenized = (
        val_raw.map(_map_build, remove_columns=val_raw.column_names)
        if val_raw is not None
        else None
    )
    return train_tokenized, val_tokenized


def resolve_device() -> Tuple[bool, bool, Optional[str], torch.dtype]:
    """Determine device map + dtype so CPU/MPS flows can reuse the logic."""
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and not use_cuda
    device_map = "auto" if use_cuda else None
    dtype = torch.bfloat16 if use_cuda else torch.float32
    return use_cuda, use_mps, device_map, dtype


def prepare_model(
    base_model: str,
    dtype: torch.dtype,
    device_map: Optional[str],
    use_mps: bool,
) -> torch.nn.Module:
    """Load the base model and inject LoRA adapters."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if use_mps:
        model.to("mps")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_trainer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
) -> Trainer:
    """Create a Trainer configured for the current dataset split."""
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_strategy = "steps" if val_dataset is not None else "no"
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        evaluation_strategy=eval_strategy,
        eval_steps=50 if val_dataset is not None else None,
        report_to="none",
        bf16=False,
        fp16=False,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )


# ------------------ Main ------------------
def main(run_training: bool = True) -> Trainer:
    """Entrypoint that wires data loading, tokenization, LoRA config, and training."""
    random.seed(SPLIT_SEED)
    system_prompt_text = resolve_system_prompt()
    train_raw, val_raw = load_from_manifest()
    tokenizer = prepare_tokenizer(BASE_MODEL)
    train_tokenized, val_tokenized = tokenize_splits(
        train_raw, val_raw, tokenizer, system_prompt_text
    )
    _, use_mps, device_map, dtype = resolve_device()
    model = prepare_model(BASE_MODEL, dtype, device_map, use_mps)
    trainer = build_trainer(model, tokenizer, train_tokenized, val_tokenized)
    run_params = {
        "base_model": BASE_MODEL,
        "max_seq_len": MAX_SEQ_LEN,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "train_rows": len(train_raw),
        "val_rows": len(val_raw) if val_raw is not None else 0,
    }
    params_as_str = {k: str(v) for k, v in run_params.items()}
    with mlflow_run(params_as_str) as active_mlflow:
        if run_training:
            train_output = trainer.train()
            trainer.save_model()
            tokenizer.save_pretrained(str(OUTPUT_DIR))
            print(f"Saved LoRA adapter + tokenizer to {OUTPUT_DIR}")
            metrics_to_log: Dict[str, float] = {}
            if hasattr(train_output, "metrics"):
                for key, value in train_output.metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_to_log[key] = float(value)
            metrics_to_log["train_samples"] = len(train_tokenized)
            if active_mlflow:
                _log_mlflow_metrics(metrics_to_log)
                if val_tokenized is not None:
                    eval_metrics = trainer.evaluate()
                    eval_numeric = {
                        key: float(val)
                        for key, val in eval_metrics.items()
                        if isinstance(val, (int, float))
                    }
                    _log_mlflow_metrics(eval_numeric)
                _log_mlflow_artifact(OUTPUT_DIR, artifact_path="adapters")
                if MANIFEST.exists():
                    _log_mlflow_artifact(MANIFEST, artifact_path="config")
    return trainer


if __name__ == "__main__":
    main()
