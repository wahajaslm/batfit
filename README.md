# BatFit Extractor (minimal fine-tune)
Train a tiny local model to convert free text -> BatFit JSON.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
make setup
make train            # CPU / Apple Silicon trainer (manifest-driven)
make build
ollama run batfit "Aggressive opener, wants 2.8 lb, mid-low SS, semi-oval handle, full profile."
```

### Train on Colab
If you prefer a hosted GPU, open `notebooks/train_batfit.ipynb` in Google Colab (File → Open Notebook → GitHub) and run the cells in order:

1. Install deps via `requirements-colab.txt`.
2. (Optional) mount Drive if you want to save checkpoints.
3. Adjust env vars (`BATFIT_BASE_MODEL`, sequence length, epochs).
4. Launch `python scripts/train_lora.py` straight from the repo root.

The notebook assumes this repo is cloned in `/content`; if not, add a `git clone` cell at the top and `cd` into the project directory before training.

## LLM vs KERN contract
- Fine-tuned LLM = translator: map cricket talk into normalized **context + interaction** JSON, never physical geometry.
- KERN = deterministic engine: reads that context, applies physics/feasibility, and emits the final bat spec.
- LLM output stops at `{ "context": {...}, "interaction": {...} }` — no edges, spine, grades, or accessories.

### Target schema
```json
{
  "context": {
    "pitch_type": "low",
    "preferred_pickup": "balanced",
    "target_weight_lb": 2.85,
    "profile_pref": "mid_low",
    "playing_style": "control",
    "pickup_goal": "balanced"
  },
  "interaction": {
    "asked": ["pitch_type","preferred_pickup","target_weight_lb"],
    "skipped": [],
    "why_next_question": null
  }
}
```

Data format
\t•\tPrimary training data lives under `data/` (manifest-driven).
\t•\t`data/manifest.yaml` mixes tasks/tenants; defaults provided for `t1_parse` + `t2_repair` context-labeling tasks.
\t•\t`policy/question/global.yaml` defines the global quick-fit + follow-up logic; tenant overrides live alongside their data (e.g., `data/tenants/<tenant>/policy.yaml`).

Tips
\t•\tKeep targets minified (one line) to reduce tokens.
\t•\tIf VRAM errors: lower sequence_len to 768 or switch to Llama-3.2-0.5B.
\t•\tOverride `BATFIT_DATA_DIR`, `BATFIT_MANIFEST`, or `BATFIT_SYSTEM_PROMPT` to point at different corpora.
\t•\tKeep outputs focused on core bat physics (weight windows, sweet spot, spine/edge dimensions, profiles); defer cosmetic add-ons to downstream logic.

## Dataset layout (new)
```
data/
  manifest.yaml
  common/prompts/system.txt
  t1_parse/{train,val}.jsonl
  t2_repair/{train,val}.jsonl
  tenants/<tenant>/*.jsonl
```
JSONL rows accept `{input, output}` or `{messages: [...], output_json}`. The trainer normalizes both and auto-splits a val set when `val` files are missing.

### Adaptive questioning cheat sheet
- Quick Fit collects only `pitch_type`, `preferred_pickup`, and optionally `target_weight_lb`, then immediately runs KERN and shows a draft bat.
- After KERN returns, consult the global `policy/question/global.yaml` (top2 score gap, risk flags, pickup deltas, catalog fit) to ask **at most one** targeted follow-up before re-running. If a tenant override file exists (e.g., `data/tenants/timberwolf/policy.yaml`), merge its rules on top.
- Track actual prompts under `interaction.asked` / `interaction.skipped` so downstream services know which defaults were assumed.
- Use progressive disclosure: always show the current best option with a single inline question; accessories appear only after the bat spec is locked.

**`.gitignore`**

.venv/
pycache/
.ipynb_checkpoints
fine-tune/outputs/

### Acceptance criteria
- `make test` runs end-to-end: train → build → sample run (returns valid JSON).
- Output is a **single JSON object**, no prose; keys match schema.
- Easy to extend: add more rows under `data/` then re-run `make train build`.

### Notes
- Keep everything minimal; no extra frameworks or boilerplate.
- Don’t add CI, Docker, or large samples—just one fixture pair as an example.
- Prefer **small base** (1B) for speed; user can scale up later.

## Tenant data (per-client seasoning)
- Store brand- or academy-specific examples under `data/tenants/<tenant>/` using the same JSONL schema as the global tasks. Each line can encode that tenant’s preferred phrasing, defaults, or policy hints.
- Reference the tenant files in `data/manifest.yaml` with a `weight` so you can control how much influence they have during fine-tuning (e.g., `weight: 0.3` to lightly bias the model toward Timberwolf terminology).
- When preparing a build for a specific client, adjust the manifest mix to include only the relevant tenant folders; for a generic build, remove or zero-out tenant entries to fall back to the global corpus.
- The runtime LLM still outputs the same `{context, interaction}` schema; KERN plus tenant-specific YAML handle the actual feasibility and SKU rules. The tenant data simply helps the model interpret that client’s users more accurately.
- Place tenant-specific question policies next to their data (e.g., `data/tenants/timberwolf/policy.yaml`) so the UX can load the global defaults first and then apply the tenant override in one step.

## Colab-friendly dependencies
- `requirements-colab.txt` pins GPU-capable wheels (including `bitsandbytes`) for Colab runtimes.
- `notebooks/train_batfit.ipynb` walks through installing those deps, setting env vars, and running `scripts/train_lora.py` end-to-end.
