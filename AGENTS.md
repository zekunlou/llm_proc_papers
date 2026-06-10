# AGNENTS

## Description

This project is now at stage 4. Stages 1–3 are complete.

## Coding Instructions

- Please use conda environment `zotero_mcp` for python arguments.
    - e.g. `conda run -n zotero_mcp python my_script.py`

## Previous Stages

### Stage 1 — PDF location and OCR scripts

Implemented two scripts in `./scripts/`:

- `locate_pdf.py`: search Zotero by citekey or free-text query using a local SQLite backend (default, fast, no API needed) or the pyzotero API. Citekey search (`--citekey KEY`) takes priority over free-text; if citekey is not found, falls back to query with a Better BibTeX "Pin BibTeX key" hint. Interactively or non-interactively select a result, create the paper directory, export `.bib` (requires API credentials), and copy the PDF from local Zotero storage. Output filename key is the detected citekey by default; override with `--output-key`.
- `pdf2md.py`: convert a PDF to markdown using a two-pass LLM pipeline per page — pass 1 extracts single-column markdown with escaped citations (`\[x\]`) and figure placeholders; pass 2 detects figures as bounding boxes (0–999 normalized coords), crops them with Pillow, and saves to an assets directory. Supports `--md`, `--assets`, `--pages`, `--verbose`, `--test-config`.

LLM backend: Qwen3.6-35b-a3b via OpenAI-compatible endpoint (`https://saia.gwdg.de/v1`). Image tokens configured via `LLM_OCR_MIN_TOKENS`/`LLM_OCR_MAX_TOKENS` in `.env` (defaults: 10000/50000).

Supporting files written: `SKILL.md` (LLM-facing reference), `README.md` (human-facing guide).

### Stage 2 — Async acceleration and observability for `pdf2md.py`

Rewrote `pdf2md.py` to process pages concurrently:

- Switched to `AsyncOpenAI`; `call_llm` is now async.
- **Within-page parallelism**: text and bbox passes run simultaneously via `asyncio.gather` — halves per-page latency.
- **Across-page parallelism**: all pages dispatched concurrently under an `asyncio.Semaphore`; results sorted by index to preserve page order.
- **Rate-limit handling**: `call_llm` catches `RateLimitError`, logs remaining quota from response headers (`x-ratelimit-remaining-minute/hour/day`), and retries after `ratelimit-reset + 5` seconds (falls back to 60 s if header absent).
- Small bbox filter changed from hard skip to log-and-crop (warning only in `--verbose`).
- **Shared `FIGURE_RULES`**: figure definition extracted into a single constant injected into both the text and bbox prompts, so both passes use identical criteria for what counts as a figure.
- New `.env` vars: `LLM_OCR_CONCURRENCY` (default 4), `LLM_OCR_DPI` (default 200).
- New CLI args: `--concurrency N`, `--log-file PATH` (appends status messages to file), `--dpi` now reads from `$LLM_OCR_DPI`.
- Added `.env.template` for safe GitHub publishing.

### Stage 3 — Model-aware thinking-mode control

Different serving backends expose hybrid-thinking toggles via incompatible `extra_body` conventions (Qwen3/vLLM uses `chat_template_kwargs.enable_thinking`, MiMo uses `thinking.type`, plain OpenAI has no boolean toggle). Added `thinking_extra_body(model, enable_thinking)`, which inspects `model.lower()` for `"qwen"` / `"mimo"` substrings and returns the matching `extra_body` dict (empty dict for unrecognized models). `call_llm` now builds `extra_body` through this dispatcher instead of hardcoding one convention. Wired so the bbox-detection pass runs with thinking enabled and the text-extraction pass runs with thinking disabled.

### Stage 4 — Bbox-first sequential passes + prompt quality rules

Changed `process_page` from running both passes concurrently (`asyncio.gather`) to running them sequentially: bbox detection first, then text extraction. The detected `sub_label`/`bbox_2d` list is formatted and injected into the text prompt (`{detected_figures}` placeholder) so the text pass reuses the exact labels the bbox pass already assigned — eliminating mismatches between markdown figure references and saved asset filenames.

New shared prompt constants (alongside `FIGURE_RULES`):
- **`SUBFIGURE_RULES`**: clarifies that subfigures sharing one caption are a single figure regardless of layout, injected into both `PROMPT_BBOX` and `PROMPT_TEXT_TEMPLATE`.
- **`LATEX_MARKDOWN_RULES`**: KaTeX/Markdown compatibility constraints (`\boldsymbol` not `\bm`, `align` instead of bare `aligned` inside `$$...$$`, no stacked notation macros).
- **`TRANSCRIPTION_RULES`**: instructs the model to transcribe the rendered page literally rather than reconstruct LaTeX source artifacts (`\ref`, `\label`, `\cite`, `\footnote`), and to use Markdown footnote syntax (`[^n]` / `[^n]: ...`) for footnote markers.

Across-page concurrency (the `asyncio.Semaphore`) is unaffected — only the within-page pass ordering changed.
