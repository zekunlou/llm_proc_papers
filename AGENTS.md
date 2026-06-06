# AGNENTS

## Description

This project is now at stage 2. Stage 1 is complete.

## Coding Instructions

- Please use conda environment `zotero_mcp` for python arguments.
    - e.g. `conda run -n zotero_mcp python my_script.py`

## Previous Stages

### Stage 1 — PDF location and OCR scripts

Implemented two scripts in `./scripts/`:

- `locate_pdf.py`: search Zotero by free-text query (title, author, abstract via `qmode=everything`), interactively or non-interactively select a result, create the paper directory, export `.bib` via pyzotero, and copy the PDF from local Zotero storage.
- `pdf2md.py`: convert a PDF to markdown using a two-pass LLM pipeline per page — pass 1 extracts single-column markdown with escaped citations (`\[x\]`) and figure placeholders; pass 2 detects figures as bounding boxes (0–999 normalized coords), crops them with Pillow, and saves to an assets directory. Supports `--md`, `--assets`, `--pages`, `--verbose`, `--test-config`.

LLM backend: Qwen3.6-35b-a3b via OpenAI-compatible endpoint (`https://saia.gwdg.de/v1`). Image tokens configured via `LLM_OCR_MIN_TOKENS`/`LLM_OCR_MAX_TOKENS` in `.env` (defaults: 10000/50000).

Supporting files written: `SKILL.md` (LLM-facing reference), `README.md` (human-facing guide).
