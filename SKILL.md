# SKILL: Paper Processing Scripts

This file describes the two CLI scripts in `scripts/` for use by LLM agents. All scripts run in the `zotero_mcp` conda environment.

**Execution prefix:** `conda run -n zotero_mcp python scripts/<script>.py`

---

## `locate_pdf.py` — Zotero search and paper directory setup

### Two modes

**Search/info mode** (no `--output`): searches across title, authors, and abstract. Prints a numbered list of matches, then full detail for the selected item. No files are created.

**Setup mode** (`--output DIR`): creates the paper directory, exports `.bib`, copies the PDF. Prints the copied PDF path to stdout.

### Usage

```
conda run -n zotero_mcp python scripts/locate_pdf.py <query> [options]
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `query` | yes | — | Free-text search: title words, author name, abstract terms, or citekey |
| `--select N` | no | — | Non-interactively pick result N (1-indexed). Required for LLM use when multiple results exist. |
| `--output DIR` | no | — | Per-paper directory to create. Omit for search/info mode only. |
| `--citekey KEY` | no | from Extra field | Override the citekey used for output filenames. |
| `--pdf-name FILENAME` | no | — | Select a specific PDF when the item has multiple attachments. |
| `--limit N` | no | `20` | Max number of search results (default: 20). |
| `--zotero-data-dir DIR` | no | `$ZOTERO_DATA_DIR` | Local Zotero data directory. |
| `--library-id ID` | no | `$ZOTERO_LIBRARY_ID` | Zotero library ID. |
| `--library-type TYPE` | no | `$ZOTERO_LIBRARY_TYPE` | `user` or `group`. |
| `--api-key KEY` | no | `$ZOTERO_API_KEY` | Zotero API key. |

### Stdout / exit codes

- **Single result or --select given**: prints item detail (search mode) or PDF path (setup mode). Exit 0.
- **Multiple results, non-interactive** (LLM): prints numbered list to stdout, message to stderr. Exit 2.
- **Multiple results, interactive** (human): prints list and prompts for selection.
- **Error**: prints error to stderr. Exit 1.

### LLM workflow (two steps)

```bash
# Step 1: search — get numbered list
conda run -n zotero_mcp python scripts/locate_pdf.py "inverse design heterodeformations"
# → exit code 2 if multiple results; list printed to stdout

# Step 2: select and set up
conda run -n zotero_mcp python scripts/locate_pdf.py "inverse design heterodeformations" \
    --select 1 --output ~/OneDrive/papers/ahmedInverseDesign2026
# → prints /path/to/paper.pdf on success
```

### Examples

```bash
# Search only (human interactive)
conda run -n zotero_mcp python scripts/locate_pdf.py "ahmed heterodeformations"

# One-shot if only one result
conda run -n zotero_mcp python scripts/locate_pdf.py "ahmed heterodeformations 2026" \
    --output ~/OneDrive/papers/ahmed2026

# Multiple PDFs — first search to see filenames, then:
conda run -n zotero_mcp python scripts/locate_pdf.py "ahmed heterodeformations" \
    --select 1 --output ~/OneDrive/papers/ahmed2026 --pdf-name main.pdf
```

### What gets created in setup mode

```
{output}/
  {citekey}.pdf      ← copied from Zotero local storage
  {citekey}.bib      ← exported via Zotero API (accurate bibtex)
  assets/            ← empty, ready for pdf2md.py figure output
```

---

## `pdf2md.py` — PDF to markdown with figure extraction

Two-pass LLM OCR per page:
1. **Text pass**: extracts content as single-column markdown; citations escaped as `\[x\]` for Obsidian compatibility; figure placeholders inserted.
2. **Bbox pass**: detects figures as bounding boxes (0–999 normalized coords), crops and saves them to `assets/`.

Final output: `{output}/{citekey}.md` (all pages concatenated with `---` separators).

### Usage

```
conda run -n zotero_mcp python scripts/pdf2md.py <pdf_path> --output DIR [options]
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `pdf_path` | yes* | — | Path to the PDF file. *Not needed with `--test-config`. |
| `--output DIR` | yes* | — | Directory to write the markdown file and assets dir into. *Not needed with `--test-config`. |
| `--md FILENAME` | no | `main_paper.md` | Markdown output filename. `.md` extension added automatically if omitted. |
| `--assets DIRNAME` | no | `assets` | Assets subdirectory name inside `--output`. |
| `--dpi INT` | no | `$LLM_OCR_DPI` (200) | PDF rendering resolution. Higher = better quality, slower. |
| `--min-tokens INT` | no | `$LLM_OCR_MIN_TOKENS` (10000) | Image token floor. Each token = 32×32 px. |
| `--max-tokens INT` | no | `$LLM_OCR_MAX_TOKENS` (50000) | Image token ceiling. Each token = 32×32 px. |
| `--model STR` | no | `$LLM_OCR_MODEL` | Model name. |
| `--endpoint STR` | no | `$LLM_ENDPOINT` | OpenAI-compatible base URL. |
| `--api-key KEY` | no | `$LLM_API_KEY` | API key. |
| `--concurrency N` | no | `$LLM_OCR_CONCURRENCY` (4) | Max pages processed concurrently. Each page also runs its two passes in parallel. |
| `--pages RANGE` | no | all | Page range, e.g. `1-5` or `3`. |
| `--log-file PATH` | no | — | Append all status messages to this file (in addition to stderr). Useful for tracking long runs. |
| `--verbose` | no | — | Print raw bbox response and pixel coordinates per figure. Use to diagnose false positives. |
| `--test-config` | no | — | Test endpoint/key/model connectivity and exit. |

### Stdout / exit codes

- Prints the absolute path to the output markdown file. Exit 0.
- Status messages are written to stderr.
- With `--test-config`: prints `Config OK. Model: ...` and exits 0, or error + exit 1.

### Examples

```bash
# Test OCR configuration
conda run -n zotero_mcp python scripts/pdf2md.py --test-config

# Convert full PDF (output: main_paper.md + assets/)
conda run -n zotero_mcp python scripts/pdf2md.py ~/OneDrive/papers/smith2023efficiency/smith2023efficiency.pdf \
    --output ~/OneDrive/papers/smith2023efficiency

# Custom output filename and assets dir
conda run -n zotero_mcp python scripts/pdf2md.py ~/OneDrive/papers/smith2023efficiency/smith2023efficiency.pdf \
    --output ~/OneDrive/papers/smith2023efficiency \
    --md smith2023efficiency --assets figs

# Convert specific pages only
conda run -n zotero_mcp python scripts/pdf2md.py ~/OneDrive/papers/smith2023efficiency/smith2023efficiency.pdf \
    --output ~/OneDrive/papers/smith2023efficiency \
    --pages 1-5
```

### Typical workflow (LLM-assisted)

```
1. locate_pdf.py <query>                                → verify item (search mode)
2. locate_pdf.py <query> --select N --output <dir>     → setup dir, copy PDF
3. pdf2md.py <pdf_path> --output <dir> [--md <name>] [--assets <dir>] → generate markdown
4. Verify figures (see below)
```

### Post-OCR figure verification (always run after pdf2md.py)

After OCR completes, verify that every figure placeholder in the markdown has a matching file in the assets dir and vice versa. Substitute `{md}` and `{assets_dir}` with the actual paths used (defaults: `main_paper.md` and `assets/`):

```bash
# Figure references in the markdown (extracts sub_labels from ![sub_label](assets/...) links)
grep -oP '!\[\K[^\]]+(?=\]\({assets_dir}/)' {output}/{md}

# Files actually present in the assets dir
ls {output}/{assets_dir}/
```

**What to check:**
- Every name from the `grep` should have a corresponding `.png` in the assets dir. Missing files mean the bbox pass failed to detect or crop that figure.
- Every file in the assets dir should appear in the markdown. Orphan files mean the bbox pass detected something the text pass did not label (likely a false positive).

If mismatches are found, re-run the affected pages with `--verbose` to inspect the raw bbox output and diagnose the issue.

---

## Environment variables (`.env` in project root)

| Variable | Used by | Description |
|---|---|---|
| `ZOTERO_LIBRARY_ID` | locate_pdf | Zotero library ID |
| `ZOTERO_LIBRARY_TYPE` | locate_pdf | `user` or `group` |
| `ZOTERO_API_KEY` | locate_pdf | Zotero API key |
| `ZOTERO_DATA_DIR` | locate_pdf | Local Zotero data directory |
| `LLM_ENDPOINT` | pdf2md | OpenAI-compatible API base URL |
| `LLM_API_KEY` | pdf2md | API key |
| `LLM_OCR_MODEL` | pdf2md | Model name |
| `LLM_OCR_DPI` | pdf2md | PDF render resolution (default 200) |
| `LLM_OCR_MIN_TOKENS` | pdf2md | Image token floor (default 10000) |
| `LLM_OCR_MAX_TOKENS` | pdf2md | Image token ceiling (default 50000) |
| `LLM_OCR_CONCURRENCY` | pdf2md | Max concurrent pages (default 4) |
