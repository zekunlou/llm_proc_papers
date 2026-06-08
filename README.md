# LLM Paper Processing Scripts

Two scripts for integrating Zotero papers into the Obsidian-based research workflow:
- `scripts/locate_pdf.py` — find a paper in Zotero, set up its directory
- `scripts/pdf2md.py` — convert a PDF to markdown with extracted figures

See `SKILL.md` for the full argparse reference used by LLM agents.

---

## Setup

### 1. Conda environment

```bash
conda activate zotero_mcp
pip install pyzotero python-dotenv openai pdf2image Pillow
# pdf2image also requires poppler:
brew install poppler   # macOS
```

### 2. `.env` file

Copy and fill in the project root `.env`:

```env
ZOTERO_LIBRARY_ID=<your library ID from zotero.org/settings/keys>
ZOTERO_LIBRARY_TYPE=user
ZOTERO_API_KEY=<your API key>
ZOTERO_DATA_DIR=~/Zotero

LLM_ENDPOINT=https://your-api-endpoint/v1
LLM_API_KEY=<your key>
LLM_OCR_MODEL=qwen3.6-35b-a3b
LLM_OCR_DPI=200
LLM_OCR_MIN_TOKENS=10000
LLM_OCR_MAX_TOKENS=50000
LLM_OCR_CONCURRENCY=4
```

### 3. Verify OCR config

```bash
conda run -n zotero_mcp python scripts/pdf2md.py --test-config
```

---

## Workflow

### Step 1 — Add paper to Zotero

Add the paper in Zotero and ensure Better BibTeX assigns a citekey (visible in the Extra field). Download the PDF attachment within Zotero.

### Step 2 — Verify item

```bash
conda run -n zotero_mcp python scripts/locate_pdf.py smith2023efficiency
```

Prints title, authors, year, abstract, and available PDF filenames. Confirm this is the right paper before proceeding.

### Step 3 — Set up paper directory

```bash
conda run -n zotero_mcp python scripts/locate_pdf.py smith2023efficiency \
    --output ~/OneDrive/papers/smith2023efficiency
```

Creates:
```
~/OneDrive/papers/smith2023efficiency/
  smith2023efficiency.pdf    ← copied from Zotero
  smith2023efficiency.bib    ← exported via Zotero API
  assets/                    ← empty, ready for figures
```

If the item has multiple PDFs, you'll see an error listing their names. Re-run with `--pdf-name <filename>` to select one.

### Step 4 — Convert PDF to markdown

```bash
conda run -n zotero_mcp python scripts/pdf2md.py \
    ~/OneDrive/papers/smith2023efficiency/smith2023efficiency.pdf \
    --output ~/OneDrive/papers/smith2023efficiency
```

Produces:
```
~/OneDrive/papers/smith2023efficiency/
  main_paper.md    ← full paper as markdown (default name)
  assets/          ← cropped figures (default dir name)
    figure_1.png
    figure_2.png
    ...
```

Use `--md` and `--assets` to override the defaults:
```bash
... --md smith2023efficiency --assets figs
# → smith2023efficiency.md + figs/
```

To process only specific pages:
```bash
... --pages 1-5
```

---

## Notes

- Citations in the markdown are escaped as `\[1\]` to avoid Obsidian treating them as links.
- Each page is separated by a `---` horizontal rule.
- Figure placeholders in the markdown (`![figure_1](assets/figure_1.png)`) match the cropped images by `sub_label`.
- Status messages go to stderr; stdout contains only the output path (for scripting).
- OCR is async: within each page, the bbox pass runs first and its detected `sub_label`s are fed into the text pass prompt (so figure references stay consistent); across pages, up to `--concurrency` pages run at the same time (default 4, configurable via `$LLM_OCR_CONCURRENCY`).
- Use `--log-file ocr.log` to capture status messages to a file for long runs.
