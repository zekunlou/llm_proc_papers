#!/usr/bin/env python3
"""Convert an academic PDF to markdown with extracted figures.

Two-pass LLM processing per page (both passes run concurrently):
  Pass 1: extract text as single-column markdown (citations escaped as \[x\])
  Pass 2: detect figures as bounding boxes (0-999 normalized coords), crop and save

Pages are also processed concurrently up to --concurrency slots.

Output: {output}/{md}.md + {output}/{assets}/{sub_label}.png files
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from PIL import Image

FIGURE_RULES = """\
A figure IS: a plot, graph, chart, schematic diagram, simulation result, microscopy image, or scientific illustration that conveys data or a scientific concept. It must occupy meaningful 2D area — roughly square or wider-than-tall with substantial height.

A figure is NOT: plain text, section headers, author affiliations, abstract text, inline or display equations, formula blocks, tables, logos, journal stamps, arXiv identifiers, page numbers, or any decorative/layout element. Do NOT mark thin horizontal bands (these are usually equations), do NOT mark the entire page, do NOT mark any region that is mostly text or symbols.\
"""

PROMPT_TEXT_TEMPLATE = """\
Convert this academic paper page to markdown. Follow these rules exactly:
- Output single column only. If the page has two columns, process the left column first, then the right column, maintaining reading order.
- Escape all citation brackets with a backslash: write \\[1\\] instead of [1], \\[Smith, 2023\\] instead of [Smith, 2023].
- For each figure, insert a placeholder on its own line using the format ![sub_label]({assets}/sub_label.png). Place the figure caption immediately after the placeholder. If it is a numbered figure, use sub_label to indicate the figure number only (e.g. figure_1, figure_2). If the figure is not numbered, use a descriptive sub_label (e.g. band_structure, phase_diagram). Use the same definition of "figure" as below:

{figure_rules}

- Tables should be rendered as markdown tables.
- Math should be rendered as LaTeX inline ($...$) or block ($$...$$).
- Output markdown only. No preamble, no explanation, no commentary.\
"""

PROMPT_BBOX = """\
This is a page from an academic paper. Mark only data-bearing scientific figures using bounding boxes.

{figure_rules}

Output a raw JSON array. Do not wrap it in markdown code fences (no ```). No explanation before or after:
[
  {{"bbox_2d": [x1, y1, x2, y2], "label": "figure", "sub_label": "figure_1"}},
  ...
]

Rules:
- Coordinates are on a 0–999 scale relative to the image dimensions.
- Each box must be tight around the figure content only (not surrounding whitespace or captions).
- A valid figure box must have both width and height greater than 10% of the page dimension (i.e. at least 100 in the 0–999 scale on each axis). Don't be too tight, it is fine to include some surrounding whitespace, but it is important to include the entire figure content and the sublabels like (a) (b) around the figure.
- label must be "figure". If it is a numbered figure, please only use sub_label to indicate the figure number (e.g. figure_1, figure_2) without any additional text. If the figure is not numbered, use a descriptive sub_label (e.g. band_structure, phase_diagram).
- sub_label: lowercase, underscores, descriptive (e.g. figure_1, band_structure, phase_diagram_a). Use the same sub_label you would assign in a text extraction pass for this page.
- If there are subfigures but are neighboring, just group them into one box with a single sub_label (e.g. figure_2) rather than trying to label them separately (e.g. figure_2a, figure_2b).
- If no scientific figures are present on this page, output exactly: []\
"""

# Module-level state set once in main() before any async work
_log_file = None
_verbose = False


def log(msg: str, verbose_only: bool = False) -> None:
    if verbose_only and not _verbose:
        return
    print(msg, file=sys.stderr)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()


def load_env():
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)


def parse_page_range(pages_str, total_pages):
    """Parse '1-5' or '3' into a 0-indexed list of page indices."""
    if not pages_str:
        return list(range(total_pages))
    parts = pages_str.strip().split("-")
    if len(parts) == 1:
        p = int(parts[0]) - 1
        return [p]
    start, end = int(parts[0]) - 1, int(parts[1]) - 1
    return list(range(start, end + 1))


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def call_llm(client: AsyncOpenAI, model, b64_image, prompt, min_pixels, max_pixels):
    """Send one image+prompt to the LLM and return the response text. Retries on 429."""
    from openai import RateLimitError
    retry_wait = 60
    while True:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "min_pixels": min_pixels,
                                "max_pixels": max_pixels,
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=20000,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            headers = getattr(getattr(e, "response", None), "headers", {})
            limit_min  = headers.get("x-ratelimit-limit-minute", "?")
            remain_min = headers.get("x-ratelimit-remaining-minute", "?")
            limit_hr   = headers.get("x-ratelimit-limit-hour", "?")
            remain_hr  = headers.get("x-ratelimit-remaining-hour", "?")
            limit_day  = headers.get("x-ratelimit-limit-day", "?")
            remain_day = headers.get("x-ratelimit-remaining-day", "?")
            reset_sec  = headers.get("ratelimit-reset", None)
            wait = (int(reset_sec) + 5) if reset_sec is not None else retry_wait
            log(
                f"Rate limit hit — minute: {remain_min}/{limit_min}, "
                f"hour: {remain_hr}/{limit_hr}, day: {remain_day}/{limit_day}, "
                f"reset in {reset_sec if reset_sec is not None else '?'}s — waiting {wait}s before retry..."
            )
            await asyncio.sleep(wait)


def parse_bbox_response(text):
    """Extract the JSON array of bboxes from LLM response."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def denormalize_bbox(coords, img_width, img_height):
    x1, y1, x2, y2 = coords
    return (
        int(x1 / 999.0 * img_width),
        int(y1 / 999.0 * img_height),
        int(x2 / 999.0 * img_width),
        int(y2 / 999.0 * img_height),
    )


async def process_page(
    client: AsyncOpenAI,
    model,
    img: Image.Image,
    min_pixels,
    max_pixels,
    assets_dir: Path,
    page_num: int,
    assets_name: str = "assets",
):
    """Run two-pass OCR on one page image concurrently. Returns markdown string."""
    b64 = image_to_base64(img)
    img_w, img_h = img.size
    prompt_text = PROMPT_TEXT_TEMPLATE.format(assets=assets_name, figure_rules=FIGURE_RULES)
    prompt_bbox = PROMPT_BBOX.format(figure_rules=FIGURE_RULES)

    log(f"  Page {page_num}: passes 1+2 starting concurrently...")
    markdown, bbox_text = await asyncio.gather(
        call_llm(client, model, b64, prompt_text, min_pixels, max_pixels),
        call_llm(client, model, b64, prompt_bbox, min_pixels, max_pixels),
    )
    log(f"  Page {page_num}: passes done.")

    log(f"  [verbose] Page {page_num} raw bbox response:\n{bbox_text}", verbose_only=True)

    bboxes = parse_bbox_response(bbox_text)

    if bboxes:
        log(f"  Page {page_num}: {len(bboxes)} figure(s), cropping...")
    for bbox in bboxes:
        coords = bbox.get("bbox_2d", [])
        sub_label = bbox.get("sub_label", f"figure_p{page_num}")
        if len(coords) != 4:
            continue
        if (coords[2] - coords[0]) < 100 or (coords[3] - coords[1]) < 100:
            log(f"  [verbose] Page {page_num} {sub_label}: small bbox {coords}, cropping anyway", verbose_only=True)
        x1, y1, x2, y2 = denormalize_bbox(coords, img_w, img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        log(
            f"  [verbose] Page {page_num} {sub_label}: norm={coords} → px=({x1},{y1},{x2},{y2}) "
            f"size={x2-x1}×{y2-y1} (image {img_w}×{img_h})",
            verbose_only=True,
        )
        cropped = img.crop((x1, y1, x2, y2))
        out_path = assets_dir / f"{sub_label}.png"
        cropped.save(out_path, format="PNG")

    return markdown


async def async_test_config(client: AsyncOpenAI, model):
    """Send a minimal text-only request to verify the endpoint/key/model."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=2000,
        )
        msg = response.choices[0].message
        reply = (msg.content or "").strip()
        print(f"Config OK. Model: {model}")
        print(f"Response: {reply}")
        sys.exit(0)
    except Exception as e:
        print(f"Config FAILED: {e}", file=sys.stderr)
        sys.exit(1)


async def async_main(args, client: AsyncOpenAI, all_pages, page_indices):
    output_dir = Path(args.output).expanduser()
    assets_dir = output_dir / args.assets
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    min_pixels = args.min_tokens * 32 * 32
    max_pixels = args.max_tokens * 32 * 32
    total = len(all_pages)

    sem = asyncio.Semaphore(args.concurrency)

    async def process_guarded(idx):
        async with sem:
            page_num = idx + 1
            log(f"Page {page_num}/{total} acquired slot.")
            md = await process_page(
                client, args.model, all_pages[idx], min_pixels, max_pixels,
                assets_dir, page_num, assets_name=args.assets,
            )
            return (idx, md)

    log(f"Processing {len(page_indices)} of {total} pages (concurrency={args.concurrency}).")
    results = await asyncio.gather(*[process_guarded(i) for i in page_indices])
    page_markdowns = [md for _, md in sorted(results)]

    combined = "\n\n---\n\n".join(page_markdowns)
    md_name = args.md if args.md.endswith(".md") else args.md + ".md"
    out_md = output_dir / md_name
    out_md.write_text(combined, encoding="utf-8")
    log(f"Written: {out_md}")
    print(str(out_md))


def main():
    global _log_file, _verbose

    load_env()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to the PDF file. Not required with --test-config.",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        help="Directory to write the markdown file and assets dir into.",
    )
    parser.add_argument(
        "--md",
        metavar="FILENAME",
        default="main_paper.md",
        help="Markdown output filename (default: main_paper.md).",
    )
    parser.add_argument(
        "--assets",
        metavar="DIRNAME",
        default="assets",
        help="Assets subdirectory name inside --output (default: assets).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=int(os.environ.get("LLM_OCR_DPI", 200)),
        help="Resolution for PDF page rendering (default: $LLM_OCR_DPI or 200).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=int(os.environ.get("LLM_OCR_MIN_TOKENS", 10000)),
        dest="min_tokens",
        help="Minimum image tokens (default: $LLM_OCR_MIN_TOKENS or 10000). "
             "Each token covers a 32×32 px patch.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("LLM_OCR_MAX_TOKENS", 50000)),
        dest="max_tokens",
        help="Maximum image tokens (default: $LLM_OCR_MAX_TOKENS or 50000). "
             "Each token covers a 32×32 px patch.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("LLM_OCR_CONCURRENCY", 4)),
        help="Max number of pages processed concurrently (default: $LLM_OCR_CONCURRENCY or 4).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_OCR_MODEL", "qwen3.6-35b-a3b"),
        help="LLM model name (default: $LLM_OCR_MODEL).",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("LLM_ENDPOINT"),
        help="OpenAI-compatible API base URL (default: $LLM_ENDPOINT).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLM_API_KEY"),
        dest="api_key",
        metavar="KEY",
        help="API key (default: $LLM_API_KEY).",
    )
    parser.add_argument(
        "--pages",
        metavar="RANGE",
        help="Page range to process, e.g. '1-5' or '3' (default: all).",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        dest="log_file",
        help="Append status log to this file in addition to stderr.",
    )
    parser.add_argument(
        "--test-config",
        action="store_true",
        dest="test_config",
        help="Send a minimal test request and exit. Validates endpoint/key/model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw bbox response and pixel coordinates for each detected figure.",
    )
    args = parser.parse_args()

    _verbose = args.verbose

    if not args.endpoint:
        print("Error: --endpoint or $LLM_ENDPOINT is required.", file=sys.stderr)
        sys.exit(1)
    if not args.api_key:
        print("Error: --api-key or $LLM_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(base_url=args.endpoint, api_key=args.api_key)

    if args.test_config:
        asyncio.run(async_test_config(client, args.model))
        return

    if not args.pdf_path:
        parser.error("pdf_path is required unless --test-config is used.")
    if not args.output:
        parser.error("--output is required.")

    pdf_path = Path(args.pdf_path).expanduser()
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    if args.log_file:
        _log_file = open(args.log_file, "a", encoding="utf-8")

    try:
        log(f"Rendering PDF at {args.dpi} DPI...")
        from pdf2image import convert_from_path

        all_pages = convert_from_path(str(pdf_path), dpi=args.dpi)
        page_indices = parse_page_range(args.pages, len(all_pages))

        asyncio.run(async_main(args, client, all_pages, page_indices))
    finally:
        if _log_file:
            _log_file.close()


if __name__ == "__main__":
    main()
