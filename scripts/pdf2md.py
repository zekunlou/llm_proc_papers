#!/usr/bin/env python3
"""Convert an academic PDF to markdown with extracted figures.

Two-pass LLM processing per page:
  Pass 1: extract text as single-column markdown (citations escaped as \[x\])
  Pass 2: detect figures as bounding boxes (0-999 normalized coords), crop and save

Output: {output}/{citekey}.md + {output}/assets/{sub_label}.png files
"""

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from PIL import Image

PROMPT_TEXT_TEMPLATE = """\
Convert this academic paper page to markdown. Follow these rules exactly:
- Output single column only. If the page has two columns, process the left column first, then the right column, maintaining reading order.
- Escape all citation brackets with a backslash: write \\[1\\] instead of [1], \\[Smith, 2023\\] instead of [Smith, 2023].
- For each figure, diagram, chart, or non-text visual element: insert a placeholder on its own line using the format ![sub_label]({assets}/sub_label.png), where sub_label is a short descriptive name in lowercase with underscores (e.g. figure_1, band_structure_plot). Place the figure caption immediately after the placeholder.
- Tables should be rendered as markdown tables.
- Math should be rendered as LaTeX inline ($...$) or block ($$...$$).
- Output markdown only. No preamble, no explanation, no commentary.\
"""

PROMPT_BBOX = """\
This is a page from an academic paper. Mark only data-bearing scientific figures using bounding boxes.

A figure IS: a plot, graph, chart, schematic diagram, simulation result, microscopy image, or scientific illustration that conveys data or a scientific concept. It must occupy meaningful 2D area — roughly square or wider-than-tall with substantial height.

A figure is NOT: plain text, section headers, author affiliations, abstract text, inline or display equations, formula blocks, tables, logos, journal stamps, arXiv identifiers, page numbers, or any decorative/layout element. Do NOT mark thin horizontal bands (these are usually equations), do NOT mark the entire page, do NOT mark any region that is mostly text or symbols.

Output a raw JSON array. Do not wrap it in markdown code fences (no ```). No explanation before or after:
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "figure", "sub_label": "figure_1"},
  ...
]

Rules:
- Coordinates are on a 0–999 scale relative to the image dimensions.
- Each box must be tight around the figure content only (not surrounding whitespace or captions).
- A valid figure box must have both width and height greater than 10% of the page dimension (i.e. at least 100 in the 0–999 scale on each axis). Don't be too tight, it is fine to include some surrounding whitespace, but it is important to include the entire figure content.
- label must be "figure".
- sub_label: lowercase, underscores, descriptive (e.g. figure_1, band_structure, phase_diagram_a). Use the same sub_label you would assign in a text extraction pass for this page.
- If there are subfigures but are neighboring, just group them into one box with a single sub_label (e.g. figure_2) rather than trying to label them separately (e.g. figure_2a, figure_2b).
- If no scientific figures are present on this page, output exactly: []\
"""


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


def call_llm(client, model, b64_image, prompt, min_pixels, max_pixels):
    """Send one image+prompt to the LLM and return the response text."""
    response = client.chat.completions.create(
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


def parse_bbox_response(text):
    """Extract the JSON array of bboxes from LLM response."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Find the JSON array
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


def process_page(client, model, img, min_pixels, max_pixels, assets_dir, page_num, assets_name="assets", verbose=False):
    """Run two-pass OCR on one page image. Returns markdown string for the page."""
    b64 = image_to_base64(img)
    img_w, img_h = img.size
    prompt_text = PROMPT_TEXT_TEMPLATE.format(assets=assets_name)

    print(f"  Pass 1: text extraction...", file=sys.stderr)
    markdown = call_llm(client, model, b64, prompt_text, min_pixels, max_pixels)

    print(f"  Pass 2: figure detection...", file=sys.stderr)
    bbox_text = call_llm(client, model, b64, PROMPT_BBOX, min_pixels, max_pixels)

    if verbose:
        print(f"  [verbose] raw bbox response:\n{bbox_text}", file=sys.stderr)

    bboxes = parse_bbox_response(bbox_text)

    if bboxes:
        print(f"  Found {len(bboxes)} figure(s), cropping...", file=sys.stderr)
    for bbox in bboxes:
        coords = bbox.get("bbox_2d", [])
        sub_label = bbox.get("sub_label", f"figure_p{page_num}")
        if len(coords) != 4:
            continue
        # Reject boxes too small on either axis (< 10% of page in normalized coords)
        if (coords[2] - coords[0]) < 100 or (coords[3] - coords[1]) < 100:
            if verbose:
                print(f"  [verbose] {sub_label}: skipped (too small: {coords})", file=sys.stderr)
            continue
        x1, y1, x2, y2 = denormalize_bbox(coords, img_w, img_h)
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if verbose:
            print(
                f"  [verbose] {sub_label}: norm={coords} → px=({x1},{y1},{x2},{y2}) "
                f"size={x2-x1}×{y2-y1} (image {img_w}×{img_h})",
                file=sys.stderr,
            )
        cropped = img.crop((x1, y1, x2, y2))
        out_path = assets_dir / f"{sub_label}.png"
        cropped.save(out_path, format="PNG")

    return markdown


def test_config(client, model):
    """Send a minimal text-only request to verify the endpoint/key/model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=2000,  # thinking model needs budget before it can write content
        )
        msg = response.choices[0].message
        reply = (msg.content or "").strip()
        print(f"Config OK. Model: {model}")
        print(f"Response: {reply}")
        sys.exit(0)
    except Exception as e:
        print(f"Config FAILED: {e}", file=sys.stderr)
        sys.exit(1)


def main():
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
        default=150,
        help="Resolution for PDF page rendering (default: 150).",
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

    if not args.endpoint:
        print("Error: --endpoint or $LLM_ENDPOINT is required.", file=sys.stderr)
        sys.exit(1)
    if not args.api_key:
        print("Error: --api-key or $LLM_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    if args.test_config:
        test_config(client, args.model)
        return  # unreachable, test_config exits

    if not args.pdf_path:
        parser.error("pdf_path is required unless --test-config is used.")
    if not args.output:
        parser.error("--output is required.")

    pdf_path = Path(args.pdf_path).expanduser()
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output).expanduser()
    assets_dir = output_dir / args.assets
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    min_pixels = args.min_tokens * 32 * 32
    max_pixels = args.max_tokens * 32 * 32

    print(f"Rendering PDF at {args.dpi} DPI...", file=sys.stderr)
    # Import here so --test-config works without pdf2image installed
    from pdf2image import convert_from_path

    all_pages = convert_from_path(str(pdf_path), dpi=args.dpi)
    page_indices = parse_page_range(args.pages, len(all_pages))
    print(f"Processing {len(page_indices)} of {len(all_pages)} pages.", file=sys.stderr)

    page_markdowns = []
    for idx in page_indices:
        img = all_pages[idx]
        page_num = idx + 1
        print(f"Page {page_num}/{len(all_pages)}...", file=sys.stderr)
        md = process_page(
            client, args.model, img, min_pixels, max_pixels, assets_dir, page_num,
            assets_name=args.assets, verbose=args.verbose,
        )
        page_markdowns.append(md)

    combined = "\n\n---\n\n".join(page_markdowns)
    md_name = args.md if args.md.endswith(".md") else args.md + ".md"
    out_md = output_dir / md_name
    out_md.write_text(combined, encoding="utf-8")
    print(f"Written: {out_md}", file=sys.stderr)
    print(str(out_md))


if __name__ == "__main__":
    main()
