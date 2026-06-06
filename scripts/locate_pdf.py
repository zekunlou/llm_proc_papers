#!/usr/bin/env python3
"""Search Zotero for a paper and optionally set up its directory.

Search mode (no --output): searches across title, authors, and abstract
using qmode=everything. Prints a numbered list of matches. If multiple
results are found, prompts interactively or requires --select N (for LLMs).

Setup mode (--output given): creates the paper directory, exports .bib,
and copies the PDF. Prints the copied PDF path to stdout.

Exit codes:
  0  success
  1  error
  2  multiple results, selection required (non-interactive only)
"""

import argparse
import os
import shutil
import sys

import bibtexparser
import textwrap
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pyzotero import zotero


def load_env():
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)


def get_citekey(item):
    """Extract BBT citekey from an item's Extra field, or return empty string."""
    extra = item.get("data", {}).get("extra", "") or ""
    for line in extra.splitlines():
        if line.strip().lower().startswith("citation key:"):
            return line.split(":", 1)[1].strip()
    return ""


def format_authors(creators):
    return ", ".join(
        c.get("lastName", c.get("name", ""))
        for c in creators
        if c.get("lastName") or c.get("name")
    )


def search_items(zot, query, limit=20):
    """Search Zotero across title, authors, and abstract/fulltext."""
    return zot.items(q=query, qmode="everything", limit=limit, itemType="-attachment")


def print_result_list(items):
    """Print a numbered list of search results."""
    for i, item in enumerate(items, 1):
        data = item["data"]
        title = data.get("title", "(no title)")
        authors = format_authors(data.get("creators", []))
        year = (data.get("date", "") or "")[:4]
        citekey = get_citekey(item)
        citekey_str = f"  [{citekey}]" if citekey else ""
        print(f"  {i}. {title}")
        print(f"     {authors} ({year}){citekey_str}")


def print_item_detail(item, pdf_attachments):
    """Print full detail for a selected item."""
    data = item["data"]
    title = data.get("title", "(no title)")
    authors = format_authors(data.get("creators", []))
    year = (data.get("date", "") or "")[:4]
    abstract = data.get("abstractNote", "") or ""
    citekey = get_citekey(item)

    print(f"Title:    {title}")
    print(f"Authors:  {authors}")
    print(f"Year:     {year}")
    print(f"Citekey:  {citekey or '(none — add a BBT citekey in Zotero Extra field)'}")
    if abstract:
        wrapped = textwrap.fill(abstract, width=80, initial_indent="  ", subsequent_indent="  ")
        print(f"Abstract:\n{wrapped}")
    print(f"PDFs ({len(pdf_attachments)}):")
    for att in pdf_attachments:
        fname = att["data"].get("filename", "(unknown)")
        print(f"  - {fname}")


def get_pdf_attachments(zot, item_key):
    children = zot.children(item_key)
    return [
        c for c in children
        if c.get("data", {}).get("contentType") == "application/pdf"
    ]


def local_pdf_path(zotero_data_dir, att_key, filename):
    return Path(zotero_data_dir).expanduser() / "storage" / att_key / filename


def select_item(items, select_n, query):
    """Return the chosen item. Prompts if interactive and select_n is None."""
    if len(items) == 1:
        return items[0]

    if select_n is not None:
        if select_n < 1 or select_n > len(items):
            print(f"Error: --select {select_n} out of range (1–{len(items)}).", file=sys.stderr)
            sys.exit(1)
        return items[select_n - 1]

    # Multiple results, no --select
    print(f"Found {len(items)} results for '{query}':")
    print_result_list(items)

    if not sys.stdin.isatty():
        # Non-interactive (LLM): print list and exit with code 2
        print(
            "\nMultiple results — re-run with --select N to choose one.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Interactive prompt
    while True:
        try:
            raw = input(f"\nSelect [1–{len(items)}] or q to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if raw.lower() == "q":
            sys.exit(0)
        try:
            n = int(raw)
            if 1 <= n <= len(items):
                return items[n - 1]
        except ValueError:
            pass
        print(f"  Enter a number between 1 and {len(items)}.")


def main():
    load_env()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "query",
        help="Free-text search: title words, author name, abstract terms, or citekey.",
    )
    parser.add_argument(
        "--select",
        type=int,
        metavar="N",
        help="Non-interactively select result N from the list (1-indexed). For LLM use.",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        help="Per-paper directory to create. If omitted, runs in search/info mode only.",
    )
    parser.add_argument(
        "--citekey",
        metavar="KEY",
        help="Override the citekey used for output filenames (default: taken from Zotero Extra field).",
    )
    parser.add_argument(
        "--pdf-name",
        metavar="FILENAME",
        help="PDF filename to select when the item has multiple attachments.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of search results to show (default: 20).",
    )
    parser.add_argument(
        "--zotero-data-dir",
        default=os.environ.get("ZOTERO_DATA_DIR", "~/Zotero"),
        metavar="DIR",
        help="Local Zotero data directory (default: $ZOTERO_DATA_DIR).",
    )
    parser.add_argument(
        "--library-id",
        default=os.environ.get("ZOTERO_LIBRARY_ID"),
        metavar="ID",
        help="Zotero library ID (default: $ZOTERO_LIBRARY_ID).",
    )
    parser.add_argument(
        "--library-type",
        default=os.environ.get("ZOTERO_LIBRARY_TYPE", "user"),
        choices=["user", "group"],
        help="Zotero library type (default: $ZOTERO_LIBRARY_TYPE).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ZOTERO_API_KEY"),
        metavar="KEY",
        help="Zotero API key (default: $ZOTERO_API_KEY).",
    )
    args = parser.parse_args()

    if not args.library_id:
        print("Error: --library-id or $ZOTERO_LIBRARY_ID is required.", file=sys.stderr)
        sys.exit(1)
    if not args.api_key:
        print("Error: --api-key or $ZOTERO_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    zot = zotero.Zotero(args.library_id, args.library_type, args.api_key)

    print(f"Searching: '{args.query}'...", file=sys.stderr)
    results = search_items(zot, args.query, limit=args.limit)

    if not results:
        print(f"No results found for '{args.query}'.", file=sys.stderr)
        sys.exit(1)

    item = select_item(results, args.select, args.query)
    item_key = item["key"]
    pdf_attachments = get_pdf_attachments(zot, item_key)

    citekey = args.citekey or get_citekey(item)

    # --- Search/info mode ---
    if args.output is None:
        print_item_detail(item, pdf_attachments)
        return

    # --- Setup mode ---
    if not citekey:
        print(
            "Error: No BBT citekey found in this item's Extra field and --citekey not given.\n"
            "Add a Citation Key to the Extra field in Zotero, or pass --citekey KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not pdf_attachments:
        print(f"Error: No PDF attachments found for '{item['data'].get('title')}'.", file=sys.stderr)
        sys.exit(1)

    if len(pdf_attachments) > 1 and args.pdf_name is None:
        filenames = [a["data"].get("filename", "?") for a in pdf_attachments]
        print("Error: Multiple PDFs found. Re-run with --pdf-name to select one:", file=sys.stderr)
        for fn in filenames:
            print(f"  {fn}", file=sys.stderr)
        sys.exit(1)

    if args.pdf_name:
        selected = [a for a in pdf_attachments if a["data"].get("filename") == args.pdf_name]
        if not selected:
            available = [a["data"].get("filename", "?") for a in pdf_attachments]
            print(f"Error: --pdf-name '{args.pdf_name}' not found. Available: {available}", file=sys.stderr)
            sys.exit(1)
        att = selected[0]
    else:
        att = pdf_attachments[0]

    att_key = att["key"]
    filename = att["data"].get("filename", "")
    src_path = local_pdf_path(args.zotero_data_dir, att_key, filename)

    if not src_path.exists():
        print(
            f"Error: Local PDF not found at {src_path}.\n"
            "Check ZOTERO_DATA_DIR and ensure the file is downloaded in Zotero.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)

    bib_content = zot.item(item_key, format="bibtex")
    if not isinstance(bib_content, str):
        bib_content = bibtexparser.dumps(bib_content)
    (output_dir / f"{citekey}.bib").write_text(bib_content, encoding="utf-8")

    dest_pdf = output_dir / f"{citekey}.pdf"
    shutil.copy2(src_path, dest_pdf)

    print(str(dest_pdf))


if __name__ == "__main__":
    main()
