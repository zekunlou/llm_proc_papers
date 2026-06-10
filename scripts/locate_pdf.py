#!/usr/bin/env python3
"""Search Zotero for a paper and optionally set up its directory.

Search mode (no --output): searches by citekey or free-text query.
Prints a numbered list of matches. If multiple results are found,
prompts interactively or requires --select N (for LLMs).

Setup mode (--output given): creates the paper directory, exports .bib
(if API credentials are available), and copies the PDF. Prints the
copied PDF path to stdout.

Exit codes:
  0  success
  1  error
  2  multiple results, selection required (non-interactive only)
"""

import argparse
import os
import shutil
import sqlite3
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


def print_result_list(items):
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


def local_pdf_path(zotero_data_dir, att_key, filename):
    return Path(zotero_data_dir).expanduser() / "storage" / att_key / filename


def select_item(items, select_n, query):
    if len(items) == 1:
        return items[0]

    if select_n is not None:
        if select_n < 1 or select_n > len(items):
            print(f"Error: --select {select_n} out of range (1–{len(items)}).", file=sys.stderr)
            sys.exit(1)
        return items[select_n - 1]

    label = query or "(citekey search)"
    print(f"Found {len(items)} results for '{label}':")
    print_result_list(items)

    if not sys.stdin.isatty():
        print(
            "\nMultiple results — re-run with --select N to choose one.",
            file=sys.stderr,
        )
        sys.exit(2)

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


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

def _sqlite_conn(db_path):
    uri = f"file:{db_path}?immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _build_item_sqlite(conn, item_key):
    """Build a pyzotero-style item dict from SQLite."""
    row = conn.execute(
        "SELECT itemID FROM items WHERE key = ?", (item_key,)
    ).fetchone()
    if not row:
        return None
    item_id = row["itemID"]

    def field_by_id(fid):
        r = conn.execute(
            "SELECT v.value FROM itemData d JOIN itemDataValues v ON d.valueID=v.valueID WHERE d.itemID=? AND d.fieldID=?",
            (item_id, fid),
        ).fetchone()
        return r["value"] if r else ""

    def field_by_name(fname):
        r = conn.execute(
            """SELECT v.value FROM itemData d
               JOIN itemDataValues v ON d.valueID=v.valueID
               JOIN fields f ON d.fieldID=f.fieldID
               WHERE d.itemID=? AND f.fieldName=?""",
            (item_id, fname),
        ).fetchone()
        return r["value"] if r else ""

    crs = conn.execute(
        """SELECT c.firstName, c.lastName, ct.creatorType
           FROM itemCreators ic
           JOIN creators c ON ic.creatorID=c.creatorID
           JOIN creatorTypes ct ON ic.creatorTypeID=ct.creatorTypeID
           WHERE ic.itemID=? ORDER BY ic.orderIndex""",
        (item_id,),
    ).fetchall()
    creators = [
        {
            "firstName": c["firstName"] or "",
            "lastName": c["lastName"] or "",
            "creatorType": c["creatorType"] or "author",
        }
        for c in crs
    ]

    return {
        "key": item_key,
        "data": {
            "title": field_by_id(1),
            "abstractNote": field_by_id(2),
            "date": field_by_name("date"),
            "extra": field_by_id(16),
            "creators": creators,
        },
    }


def search_by_citekey_sqlite(db_path, citekey):
    conn = _sqlite_conn(db_path)
    try:
        rows = conn.execute(
            """SELECT i.key FROM items i
               JOIN itemData id ON i.itemID=id.itemID
               JOIN itemDataValues idv ON id.valueID=idv.valueID
               WHERE id.fieldID=16
                 AND LOWER(idv.value) LIKE LOWER(?)""",
            (f"%citation key: {citekey}%",),
        ).fetchall()
        return [item for row in rows if (item := _build_item_sqlite(conn, row["key"]))]
    finally:
        conn.close()


def search_items_sqlite(db_path, query, limit=20):
    conn = _sqlite_conn(db_path)
    try:
        pattern = f"%{query}%"
        rows = conn.execute(
            """SELECT DISTINCT i.key FROM items i
               LEFT JOIN itemCreators ic ON i.itemID=ic.itemID
               LEFT JOIN creators c ON ic.creatorID=c.creatorID
               LEFT JOIN itemData td ON i.itemID=td.itemID AND td.fieldID=1
               LEFT JOIN itemDataValues tv ON td.valueID=tv.valueID
               WHERE tv.value LIKE ?
                  OR c.firstName || ' ' || c.lastName LIKE ?
                  OR c.lastName LIKE ?
               LIMIT ?""",
            (pattern, pattern, pattern, limit * 2),
        ).fetchall()
        results = []
        seen = set()
        for row in rows:
            key = row["key"]
            if key not in seen:
                seen.add(key)
                item = _build_item_sqlite(conn, key)
                if item:
                    results.append(item)
                if len(results) >= limit:
                    break
        return results
    finally:
        conn.close()


def get_pdf_attachments_sqlite(db_path, item_key):
    conn = _sqlite_conn(db_path)
    try:
        rows = conn.execute(
            """SELECT att.key, ia.path
               FROM items parent
               JOIN itemAttachments ia ON parent.itemID=ia.parentItemID
               JOIN items att ON ia.itemID=att.itemID
               WHERE parent.key=? AND ia.contentType='application/pdf'""",
            (item_key,),
        ).fetchall()
        return [
            {
                "key": row["key"],
                "data": {
                    "contentType": "application/pdf",
                    "filename": (row["path"] or "").removeprefix("storage:"),
                },
            }
            for row in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# pyzotero backend
# ---------------------------------------------------------------------------

def search_by_citekey_pyzotero(zot, citekey):
    # citekeys live in the Extra field, requires everything scope
    return zot.items(q=citekey, qmode="everything", limit=5, itemType="-attachment")


def search_items_pyzotero(zot, query, limit=20):
    return zot.items(q=query, limit=limit, itemType="-attachment")


def get_pdf_attachments_pyzotero(zot, item_key):
    children = zot.children(item_key)
    return [
        c for c in children
        if c.get("data", {}).get("contentType") == "application/pdf"
    ]


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def do_search_by_citekey(backend, db_path, zot, citekey):
    if backend == "sqlite":
        return search_by_citekey_sqlite(db_path, citekey)
    return search_by_citekey_pyzotero(zot, citekey)


def do_search_items(backend, db_path, zot, query, limit=20):
    if backend == "sqlite":
        return search_items_sqlite(db_path, query, limit)
    return search_items_pyzotero(zot, query, limit)


def do_get_pdf_attachments(backend, db_path, zot, item_key):
    if backend == "sqlite":
        return get_pdf_attachments_sqlite(db_path, item_key)
    return get_pdf_attachments_pyzotero(zot, item_key)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    load_env()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Free-text search: title words, author name, or abstract terms.",
    )
    parser.add_argument(
        "--citekey",
        metavar="KEY",
        help="Search by BBT citekey (from Zotero Extra field). Takes priority over query.",
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
        "--output-key",
        metavar="KEY",
        help="Key used for output filenames (default: citekey from Zotero Extra field).",
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
        "--backend",
        choices=["sqlite", "pyzotero"],
        default="sqlite",
        help="Search backend: sqlite (default, fast, local) or pyzotero (API).",
    )
    parser.add_argument(
        "--zotero-data-dir",
        default=os.environ.get("ZOTERO_DATA_DIR", "~/Zotero"),
        metavar="DIR",
        help="Local Zotero data directory (default: $ZOTERO_DATA_DIR or ~/Zotero).",
    )
    parser.add_argument(
        "--zotero-db",
        metavar="PATH",
        help="Path to zotero.sqlite (default: <zotero-data-dir>/zotero.sqlite).",
    )
    parser.add_argument(
        "--library-id",
        default=os.environ.get("ZOTERO_LIBRARY_ID"),
        metavar="ID",
        help="Zotero library ID (default: $ZOTERO_LIBRARY_ID). Required for pyzotero backend.",
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
        help="Zotero API key (default: $ZOTERO_API_KEY). Required for pyzotero backend.",
    )
    args = parser.parse_args()

    if not args.query and not args.citekey:
        parser.error("at least one of 'query' or '--citekey' is required.")

    data_dir = Path(args.zotero_data_dir).expanduser()
    db_path = Path(args.zotero_db).expanduser() if args.zotero_db else data_dir / "zotero.sqlite"

    zot = None
    if args.backend == "pyzotero":
        if not args.library_id:
            print("Error: --library-id or $ZOTERO_LIBRARY_ID is required for pyzotero backend.",
              file=sys.stderr)
            sys.exit(1)
        if not args.api_key:
            print("Error: --api-key or $ZOTERO_API_KEY is required for pyzotero backend.", file=sys.stderr)
            sys.exit(1)
        zot = zotero.Zotero(args.library_id, args.library_type, args.api_key)

    # --- Search ---
    results = []

    if args.citekey:
        print(f"Searching by citekey: '{args.citekey}'...", file=sys.stderr)
        results = do_search_by_citekey(args.backend, db_path, zot, args.citekey)
        if not results:
            print(f"Error: No item found for citekey '{args.citekey}'.", file=sys.stderr)
            print(
                "Hint: Remember to 'Pin BibTeX key' if you are using Better BibTeX.",
                file=sys.stderr,
            )
            if args.query:
                print(f"Falling back to query: '{args.query}'...", file=sys.stderr)
            else:
                sys.exit(1)

    if not results and args.query:
        print(f"Searching: '{args.query}'...", file=sys.stderr)
        results = do_search_items(args.backend, db_path, zot, args.query, limit=args.limit)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    item = select_item(results, args.select, args.query)
    item_key = item["key"]
    pdf_attachments = do_get_pdf_attachments(args.backend, db_path, zot, item_key)

    citekey_for_output = args.output_key or get_citekey(item)

    # --- Search/info mode ---
    if args.output is None:
        print_item_detail(item, pdf_attachments)
        return

    # --- Setup mode ---
    if not citekey_for_output:
        print(
            "Error: No BBT citekey found in this item's Extra field and --output-key not given.\n"
            "Add a Citation Key to the Extra field in Zotero, or pass --output-key KEY.",
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

    # BibTeX export: try pyzotero if credentials available
    _zot_for_bib = zot
    if _zot_for_bib is None and args.library_id and args.api_key:
        try:
            _zot_for_bib = zotero.Zotero(args.library_id, args.library_type, args.api_key)
        except Exception:
            _zot_for_bib = None

    if _zot_for_bib:
        try:
            bib_content = _zot_for_bib.item(item_key, format="bibtex")
            if not isinstance(bib_content, str):
                bib_content = bibtexparser.dumps(bib_content)
            (output_dir / "bibtex.bib").write_text(bib_content, encoding="utf-8")
        except Exception as e:
            print(f"Warning: BibTeX export failed: {e}", file=sys.stderr)
    else:
        print(
            "Warning: BibTeX export skipped — set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY to enable.",
            file=sys.stderr,
        )

    dest_pdf = output_dir / "main_paper.pdf"
    shutil.copy2(src_path, dest_pdf)

    print(str(dest_pdf))


if __name__ == "__main__":
    main()
