import os
import re
import html

from config import DOCS_PATH


def load_documents():
    """
    Load every .txt document from the docs folder.

    Each file starts with a metadata header block (key: value lines, one per
    line) followed by a blank line and then the body. We parse the header into
    a dict and keep the body as raw text for cleaning + chunking.

    Returns a list of dicts:
      - "title"    : human label, e.g. "Jeremy Grifski — CSE professor reviews"
      - "source"   : where it came from, e.g. "Rate My Professors"
      - "url"      : the source URL
      - "filename" : the .txt file name (used as the attribution key)
      - "body"     : the document body (everything after the header block)
    """
    documents = []
    for filename in sorted(os.listdir(DOCS_PATH)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(DOCS_PATH, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        header, _, body = raw.partition("\n\n")
        meta = _parse_header(header)

        documents.append({
            "title": meta.get("TITLE", filename.replace(".txt", "")),
            "source": meta.get("SOURCE", "Unknown"),
            "url": meta.get("URL", ""),
            "filename": filename,
            "body": body,
        })

    print(f"Loaded {len(documents)} document(s): {[d['filename'] for d in documents]}")
    return documents


def _parse_header(header_block):
    """Turn 'KEY: value' lines into a dict. Ignores lines without a colon."""
    meta = {}
    for line in header_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().upper()] = value.strip()
    return meta


def clean_text(text):
    """
    Normalize a block of text so noise doesn't end up in an embedding.

    Our documents were collected as plain text, but reviews copied from a
    rendered web page commonly carry HTML entities and ragged whitespace.
    This pass:
      - unescapes HTML entities (&amp; -> &, &#39; -> ')
      - strips any leftover HTML tags
      - collapses runs of spaces/tabs and trims each line
    """
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)            # drop stray HTML tags
    text = re.sub(r"[ \t]+", " ", text)            # collapse horizontal space
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def chunk_document(doc):
    """
    Split one document into retrieval-ready chunks.

    Strategy: ONE CHUNK PER REVIEW (content-aware, not fixed-width).

    Each review on Rate My Professors is a self-contained opinion (1-4
    sentences), and each course-catalog entry is a self-contained description.
    Splitting on those natural boundaries means a chunk is never half a review,
    and we don't need character overlap to stitch a thought back together.

    The catch with review text: an individual review almost never repeats the
    professor's name ("he explains concepts clearly", not "Professor Green
    explains..."). If we embedded the bare review, a query like "is George
    Green a good systems professor?" couldn't match it. So we PREPEND a context
    line — "[<title> | <source>]" — to every chunk. That single line injects
    the professor name, the course context, and the source into the embedded
    text, which is what makes name-based queries retrievable.

    Entries are separated by blank lines in the source file. The leading
    "OVERALL: ..." entry becomes a summary chunk (carries the rating numbers);
    each "REVIEW | ..." / "COURSE | ..." entry becomes its own chunk.

    Returns a list of dicts, each with:
      - "text"     : context line + cleaned entry text
      - "metadata" : {source, title, source_name, url, position}
      - "chunk_id" : unique id, e.g. "grifski_jeremy_2"
    """
    body = clean_text(doc["body"])
    entries = [e.strip() for e in re.split(r"\n\s*\n", body) if e.strip()]

    context = f"[{doc['title']} | {doc['source']}]"
    prefix = doc["filename"].replace(".txt", "")

    chunks = []
    for position, entry in enumerate(entries):
        if len(entry) < 25:          # drop fragments with no standalone meaning
            continue
        chunks.append({
            "text": f"{context}\n{entry}",
            "metadata": {
                "source": doc["filename"],
                "title": doc["title"],
                "source_name": doc["source"],
                "url": doc["url"],
                "position": position,
            },
            "chunk_id": f"{prefix}_{position}",
        })

    return chunks


if __name__ == "__main__":
    # Quick standalone check: load, chunk, and print a few samples + the count.
    docs = load_documents()
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_document(d))

    print(f"\nTotal chunks: {len(all_chunks)}\n")
    print("=== 5 sample chunks ===")
    for c in all_chunks[:5]:
        print(f"\n--- {c['chunk_id']} (source: {c['metadata']['source']}) ---")
        print(c["text"])
