import gradio as gr

from ingest import load_documents, chunk_document
from retriever import embed_and_store, retrieve, get_collection
from generator import generate_response


# ---------------------------------------------------------------------------
# Ingestion — runs once on startup
# ---------------------------------------------------------------------------

def run_ingestion():
    """Load docs, chunk them, embed into ChromaDB. Skips if already populated.

    To re-ingest after changing chunking, delete ./chroma_db and restart.
    """
    collection = get_collection()
    if collection.count() > 0:
        print(f"Vector store already populated ({collection.count()} chunks). Skipping ingestion.")
        print("To re-ingest, delete the ./chroma_db folder and restart.")
        return

    print("Ingesting documents...")
    all_chunks = []
    for doc in load_documents():
        all_chunks.extend(chunk_document(doc))

    if all_chunks:
        embed_and_store(all_chunks)
        print(f"Ingestion complete. {len(all_chunks)} chunks stored.")
    else:
        print("\n⚠️  No chunks produced — check ingest.py / the docs folder.\n")


# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------

def handle_query(question):
    if not question or not question.strip():
        return "Ask a question to get started.", ""
    retrieved = retrieve(question)
    result = generate_response(question, retrieved)

    sources = "\n".join(f"• {s}" for s in result["sources"]) or "(none)"
    # Show retrieved chunks + distances so grounding is visible in the demo.
    retrieved_view = "\n\n".join(
        f"[{r['distance']:.3f}] {r['text']}" for r in retrieved
    )
    return result["answer"], sources, retrieved_view


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(primary_hue="red"), title="The Unofficial Guide") as demo:
    gr.HTML("""
        <div style="text-align:center; padding:1.25rem 0 0.5rem;">
            <h1 style="font-size:2rem; font-weight:700; color:#7f1d1d; margin:0;">
                🎓 The Unofficial Guide — OSU CSE
            </h1>
            <p style="color:#6b7280; font-size:1rem; margin:0.4rem 0 0;">
                Ask about OSU CSE professors and courses. Answers come only from
                real student reviews — with sources.
            </p>
        </div>
    """)

    inp = gr.Textbox(
        label="Your question",
        placeholder='e.g. "Which professor is best for Software 1?"',
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Sources (documents used)", lines=3)
    with gr.Accordion("Retrieved chunks (distance score + text)", open=False):
        retrieved_view = gr.Textbox(label="", lines=12)

    gr.Examples(
        examples=[
            "Which professor is best for Software 1 (CSE 2221)?",
            "What do students say about KT Vandergriff's exams?",
            "Is Luan Duong good for operating systems?",
            "Which CSE professor gives the most useful feedback?",
            "Who teaches the easiest intro programming class?",
            "What is the parking situation near the CSE building?",
        ],
        inputs=inp,
    )

    btn.click(handle_query, inputs=inp, outputs=[answer, sources, retrieved_view])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources, retrieved_view])


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  The Unofficial Guide — starting up")
    print("=" * 50 + "\n")
    run_ingestion()
    demo.launch()
