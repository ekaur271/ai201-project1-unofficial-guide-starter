# Planning — The Unofficial Guide (OSU CSE)

> ⚠️ **Read before submitting.** This is a working draft that accurately
> reflects the system as built. The assignment explicitly says **do not let an
> AI fill in your planning.md.** Before you submit, rewrite the reasoning in the
> **Chunking Strategy**, **Retrieval Approach**, **Anticipated Challenges**, and
> **AI Tool Plan** sections in your own words — these are the parts graders use
> to check that *you* understand the design. The factual sections (Domain,
> Documents, Evaluation Plan, Architecture) can stay close to this, but make
> sure you can explain every line out loud.

## Domain

Unofficial student knowledge about **CSE professors and courses at Ohio State
University**. Official channels (the course catalog, the CSE department site)
tell you what a course *covers* — they never tell you whether a professor's
exams are fair, whether attendance actually matters, or which instructor to
pick when several teach the same core course. That real signal lives in
scattered student reviews, and a student comparing sections has to read dozens
of them across multiple pages to form a picture. This system makes that
collective student opinion searchable and answerable in one place, with sources.

## Documents

11 documents (66 chunks), two source types for variety:

**Rate My Professors — 10 professor pages** (one document per professor, each
with a stats summary + ~5 verbatim student reviews):
- Jeremy Grifski (CSE 2231) — ratemyprofessors.com/professor/2743316
- Naomi Zweben (CSE 2221) — /professor/2748822
- Christopher Domas (CSE 2421) — /professor/1909019
- Adil Ibrahim (CSE 2431, 1223) — /professor/3035318
- KT Vandergriff (CSE 2321, 2221) — /professor/2942475
- Joe Barker (CSE 3521) — /professor/1628244
- Mark Jackson (CSE 2111) — /professor/2125553
- Luan Duong (CSE 2431, 2321) — /professor/3034233
- George Green (CSE 2431, 3430, 2421) — /professor/2012390
- Michael Fritz (CSE 1224) — /professor/1943610

**OSU CSE Department course website — 1 catalog document:** official
descriptions for CSE 2221/2231/2321/2421/2431/3521 — cse.osu.edu/courses

Coverage spans the intro sequence (1223/1224/2111), the software core
(2221/2231), foundations (2321), systems (2421/2431), and AI (3521), and ranges
from highly-rated professors (Grifski, Zweben, Duong) to poorly-rated ones
(Ibrahim 28%, Vandergriff 37%). Reddit (r/OSU) and osuprofs.com were intended
sources but both block automated fetching — they can be added later by pasting
text into `documents/`.

## Chunking Strategy

**Chunk size:** one review per chunk (~50–400 characters; not a fixed width).
**Overlap:** none.
**Why this fits the documents:** A Rate My Professors review *is* a complete,
self-contained opinion — usually 1–4 sentences. Fixed-width 300-character
slicing would cut one review into two and merge the tail of one reviewer with
the head of the next, so neither piece would be a clean, matchable thought.
Splitting on the natural review boundary instead gives each chunk exactly one
coherent opinion, which is why no character overlap is needed — there is no
sentence being split mid-thought to stitch back together.

The one real risk with review text: an individual review almost never repeats
the professor's name ("he explains concepts clearly", not "Professor Green
explains..."). A bare review embedding therefore can't match a name-based query
like "is George Green good?". The fix is a **context line prepended to every
chunk** — `[<professor> — CSE professor reviews | Rate My Professors]` — which
injects the professor name, course context, and source into the embedded text.
This is the single most important chunking decision in the project.

*How I'd know it's wrong:* too small → retrieval returns fragments with no
standalone meaning and distances stay high; too large → a query for one
professor pulls chunks mentioning three, diluting the answer.

## Retrieval Approach

- **Embedding model:** `all-MiniLM-L6-v2` via sentence-transformers — local, no
  API key, 384-dim, fast. Plenty for short review text.
- **Vector store:** ChromaDB (persistent, cosine distance).
- **top-k = 5.** Opinions are spread across many short reviews, so one or two
  chunks rarely capture the consensus; 5 gathers enough voices without dragging
  in unrelated professors. Too few → miss the consensus; too many → dilute with
  loosely-related reviews that pull the answer off-target.

Semantic search works here because "is the grading harsh?" matches "tough
grader" / "rough tests even with a curve" without sharing those exact words.

*Production tradeoffs (cost no object):* a larger model (e.g. `bge-large`,
OpenAI `text-embedding-3-large`) for sharper distinctions on near-duplicate
reviews; longer context if I later chunked whole threads; multilingual only if
sources weren't English. For this corpus MiniLM is the right call — the bottleneck
is the data, not the embedder.

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Which professor should I take for Software 1 (CSE 2221)? | **Naomi Zweben** — cares about student success, makes the weed-out class feel manageable, posts recordings, fair exams aligned with class. |
| 2 | What do students say about KT Vandergriff's exams? | Very tough, little/no exam practice provided, lots of self-study from slides; split opinions (kind in person, rough tests even with a curve). |
| 3 | Is Luan Duong a good choice for Operating Systems (CSE 2431)? | Yes — called a "gem"/"GOAT", patient, ~13% extra credit, fair-but-hard exams; dry lectures, accent students get used to. |
| 4 | Which CSE professor gives the most useful feedback? | Vandergriff (insightful lab feedback) and Barker's TAs (good HW feedback). **Hard comparative question — likely partial; this is the planned failure case.** |
| 5 | What is the parking situation near the CSE building? | **Out of scope** — no document covers parking; system must refuse. |

## Anticipated Challenges

1. **Comparative/superlative queries** ("the *most* useful feedback", "the
   *best* professor"). Semantic search returns chunks similar to the query, but
   "most" requires ranking *across* professors — the system can only summarize
   whichever reviews happened to retrieve, not truly compare all of them. (Q4.)
2. **Pronoun-only reviews.** Reviews referring to "he/she" with no name are
   unmatchable on their own — mitigated by the prepended context line, but a
   query about a professor with very few reviews could still under-retrieve.
3. **Opinion vs. fact.** Reviews are subjective and often contradict each other;
   the generator must represent disagreement ("reviews are mixed") rather than
   pick a side and state it as fact.
4. **Out-of-scope confidence.** The LLM will happily answer a parking question
   from general knowledge unless grounding is strictly enforced (Q5).

## AI Tool Plan

> Rewrite this section in your own words before submitting — describe what *you*
> actually directed the AI to do.

- **Ingestion + chunking:** gave Claude the Documents + Chunking Strategy
  sections and the document format, asked it to implement `load_documents()`,
  `clean_text()`, and a review-aware `chunk_document()`. Reviewed that it split
  on review boundaries and prepended the context line.
- **Embedding + retrieval:** asked for `embed_and_store()` and `retrieve()`
  against ChromaDB with `all-MiniLM-L6-v2` and source metadata; verified
  distance scores on real queries before adding generation.
- **Generation + UI:** asked for a grounded `generate_response()` (refusal +
  programmatic source attribution) and a Gradio interface; checked that the
  system prompt *enforces* grounding rather than suggesting it.

## Architecture

```
┌──────────────────┐   ┌──────────────┐   ┌─────────────────────────┐
│ Document         │   │  Chunking    │   │ Embedding + Vector Store│
│ Ingestion        │──▶│ 1 review/    │──▶│ all-MiniLM-L6-v2        │
│ documents/*.txt       │   │ chunk +      │   │ → ChromaDB (cosine)     │
│ (RMP + catalog)  │   │ context line │   │   + source metadata     │
│ clean_text()     │   │              │   │                         │
└──────────────────┘   └──────────────┘   └────────────┬────────────┘
                                                        │
                          ┌─────────────────────────────▼───────────┐
                          │ Retrieval                                │
                          │ semantic search, top-k = 5               │
                          └─────────────────────────────┬───────────┘
                                                        │ retrieved chunks
                          ┌─────────────────────────────▼───────────┐
                          │ Generation                               │
                          │ Groq llama-3.3-70b-versatile             │
                          │ grounded prompt + programmatic citations │
                          │ → Gradio UI (app.py)                     │
                          └──────────────────────────────────────────┘
```
