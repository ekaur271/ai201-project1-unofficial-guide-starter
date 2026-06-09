# Planning — The Unofficial Guide (OSU CSE)

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

**Chunk size:** one review per chunk (about 50–400 characters — not a fixed width).
**Overlap:** none.

I decided to chunk by review instead of by a fixed character count because when
I read through my documents, each Rate My Professors review is already a
complete little opinion, usually only 1–4 sentences. If I used a 300-character
splitter like the starter example, it would cut a review in half and glue the
end of one student's review onto the start of another's, so a chunk wouldn't be
a clean thought I could match a question against. Splitting on the review
boundary means every chunk is exactly one opinion, and that's also why I don't
need any overlap — I'm never cutting a sentence in the middle, so there's
nothing to stitch back together.

The tricky part with reviews is that they almost never say the professor's name
— they just say "he explains things well" or "her exams are hard." So if I just
embedded the raw review, a question like "is George Green a good professor?"
wouldn't match his reviews at all. To fix that, I prepend a context line to
every chunk that has the professor's name, course, and source, like
`[George Green — CSE professor reviews | Rate My Professors]`. That one line is
what makes name-based questions actually work, and it's the most important
chunking decision I made.

I'd know my chunks were wrong if they were too small (retrieval would return
fragments that don't mean anything on their own and distance scores would stay
high) or too large (a question about one professor would pull back chunks
mentioning three different professors and the answer would get watered down).

## Retrieval Approach

I'm using **all-MiniLM-L6-v2** through sentence-transformers for embeddings and
**ChromaDB** (cosine distance) as the vector store. I picked MiniLM because it
runs locally with no API key or rate limits and it's plenty good for short
review text.

I set **top-k = 5**. Since opinions about a professor are spread across a bunch
of short reviews, grabbing only one or two chunks usually isn't enough to get
the full picture, but if I grab too many I start pulling in reviews about other
professors that water down the answer. Five felt like enough voices to summarize
a consensus without dragging in unrelated stuff, and my retrieval testing backed
that up. Semantic search is what makes this work even when the words don't match
— a question like "is the grading harsh?" still finds reviews that say "tough
grader" or "rough tests even with a curve."

If I were deploying this for real and cost wasn't an issue, I'd think about
switching to a bigger embedding model (like bge-large or OpenAI's
text-embedding-3-large) to better tell apart reviews that sound really similar,
and I'd care about longer context if I ever chunked whole Reddit threads instead
of single reviews. Multilingual support wouldn't matter here since everything is
in English. But for this project the real limit is how many reviews I collected,
not the embedding model, so MiniLM is the right call.

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Which professor should I take for Software 1 (CSE 2221)? | **Naomi Zweben** — cares about student success, makes the weed-out class feel manageable, posts recordings, fair exams aligned with class. |
| 2 | What do students say about KT Vandergriff's exams? | Very tough, little/no exam practice provided, lots of self-study from slides; split opinions (kind in person, rough tests even with a curve). |
| 3 | Is Luan Duong a good choice for Operating Systems (CSE 2431)? | Yes — called a "gem"/"GOAT", patient, ~13% extra credit, fair-but-hard exams; dry lectures, accent students get used to. |
| 4 | Which CSE professor gives the most useful feedback? | Vandergriff (insightful lab feedback) and Barker's TAs (good HW feedback). **Hard comparative question — likely partial; this is the planned failure case.** |
| 5 | What is the parking situation near the CSE building? | **Out of scope** — no document covers parking; system must refuse. |

## Anticipated Challenges

1. **"Most" and "best" questions.** I'm worried about questions like "which
   professor gives the *most* useful feedback," because semantic search only
   pulls back the chunks closest to my question, not every professor. So the
   system can only summarize whichever reviews happened to come up — it can't
   actually rank all the professors against each other. (This ended up being my
   failure case, Q4.)
2. **Reviews that only use pronouns.** Since reviews say "he" or "she" instead of
   the name, a review on its own is hard to match. My context line helps with
   this, but a professor with only a couple of reviews could still get
   under-retrieved.
3. **Opinions that contradict each other.** Reviews are subjective and students
   disagree all the time, so I need the generator to say "reviews are mixed"
   instead of just picking one side and stating it like it's a fact.
4. **The model answering things it shouldn't.** If I don't lock down the prompt,
   the LLM will happily answer something like a parking question from its own
   general knowledge instead of admitting the documents don't cover it (Q5).

## AI Tool Plan

I plan to use Claude to help me write the code for each stage, but I'm making
the design decisions myself and checking its output before I trust it.

- **Ingestion + chunking:** I'll give Claude my Documents and Chunking Strategy
  sections plus the format of my document files, and ask it to write
  `load_documents()`, `clean_text()`, and a review-aware `chunk_document()`. I'll
  check that it actually splits on review boundaries and prepends my context
  line, not just slices by character count.
- **Embedding + retrieval:** I'll ask it to write `embed_and_store()` and
  `retrieve()` against ChromaDB using all-MiniLM-L6-v2 with source metadata, and
  I'll test the distance scores on real questions before I add any generation.
- **Generation + UI:** I'll ask it to write the grounded `generate_response()`
  (with a refusal for out-of-scope questions and programmatic source citations)
  and the Gradio interface, and I'll read the system prompt to make sure it
  actually enforces grounding instead of just suggesting it.

## Architecture

```
┌──────────────────┐   ┌──────────────┐   ┌─────────────────────────┐
│ Document         │   │  Chunking    │   │ Embedding + Vector Store│
│ Ingestion        │──▶│ 1 review/    │──▶│ all-MiniLM-L6-v2        │
│ documents/*.txt  │   │ chunk +      │   │ → ChromaDB (cosine)     │
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
