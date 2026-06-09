# The Unofficial Guide — OSU CSE Professors & Courses

A Retrieval-Augmented Generation (RAG) system that answers plain-language
questions about CSE professors and courses at Ohio State, grounded only in real
student reviews — with sources.

## Setup

```bash
cd ai201-project1-unofficial-guide
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your real Groq key in .env
python app.py                 # ingests on first run, launches UI at :7860
```

---

## Domain and Document Sources

Unofficial student knowledge about **CSE professors and courses at Ohio State
University**. The official catalog tells you what a course covers; it never
tells you whether a professor's exams are fair or which section to pick. That
signal lives in scattered student reviews — this system makes it searchable.

**11 documents, two source types:**
- **Rate My Professors (10 professor pages):** Grifski (CSE 2231), Zweben
  (2221), Domas (2421), Ibrahim (2431/1223), Vandergriff (2321/2221), Barker
  (3521), Jackson (2111), Duong (2431/2321), Green (2431/3430/2421), Fritz
  (1224). See `sources.md` for direct URLs.
- **OSU CSE course website (1 catalog doc):** official descriptions for CSE
  2221/2231/2321/2421/2431/3521 — https://cse.osu.edu/courses

Reddit (r/OSU) and osuprofs.com were intended sources but both block automated
fetching; they can be added by pasting text into `documents/`.

## Chunking Strategy

**One review per chunk** (~50–400 chars), **no overlap**, with a **context line
prepended to every chunk**: `[<professor> — CSE professor reviews | Rate My
Professors]`.

A Rate My Professors review is already a complete, self-contained opinion
(1–4 sentences). Fixed-width slicing would cut a review in half and merge the
tail of one reviewer with the head of the next, so no chunk would be a clean,
matchable thought. Splitting on the natural review boundary gives each chunk
exactly one coherent opinion — which is also why no overlap is needed.

The key problem with review text: reviews rarely repeat the professor's name
("he explains concepts clearly", not "Professor Green explains..."), so a bare
review can't match a name-based query. The prepended context line injects the
professor name, course, and source into the embedded text, making name-based
retrieval work. **Result: 66 chunks across 11 documents.**

## Sample Chunks (5, each with source)

1. **`zweben_naomi.txt`** — `[Naomi Zweben — CSE professor reviews | Rate My Professors]` / REVIEW | CSE 2221 | Dec 2025: "Genuinely the model of what a prof/lecturer should be. She explains concepts very well, is accessible outside of class... Software 1 is an intimidating class but she makes it feel easy. Homework assignments and exams are all very fair and she prepares you well."
2. **`vandergriff_kt.txt`** — `[KT Vandergriff — CSE professor reviews | Rate My Professors]` / REVIEW | CSE 2321 | Mar 2026: "DO NOT TAKE FOUNDATIONS 1 WITH THEM. IF YOU ARE ANYTHING LESS THAN STELLAR AT MATH, THIS PROFESSOR IS NOT FOR YOU."
3. **`duong_luan.txt`** — `[Luan Duong — CSE professor reviews | Rate My Professors]` / REVIEW | CSE 2431 | May 2026: "GEM IN THE CSE DEPT. Exams are hard, study well, but he adds a couple point curve and they are fair exams. One midterm, one final, collectively worth 50% of grade. But he offered 13% of extra credit..."
4. **`barker_joe.txt`** — `[Joe Barker — CSE professor reviews | Rate My Professors]` / REVIEW | CSE 3521 | Oct 2025: "Huge ego but maybe the weirdest professor I've ever had. He gets offended when students ask questions so we just stopped asking..."
5. **`cse_course_catalog.txt`** — `[OSU CSE core course catalog descriptions | OSU CSE Department course website]` / COURSE | CSE 2221: Software I: "Intellectual foundations of software engineering; design-by-contract principles; ... This is the first course in the Software sequence and is commonly considered a weed-out course for the CSE major."

## Embedding Model & Production Tradeoffs

**Model used:** `all-MiniLM-L6-v2` via sentence-transformers — local, no API
key, no rate limits, 384-dim, fast. A strong fit for short review text.

**If deploying for real users (cost no object):** I'd consider a larger model
(`bge-large`, OpenAI `text-embedding-3-large`) for sharper separation between
near-duplicate reviews; longer context only if I chunked whole Reddit threads
instead of single reviews; multilingual only if sources weren't English. For
*this* corpus the bottleneck is data coverage, not the embedder, so MiniLM is
the right tradeoff — latency and cost stay at zero.

## Retrieval Test Results (verified, no LLM needed)

Run `python retriever.py` to reproduce. Cosine distance, lower = better.

**Q1 — "Which professor should I take for Software 1 (CSE 2221)?"** (top-4)
| dist | source | chunk |
|------|--------|-------|
| 0.493 | zweben_naomi | "...she makes it feel easy... exams are all very fair" |
| 0.523 | ibrahim_adil | OVERALL stats |
| 0.527 | domas_christopher | OVERALL stats |
| 0.537 | duong_luan | OVERALL stats |

*Why relevant:* the top hit is the correct professor (Zweben teaches 2221) and
the exact review a student asking this would want — it directly addresses
whether Software 1 is manageable under her.

**Q2 — "What do students say about KT Vandergriff's exams?"** (top-4)
| dist | source | chunk |
|------|--------|-------|
| 0.350 | vandergriff_kt | "very tough course... no test practice... self studying" |
| 0.390 | vandergriff_kt | "DO NOT TAKE FOUNDATIONS 1..." |
| 0.396 | vandergriff_kt | "insightful feedback during labs" |
| 0.417 | vandergriff_kt | OVERALL stats |

*Why relevant:* all four chunks are the correct professor with low distances,
and they capture both sides students report — tough/no-practice exams *and*
helpful in-person feedback. The context line is what makes the name match work.

**Q3 — "Is Luan Duong good for Operating Systems (CSE 2431)?"** top hit
`duong_luan` at 0.467, all top-4 from Duong's CSE 2431 reviews — correct prof and course.

**Out-of-scope check — "parking near the CSE building":** best distance 0.731,
all results off-topic → triggers grounded refusal (see Q5 below).

## How Grounded Generation Is Enforced

Grounding is enforced in two places, not left to chance:
1. **System prompt (`generator.py`):** instructs the model to answer *only* from
   the provided passages, to attribute claims to students ("students say...",
   "reviews are mixed"), to represent disagreement, and to reply with an exact
   refusal string — *"I don't have enough information on that in the documents
   I've collected."* — when the passages don't cover the question. `temperature=0.2`
   keeps answers close to the retrieved text.
2. **Programmatic source attribution:** the `sources` list is built from the
   retrieved chunks' metadata in code, so citations are guaranteed regardless of
   whether the LLM adds them.

## Example Responses

**In-scope #1 — "Which professor should I take for Software 1 (CSE 2221)?"**
> According to the reviews, one reviewer notes that Naomi Zweben is "Genuinely
> the model of what a prof/lecturer should be" for CSE 2221 (Software 1), saying
> she explains concepts well, is accessible, and makes the class feel easy.
> Students say her homework assignments and exams are fair, and she prepares
> students well...
>
> **Sources:** Naomi Zweben — Rate My Professors (ratemyprofessors.com/professor/2748822) *(+ the other retrieved chunks)*

**In-scope #2 — "Is Luan Duong a good choice for Operating Systems (CSE 2431)?"**
> Students say that Luan Duong is a great choice for Operating Systems (CSE 2431).
> One reviewer notes that he is "genuinely one of the best professors I have had
> at OSU"... Another calls him a "GEM IN THE CSE DEPT" and praises his caring
> teaching style and significant extra credit... some note exams can be
> challenging, but he adds a curve and offers extra credit. Overall, students
> highly recommend Professor Duong.
>
> **Sources:** Luan Duong — Rate My Professors (ratemyprofessors.com/professor/3034233)

**Out-of-scope — "What is the parking situation near the CSE building?"**
> I don't have enough information on that in the documents I've collected.

This confirms grounding works: no document mentions parking (best retrieval
distance 0.727), so the system refuses instead of inventing an answer.

*Note on attribution:* the Sources list reflects **all top-k retrieved chunks**,
not only the ones quoted — so a focused answer (e.g. Q1, which only uses Zweben)
still lists the other four retrieved sources. A stricter version would cite only
the chunks the LLM actually drew from.

## Query Interface

A Gradio web app (`python app.py`, http://localhost:7860).
- **Input:** a single textbox for a plain-language question (+ example buttons).
- **Output:** an **Answer** box, a **Sources** box (documents the answer drew
  from, with URLs), and a collapsible **Retrieved chunks** panel showing each
  chunk's distance score — so grounding is visible during the demo.

**Sample transcript:**
```
Q: Which professor should I take for Software 1 (CSE 2221)?

Answer: According to the reviews, one reviewer notes that Naomi Zweben is
"Genuinely the model of what a prof/lecturer should be" for CSE 2221 (Software 1),
saying she explains concepts well, is accessible, and makes the class feel easy.
Students say her homework assignments and exams are fair, and she prepares
students well.

Sources:
• Naomi Zweben — CSE professor reviews (ratemyprofessors.com/professor/2748822)
• (plus other retrieved chunks shown in the Retrieved-chunks panel)
```

## Evaluation Report

Run: `python evaluate.py`. Results below are from an actual run.

| # | Question | Expected | System response (summary) | Judgment |
|---|----------|----------|---------------------------|----------|
| 1 | Best professor for Software 1 (CSE 2221)? | Naomi Zweben — caring, makes weed-out class manageable, fair exams | Cites Zweben: "model of what a prof should be," explains well, fair exams, prepares you well | **Accurate** |
| 2 | What do students say about KT Vandergriff's exams? | Very tough, no exam practice, self-study; mixed (kind in person) | "Really rough, even with a curve"; no test practice; lots of self-study; notes reviews are mixed | **Accurate** |
| 3 | Is Luan Duong good for Operating Systems (CSE 2431)? | Yes — "gem", patient, ~13% extra credit, fair-but-hard | "Great choice"; quotes "GEM IN THE CSE DEPT," caring, extra credit, exams hard but curved | **Accurate** |
| 4 | Which CSE professor gives the most useful feedback? | Comparative — should hedge; Vandergriff & Barker's TAs are explicitly praised for feedback | Confidently declares "George Green is the professor who gives the most useful feedback" | **Inaccurate (failure case)** |
| 5 | Parking near the CSE building? | Out of scope → refusal | "I don't have enough information on that in the documents I've collected." | **Accurate** |

### Failure case (confirmed): Q4, "Which professor gives the *most useful* feedback?"

The system answered: *"George Green is the professor who gives the most useful
feedback."* George Green **is** praised for feedback in the reviews, so the
answer isn't fabricated — but the **"most" claim is unsupported**, and the
system never even considered the professors most explicitly praised for
feedback: KT Vandergriff ("she gave really insightful feedback during labs,
including how to fix mistakes") and Joe Barker ("the TAs give good feedback on
the HWs"). Neither chunk made the top-5, so the LLM never saw them.

**Why it failed — a retrieval-architecture limitation, not a generation bug.**
Top-k semantic search returns the 5 chunks closest to the phrase "useful
feedback" and nothing else. A superlative question ("*most* useful") requires
ranking *across all professors*, but the model only ever sees a biased sample of
5 chunks, so it crowns whichever professor happened to surface. It then states
that as a definitive answer rather than hedging that it can't truly compare all
instructors. **Fixes worth exploring:** retrieve a wider net (higher k) before
asking for a comparison, or tighten the system prompt to refuse ranking
questions it can't substantiate across the full corpus.

## Spec Reflection

Writing the chunking section in planning.md before I wrote any code was the part
of the spec that helped me most. When I was reading through the Rate My
Professors reviews, it was tempting to just reach for a fixed 300-character
splitter like the example warned against. But forcing myself to write down *why*
a chunk size fit my documents made me realize that each review is already a
complete thought, so the natural split was one review per chunk. Having that
decision locked in before coding meant I didn't waste time later fighting chunks
that cut reviews in half.

Where my implementation diverged from the spec: I didn't plan for the professor
name problem at all. My original plan was just "split on each review." Once I
actually embedded the chunks and tested retrieval, I noticed that a query like
"is George Green a good professor?" wasn't matching his reviews, because reviews
say "he" or "Professor Green" inconsistently and a lot of them never name him at
all. So I went back and added a context line to the front of every chunk with
the professor's name, course, and source. That wasn't in my original spec — the
data forced it — and I updated planning.md to reflect it.

## AI Usage

I used Claude to help me build this, but I made the design decisions and checked
its work at each step. Three specific instances:

1. **Review-aware chunking.** I gave Claude my chunking strategy section and the
   format of my document files and asked it to write the `chunk_document()`
   function. The first thing I checked was whether it actually split on review
   boundaries instead of a fixed width, and I'm the one who decided to prepend
   the professor context line after retrieval testing showed name-based queries
   were failing — I directed that change rather than it being suggested.

2. **Grounding enforcement.** I had Claude write the generation prompt, but I
   specifically checked that the system prompt *enforced* answering only from the
   retrieved reviews — including an exact refusal string for out-of-scope
   questions — instead of just politely suggesting it. I also made sure the
   source list was built from the chunk metadata in code, so citations don't
   depend on the LLM remembering to add them.

3. **The empty-answer bug.** During my evaluation run, question 2 came back with
   a blank answer once. Instead of assuming the whole thing was broken, I re-ran
   it a few times, saw it was a transient empty response from Groq, and added a
   one-retry guard rather than a bigger change. That told me it was a flaky API
   response, not a real pipeline bug.
