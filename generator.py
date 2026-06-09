from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)

# System prompt: grounding is enforced here, not suggested. The model is told
# to answer ONLY from the provided reviews, to refuse when the reviews don't
# cover the question, and to attribute claims to professors by name.
_SYSTEM_PROMPT = """You are the Unofficial Guide, an assistant that answers questions about CSE professors and courses at Ohio State University using ONLY the student reviews and course descriptions provided in the user's message. Follow these rules without exception:

1. Use only the provided passages. Do not use outside or prior knowledge about these professors, courses, or Ohio State — even if you are confident you know the answer.

2. These are student opinions, not facts. Attribute claims to the source: say "students say...", "one reviewer notes...", or "reviews are mixed...". Do not state an opinion as objective truth.

3. Name the professor and, when available, the course the passage refers to. Each passage is labeled with its source title — use that label.

4. If the passages disagree, represent both sides rather than picking one.

5. If the provided passages do not contain enough information to answer, reply with exactly: "I don't have enough information on that in the documents I've collected." Do not fill the gap with general knowledge.

6. Do not invent ratings, quotes, or details that are not in the passages."""


def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved review chunks.

    `retrieved_chunks` is the list returned by retrieve(). Each item is a dict
    with "text", "source", "title", "url", "distance".

    Returns a dict:
      - "answer"  : the grounded answer string
      - "sources" : de-duplicated list of source titles used as context
    Source attribution is guaranteed programmatically from the retrieved
    chunks' metadata — it does not depend on the LLM remembering to cite.
    """
    if not retrieved_chunks:
        return {
            "answer": "I don't have enough information on that in the documents I've collected.",
            "sources": [],
        }

    # Label each passage with its source title so the model can attribute, and
    # so multiple reviews about different professors never get conflated.
    context_blocks = [
        f"[Source {i} — {c['title']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, start=1)
    ]
    context = "\n\n".join(context_blocks)
    user_message = f"Student reviews and course info:\n\n{context}\n\nQuestion: {query}"

    # Groq very occasionally returns empty content; retry once if so.
    answer = ""
    for _ in range(2):
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,   # keep answers close to the retrieved text
        )
        answer = (response.choices[0].message.content or "").strip()
        if answer:
            break

    # Programmatic source attribution: de-dupe titles, preserve retrieval order.
    seen, sources = set(), []
    for c in retrieved_chunks:
        label = f"{c['title']} ({c['url']})" if c["url"] else c["title"]
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return {"answer": answer, "sources": sources}
