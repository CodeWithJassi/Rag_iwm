"""Every prompt in the system. Kept together so prompt iteration never touches
pipeline code, and so you can diff prompt changes against retrieval quality.
"""

# ==================================================================== summarise
CHUNK_PROMPT = """You are a senior buy-side analyst reading one section of an equity research report on {company}.
Extract ONLY facts present in the text — do not invent or extrapolate. Pull out:
- Broker name, analyst name, recommendation (Buy/Hold/Sell), CMP, target price, valuation method
- Key investment thesis points (max 4 bullets)
- Financial figures: revenue, EBITDA, EBITDA margin, PAT, EPS, ROE, ROCE (historical and forecast)
- Key assumptions: volume growth, margins, capex, order inflow, utilisation, pricing
- Risks mentioned (company-specific and macro)
Output as brief bullet points only. Skip anything not present in the text.

SECTION:
{text}
"""

REDUCE_PROMPT = """You are a senior buy-side fund manager writing a quick internal note on {company} for an investment committee.
Use ONLY the facts provided below — do not invent numbers. Keep it tight, under 500 words total.
Structure it as: Recommendation & target, Thesis, Financials, Key assumptions, Risks.

EXTRACTED FACTS:
{text}
"""

# Extended from the original PRICE_PROMPT: same single call now also returns the
# broker and the rating, both of which the dashboard needs.
FACTS_PROMPT = """Extract four fields from this equity research report on {company}.
Use ONLY values explicitly stated in the text. If a field is absent, use null.

- broker: the research house publishing the report (e.g. "Motilal Oswal", "ICICI Securities")
- recommendation: exactly one of "Buy", "Hold", "Sell", or null. Map synonyms:
  Accumulate/Add/Outperform -> Buy; Neutral/Equal-weight -> Hold; Reduce/Underperform -> Sell
- current_price: the current market price (CMP), as a number, no currency symbol
- target_price: the analyst's target price, as a number, no currency symbol

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"broker": ..., "recommendation": ..., "current_price": ..., "target_price": ...}}

REPORT TEXT:
{text}
"""

# ==================================================================== query prep
CONTEXTUALIZE_PROMPT = """Given a chat history and the user's latest message, rewrite the latest message as a standalone question that makes sense without the history.

Rules:
- Resolve pronouns and references ("it", "that number", "the same period") using the RECENT messages first. Older messages are less likely to be what the user is referring to.
- The LAST assistant + user pair is the most important context. Only reach further back when the latest message's reference is clearly to an earlier topic.
- Do NOT answer the question. Do NOT add information not implied by the history.
- If the latest message is already standalone, return it unchanged.
- Return ONLY the rewritten question, nothing else.

CHAT HISTORY (oldest first):
{history}

LATEST MESSAGE:
{query}
"""

DECOMPOSE_PROMPT = """You are preparing a search plan for a question about an equity research report on {company}.

Break the question into the minimum set of independent sub-questions needed to answer it fully.

CRITICAL — SCOPE RULES:
- Each sub-question MUST be about {company} specifically. Do NOT generalise to the
  industry, sector, peers, or market unless the original question explicitly asks
  about those. "Growth in Thyrocare's diagnostic business" stays "Thyrocare's
  diagnostic growth" — it does NOT become "industry growth forecast."
- Each sub-question must be NARROWER than or EQUAL to the original. If the
  original asks about one company and one metric, every sub-question must also
  be about that company and that metric.
- NEVER broaden scope. The sub-questions collectively must cover exactly what
  was asked — nothing more, nothing less.

SPLITTING RULES:
- A simple, single-topic question yields exactly ONE sub-question: the question
  itself. Do not invent extra angles.
- Only split when the question genuinely asks for separate things that need
  separate evidence (e.g. "compare revenue growth and margin trends" → 2
  sub-questions; "what is the target price" → 1).
- Maximum {max_subs} sub-questions.
- NEVER generate near-duplicates. If two sub-questions would retrieve the same
  evidence, merge them.
- Paired facets like "domestic and exports", "revenue and profit", "volume and
  pricing" usually appear in the SAME paragraphs of a report. Do NOT split them
  into separate sub-questions — one sub-question covering both facets will
  retrieve the right evidence and produce a cleaner answer.

Respond with ONLY a JSON array of strings, no prose, no markdown fences.
Example: ["What is the target price for {company}?", "What are the key downside risks for {company}?"]

QUESTION:
{query}
"""


REPHRASE_PROMPT = """Write {n} different search queries that would each retrieve passages answering this question from an equity research report.

Vary the vocabulary and framing — use the terminology a research analyst would write in the report itself, not the terminology of the question. Include the original phrasing as one of them.

Respond with ONLY a JSON array of {n} strings, no prose, no markdown fences.

QUESTION:
{query}
"""

# ==================================================================== generate
SUB_ANSWER_PROMPT = """Answer the question using ONLY the numbered context passages below, which come from an equity research report on {company}.

Rules:
- If the passages contain NO facts that address any part of the question, reply
  exactly: INSUFFICIENT
- If the question asks about multiple things (e.g. "domestic and exports") and you
  can only answer some of them, provide what you found and briefly note what is
  missing. Do NOT reply INSUFFICIENT when you have partial information — a partial
  answer with a noted gap is better than silence.
- Never use outside knowledge. Never estimate or infer a number that is not written down.
- SCAN each passage for explicit numbers, percentages, basis points, or rupee amounts
  that answer the question. If a passage says "decline of ~6%" and the question asks
  about a percentage decline, that is your answer — state it directly.
- You MUST cite at least one passage number for EVERY sentence you write, like [1] or [3].
  This is not optional — every factual statement needs a [n] marker pointing to its source.
  A sentence without a citation is a hallucination. Cite more than one where relevant.
- Write in third person, referring to the company by name. Do NOT use "we," "our," "us,"
  or any first-person language — you are summarising a report, you are not the analyst
  who wrote it.
- Answer directly. Do NOT start with "Yes," "No," "According to [1]," "Based on the
  report," or any similar preamble. State the facts immediately.
- Provide a complete, self-contained answer in 1–3 sentences. The reader should not
  need to read the question to understand your answer.
- If the context contains similar figures or metrics from different sections or time
  periods, explicitly state which section or period each figure comes from. Do not
  conflate numbers from different contexts.

CONTEXT PASSAGES:
{context}

QUESTION:
{query}
"""

SYNTHESIZE_PROMPT = """You are a senior analyst answering a colleague's question about an equity research report on {company}.

Below are the sub-questions that were researched and the answer found for each. Combine them into one coherent answer to the original question.

Rules:
- Use ONLY the sub-answers below. Do not add facts, numbers or reasoning of your own.
- You MUST preserve every [n] citation marker exactly as it appears in the sub-answers.
  Every factual statement you write must carry at least one [n] citation. This is not optional.
- Write in third person. Do NOT use "we," "our," "us," or any first-person language.
- Answer directly. Do NOT start with "Yes," "No," "According to," "Based on the
  report," or any similar preamble. State the facts immediately.
- Write as flowing prose or tight bullets, whichever suits the question. No headings.
- If every sub-answer is INSUFFICIENT, reply exactly: INSUFFICIENT
- Some sub-answers may be marked INSUFFICIENT — those facets could not be answered
  from the report. Do NOT paste the word INSUFFICIENT into your response. Synthesise
  only from the sub-answers that have real content. If the original question asks
  about multiple facets and some are missing, briefly note the gap at the end
  (e.g. "The report does not provide a separate breakdown for export sales.")
{missing_note}
ORIGINAL QUESTION:
{query}

RESEARCHED SUB-ANSWERS:
{sub_answers}
"""

# ==================================================================== judge
# Deep-search mode only. These grade the answer we already generated -- they do
# not need ground truth, only the answer and the passages it claims to rest on.
JUDGE_PROMPT = """You are auditing an AI-generated answer for an investment research desk.

Score two things from 0.0 to 1.0:

1. "faithfulness": What fraction of the factual claims in the ANSWER are directly
   supported by the CONTEXT? Evaluate claims against ALL passages — do NOT penalise
   a claim just because its [n] citation marker points to the wrong passage number.
   If the claim's content appears anywhere in the context, it is supported.
   - A claim with a number not present anywhere in the context is unsupported.
   - If the answer merely summarises or restates what is in the context without
     fabricating, score high even if some details are missing.
   - Score 0.0 only if the answer clearly invents numbers, facts, or entities not
     in the context.
   - Statements like "the report does not provide X" or "the report does not
     contain Y" are NOT factual claims — they are meta-commentary about report
     coverage. Do NOT count them as unsupported claims and do NOT let them lower
     the faithfulness score. Only evaluate substantive factual assertions.

2. "relevancy": How well does the CONTEXT actually address the QUESTION? Score low
   if the passages are about the right company but the wrong topic.

Also list any claim in the answer that is clearly not supported by the context.
Only flag material fabrications — omitted details are not unsupported claims.
Do NOT flag "the report does not provide..." statements as unsupported.

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"faithfulness": 0.0, "relevancy": 0.0, "unsupported": ["..."]}}

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
{answer}
"""

# ==================================================================== chunking
# Semantic chunking: the LLM reads a block of text and identifies where the
# topic or subject shifts — natural breakpoints for chunk boundaries.
SEMANTIC_CHUNK_PROMPT = """You are preparing an equity research report for semantic indexing.

Below is a section of a broker report. Identify where the TOPIC or SUBJECT shifts — where the text moves from one idea to a different one. These are the natural breakpoints where a new chunk should start.

Rules:
- Return the PARAGRAPH INDICES (0-based) where a new chunk should BEGIN.
- A new chunk starts when: the topic changes (e.g. from revenue to margins), the section implicitly shifts (e.g. from thesis to risks), or a new entity/financial metric is introduced.
- Do NOT split mid-thought — err on the side of fewer splits.
- If the entire block is one coherent topic, return an empty array.
- Paragraph index 0 is always the start of the first chunk — do NOT include 0.

Respond with ONLY a JSON array of integers, no prose, no markdown fences.
Example: [3, 7, 11] means chunks start at paragraphs 0, 3, 7, and 11.

TEXT (paragraphs numbered for reference):
{paragraphs}
"""
