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

# DECOMPOSE_PROMPT = """You are preparing a search plan for a question about an equity research report on {company}.

# Break the question into the minimum set of independent sub-questions needed to answer it fully.
# - A simple question yields exactly ONE sub-question: the question itself.
# - Only split when the question genuinely asks for separate things that need separate evidence
#   (e.g. "compare revenue growth and margin trends" → 2 sub-questions; "what is the target price" → 1).
# - Maximum {max_subs} sub-questions. Each must be self-contained and searchable on its own.
# - NEVER generate near-duplicates. If two sub-questions would retrieve the same evidence, merge them.

# Respond with ONLY a JSON array of strings, no prose, no markdown fences.
# Example: ["What is the target price?", "What are the key downside risks?"]

# QUESTION:
# {query}
# """

DECOMPOSE_PROMPT = """You are preparing a search plan for a question about an equity research report on {company}.

IF the question is simple like "Why did Ronny said Sam is Apple of his eyes?" can be simplified or diversify to find more relevant chunks, so making more questions like "the reason behind ronny's statement for SAm to be loved one?"
THEN do this:
  [Generate 3 different variations of this query that would help retrieve relevant documents:
  Original query: {query}
  Return 3 alternative queries that rephrase or approach the same question from different angles.]

IF the question is made up of several other questions like "When was Ram born and died?" can be simplified into two questions for better search, i.e., "When was Ram born?" and "When was Ram died?".
THEN do this:
  [Break the question into the minimum set of independent sub-questions needed to answer it fully.
  - A simple question yields exactly ONE sub-question: the question itself.
  - Only split when the question genuinely asks for separate things that need separate evidence
  (e.g. "compare revenue growth and margin trends" → 2 sub-questions; "what is the target price" → 1).
  - Maximum {max_subs} sub-questions. Each must be self-contained and searchable on its own.
  - NEVER generate near-duplicates. If two sub-questions would retrieve the same evidence, merge them.
  query : {query}
  ]

Respond with ONLY a JSON array of strings, no prose, no markdown fences, add your generated question with the original one.
Example: ["What is the target price?", "What are the key downside risks?"]

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
- If the passages do not contain the answer, reply exactly: INSUFFICIENT
- Never use outside knowledge. Never estimate or infer a number that is not written down.
- You MUST cite at least one passage number for EVERY sentence you write, like [1] or [3].
  This is not optional — every factual statement needs a [n] marker pointing to its source.
  A sentence without a citation is a hallucination. Cite more than one where relevant.
- Be direct. No preamble.

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
- Write as flowing prose or tight bullets, whichever suits the question. No headings.
- If every sub-answer is INSUFFICIENT, reply exactly: INSUFFICIENT
- Always provide answers in full sentences that restate the subject clearly, rather than just giving short fragments. Ensure responses are explicit, self-contained, and easy to understand without additional context.

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
   supported by the CONTEXT? A claim with a number not present in the context is
   unsupported. If the answer merely summarises or restates what is in the context
   without fabricating, score high even if some details are missing. Score 0.0 only
   if the answer clearly invents numbers, facts, or entities not in the context.

2. "relevancy": How well does the CONTEXT actually address the QUESTION? Score low
   if the passages are about the right company but the wrong topic.

Also list any claim in the answer that is clearly not supported by the context.
Only flag material fabrications — omitted details are not unsupported claims.

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"faithfulness": 0.0, "relevancy": 0.0, "unsupported": ["..."]}}

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
{answer}
"""
