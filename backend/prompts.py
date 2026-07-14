"""Every prompt in the system. Kept together so prompt iteration never touches
pipeline code, and so you can diff prompt changes against retrieval quality.
"""

# ==================================================================== summarise
CHUNK_PROMPT = """You are a senior buy-side analyst reading one section of an equity research report on {company}.

Extract ONLY facts explicitly stated in the text.  DO NOT guess, infer, compute, or
extrapolate any number that is not written down.  A hallucinated number is worse
than a gap — the desk acts on these figures.

PLACEHOLDER RULES:
- If a figure is discussed but the exact number is not stated, write "--"
- If a field (broker, rating, CMP, etc.) is completely absent from this section,
  write "NOT FOUND" — do not carry over a value you saw in a different section
- If a number is stated but you are uncertain about its unit or context, include it
  but append "[?]" — e.g. "Revenue: Rs 450[?] Mn"

Extract:
- Broker name, analyst name, recommendation (Buy/Hold/Sell), CMP, target price,
  valuation method — use "NOT FOUND" for each genuinely missing field
- Key investment thesis points (max 4 bullets)
- Financial figures WITH UNITS: revenue, EBITDA, EBITDA margin, PAT, EPS, ROE,
  ROCE (label each as historical or forecast, and the period — e.g. "FY26E")
- Key assumptions: volume growth, margins, capex, order inflow, utilisation, pricing
- Risks mentioned (company-specific and macro)

Output as brief bullet points only.  Every number MUST carry its unit.

SECTION:
{text}
"""

REDUCE_PROMPT = """You are a senior buy-side fund manager writing a quick internal note on {company} for an investment committee.

CRITICAL — ANTI-HALLUCINATION RULES:
- Use ONLY the facts provided below.  Do NOT add any number, date, percentage, or
  name that does not appear in the extracts.
- If all extracts say "NOT FOUND" for a field, write "NOT FOUND" in your note
  rather than omitting it — an explicit gap is better than silent omission.
- If extracts disagree on a number (e.g. one says Rs 450 Cr, another says
  Rs 4,500 Mn), report the discrepancy rather than picking one: "Rs 450 Cr
  (alternatively stated as Rs 4,500 Mn)".
- If a section heading is listed but no facts were extracted for it, write
  "— No data in report" under that heading.
- Write "—" (em-dash) for any expected field where the report genuinely provides
  no information.  Never invent a plausible-sounding number to fill a gap.

Structure it as: Recommendation & target, Thesis, Financials, Key assumptions, Risks.
Keep it tight, under 500 words total.

EXTRACTED FACTS:
{text}
"""

# Fact extraction for the dashboard header — broker, rating, CMP, target price.
# These four fields drive the recommendation badge and the upside gauge, so a
# hallucinated number here is highly visible and directly misleading.
FACTS_PROMPT = """Extract four fields from this equity research report on {company}.

RULES:
- Use ONLY values explicitly stated in the text.  If a field is genuinely absent
  (searched the entire cover page, not mentioned anywhere), use the string
  "NOT FOUND" instead of null — this lets the validator distinguish "LLM output
  was empty" from "LLM searched and found nothing."
- Do NOT guess the broker from the file name, URL, or watermark.  If the broker
  name is not written out in the prose (e.g. "Motilal Oswal Research"), use
  "NOT FOUND".
- Do NOT estimate a target price from a chart axis or a percentage move.  If the
  exact target price is not printed as a number, use "NOT FOUND".

Fields:
- broker: the research house publishing the report.  "NOT FOUND" if absent.
- recommendation: exactly one of "Buy", "Hold", "Sell".  Map synonyms:
  Accumulate/Add/Outperform -> Buy; Neutral/Equal-weight -> Hold;
  Reduce/Underperform -> Sell.  "NOT FOUND" if unstated.
- current_price: CMP as a number, no currency symbol.  "NOT FOUND" if absent.
- target_price: target price as a number, no currency symbol.  "NOT FOUND" if absent.

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"broker": "...", "recommendation": "...", "current_price": ..., "target_price": ...}}

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
- UNITS ARE MANDATORY.  When citing a figure, ALWAYS include its unit — crore, lakh,
  million, billion, %, bps, ₹, $, times (x), etc.  If a table row is labelled
  "EBITDA (Rs Mn)" and the cell shows "5,248", the answer MUST say "Rs 5,248 million"
  or "Rs 524.8 crore" — NOT "5,248".  A number without a unit is an incomplete answer,
  even if the citation marker is present.  If the context includes a [UNITS: ...] note
  above a table, apply those units to every figure you cite from that table.

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

# ==================================================================== table enrichment
TABLE_ENRICH_PROMPT = """You are analyzing a table from an equity research report on {company}.

TABLE (markdown):
{markdown}

Return a JSON object with SIX fields.  Be specific — use the actual numbers,
dates, entity names, and units visible in the table.

1. "questions": 3-5 natural-language questions an equity analyst might ask that
   this exact table answers.  Write them the way an analyst types into a search
   box — concise, specific, varying the phrasing.  Include time periods and
   units in the questions themselves (e.g. "What was EBITDA in Rs Mn for FY27E?"
   not "What was EBITDA for FY27E?").

2. "summary": One sentence describing what this table shows.  State the entity,
   time period, direction (up/down/flat), magnitude WITH UNITS, and the most
   salient figure.  Example: "Thyrocare's quarterly revenue grew 12% QoQ to
   Rs 450cr in Q3 FY24 with EBITDA margins expanding 180 bps."

3. "key_metrics": A list of the most decision-relevant facts from this table.
   Each item MUST include the unit — write "FY27E EBITDA: Rs 5,248 Mn" not
   "FY27E EBITDA: 5,248".  If the unit appears in the column header or row
   label (e.g. "(Rs Mn)", "(%)", "(x)"), attach it to every value you cite.
   Pull totals, growth rates, margins, extremes (highest / lowest), and
   year-end figures.  Max 8 items.  Prefer exact numbers from the table — do
   not compute derived values unless the table shows them.

4. "unit_context": A single sentence declaring the units this table uses.  Scan
   the column headers and row labels for unit indicators — "(Rs Mn)", "(Rs Cr)",
   "(%)", "(x)", "($)", "lakhs", "millions", "billions", "bps", etc.  Be
   comprehensive: if the table has both monetary and percentage columns, mention
   both.  Example: "Monetary values in Rs millions (Rs Mn); percentages and
   margins in %."

5. "tags": 4-8 single-word or short-phrase topic tags (lowercase).  Choose tags
   that would help someone find this table by searching.  Include:
   - The type of data: "quarterly", "annual", "segment_breakdown", "peer_comparison",
     "p&l", "balance_sheet", "cash_flow", "ratio", "valuation_multiple"
   - The financial topics covered: "revenue", "margin", "profitability", "growth",
     "debt", "capex", "operations", "guidance", "dividend"
   - Any specific line items: "ebitda", "pat", "eps", "roe", " roce"
   Only include tags where the table actually contains those topics.  Do not tag
   "debt" if the table has no debt data.  Prefer the tag labels used in equity
   research — use "revenue" not "sales_figure", "margin" not "profitability_pct".

6. "importance": an integer from 1 (lowest) to 5 (highest) rating how
   decision-relevant this table is for an equity analyst.  Use these criteria:
   - 5: Contains target price, recommendation, valuation multiples, or summary
        financials (revenue, EBITDA, PAT) that directly support an investment call.
   - 4: Detailed segment breakdowns, quarterly trends, peer comparison tables,
        or forward estimates/guidance.
   - 3: Supporting data — cost breakdowns, ratio analysis, capex plans, or
        historical financials beyond the forecast period.
   - 2: Background data — industry stats, macro indicators, or diluted per-share
        metrics that are secondary to the main thesis.
   - 1: Administrative — page headers, index of tables, glossary entries,
        disclaimers, or formatting artifacts misidentified as tables.

Respond with ONLY a JSON object, no prose, no markdown fences.
{{"questions": [...], "summary": "...", "key_metrics": [...], "unit_context": "...", "tags": [...], "importance": 3}}
"""

# ==================================================================== image captioning
IMAGE_CAPTION_PROMPT = """You are analyzing an image from an equity research report on {company}.

Describe what you see in complete detail, as a single paragraph.  Include:

1. TYPE: What kind of visual is this?  (line chart, bar chart, pie chart, table
   screenshot, diagram, handwritten note, photo, logo, map, or other)

2. DATA: Every number, label, percentage, and unit visible.  Read row labels,
   column headers, axis titles, legend entries, and data point callouts.  If a
   number has a unit (Rs Mn, %, bps, x, $, etc.) INCLUDE THE UNIT.  "Revenue:
   Rs 4,500 Mn in FY26E" — not "Revenue: 4,500."

3. TREND: The key direction, comparison, or insight the visual conveys.  Is
   something growing, declining, accelerating, inflecting?  Which is the highest /
   lowest value shown?  What is the time period?

4. ANNOTATIONS: Any callouts, arrows, circled numbers, handwritten marks, or
   text boxes overlaid on the visual.  Broker reports often have analyst
   scribbles or margin notes — transcribe them.

5. SOURCE: If the image has a source line or footnote (e.g. "Source: Company,
   IWM Research"), note it.

Be specific and factual.  Do NOT guess values you cannot clearly read — say
"illegible" if text is too blurry.  Write in third person, referring to the
company by name.  Output as a single paragraph with no headings or formatting.
"""

# ==================================================================== OCR
# Transcription prompt — different from the captioning prompt above.  This asks
# the VLM to reproduce ALL visible text faithfully, preserving headings,
# paragraphs, table structure, and noting handwritten marks.  The output becomes
# the page text that flows into chunking, summarisation, and retrieval.

OCR_PROMPT = """Transcribe ALL visible text from this page of an equity research report on {company}.

Rules:
- Reproduce every word, number, heading, and paragraph exactly as it appears.
  Preserve the reading order (top to bottom, left to right).
- For tables: use markdown table format (| col | col |).  Preserve all column
  headers, row labels, and numeric values WITH their units (Rs Mn, %, bps, etc.).
  Even if the table is poorly printed or skewed, do your best to reconstruct it.
- For handwritten notes, annotations, or margin scribbles: transcribe them inside
  [HANDWRITTEN: ...] markers at the position where they appear on the page.
  Analyst scribbles often contain important corrections or observations — include
  them faithfully.
- For misprinted or blurry text: if you can reasonably infer the value from
  context (e.g. a smudged digit in a column of numbers), include it with a [?]
  marker: "Revenue: Rs 4[?]50 Mn".  If completely illegible, write [ILLEGIBLE].
- For charts and graphs: describe them briefly in [CHART: ...] markers — note
  the chart type, axis labels, visible data points, and the key trend shown.
- Output ONLY the transcribed text.  No preamble, no commentary, no markdown
  fences.  The output should look like the original page rendered as plain text.
"""
