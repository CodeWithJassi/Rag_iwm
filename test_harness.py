#!/usr/bin/env python3
"""
Automated RAG evaluation harness.

Reads ``IWM_chatbot-Testing.csv``, runs every question through the pipeline
for each (LLM, embedding-model) combination.  An LLM judge compares the
generated answer against the ground-truth expected answer and rates it
Good / Okay / Bad — no human input needed.

Two-phase workflow
──────────────────
  Phase 1 — Find the best embedding model
      Pick one LLM (default: llama3.1:8b).  Test it against every embedding
      model.  The harness re-ingests the report between embedding switches.

  Phase 2 — Find the best LLM
      Lock the best embedding model from Phase 1.  Test every LLM against it.

Output
──────
  Results are written to ``test_results_<timestamp>.csv`` with columns for
  each (LLM × embed) combination, containing the generated answer and rating.

Usage
─────
  python test_harness.py --report-id 20

  # Phase 1 only:
  python test_harness.py --report-id 20 --phase1-only

  # Use a different base LLM for Phase 1:
  python test_harness.py --report-id 20 --base-llm gemma4:26b

  # Use a different LLM for judging (default: same as base-llm):
  python test_harness.py --report-id 20 --judge-llm gemma4:26b
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "IWM_chatbot-Testing.csv"

# ── models to sweep ──────────────────────────────────────────────────
LLMS = [
    "llama3.1:8b",
    "gemma4:26b",
    "MichelRosselli/GLM-4.5-Air:Q4_K_M",
]

EMBED_MODELS = [
    "nomic-embed-text:latest",
    "mxbai-embed-large:latest",
    "qwen3-embedding:4b",
]

# ── helpers ───────────────────────────────────────────────────────────

def _env_path() -> Path:
    return ROOT / ".env"


def _set_env(key: str, value: str) -> None:
    ep = _env_path()
    lines = ep.read_text().splitlines() if ep.exists() else []
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}=") or line.startswith(f"# {key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    ep.write_text("\n".join(lines) + "\n")


def _reingest(report_id: int) -> None:
    """Re-embed *report_id* with the current EMBED_MODEL (clean subprocess)."""
    print(f"   Re-ingesting report {report_id} …")
    p = subprocess.run(
        [sys.executable, str(ROOT / "backend" / "_reingest.py"), str(report_id)],
        capture_output=True, text=True, timeout=600, cwd=str(ROOT),
    )
    # Always print the subprocess output so the user sees progress
    for line in p.stdout.strip().splitlines():
        print(f"   {line}")
    if p.returncode != 0:
        print(f"   ❌ Re-ingestion failed:\n{p.stderr}")
        sys.exit(1)


def _read_questions() -> list[dict]:
    """Parse the CSV into a list of {question, expected, answerable} dicts."""
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    questions = []
    for row in rows[2:]:  # skip header rows
        q = (row[0] or "").strip()
        expected = (row[1] or "").strip()
        if not q:
            continue
        answerable = not expected.startswith("I don't have enough")
        questions.append({
            "question": q,
            "expected": expected,
            "answerable": answerable,
        })
    return questions


# ── judge ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a RAG system's answer quality. Compare the GENERATED answer against the EXPECTED (ground-truth) answer and rate it.

QUESTION:
{question}

EXPECTED ANSWER (ground truth from the broker report):
{expected}

GENERATED ANSWER (what the RAG system produced):
{generated}

Scoring rules:
- "Good"   — the generated answer captures the key facts from the expected
             answer. Minor wording differences are fine. Numbers must match
             exactly or be within reasonable rounding. If the expected answer
             says the system should abstain, and it did abstain, that is Good.
- "Okay"   — the generated answer is partially correct but misses some
             important facts, or includes minor inaccuracies alongside correct
             information. Directionally right but not fully reliable.
- "Bad"    — the generated answer is factually wrong, hallucinates numbers
             not in the expected answer, contradicts the expected answer,
             answers when it should have abstained, or abstains when the
             expected answer shows the information was available.

Respond with ONLY one word: Good, Okay, or Bad."""


def _llm_rate(question: str, expected: str, generated: str,
              judge_llm: str) -> str:
    """Use an LLM to compare generated vs expected and return Good/Okay/Bad."""
    sys.path.insert(0, str(ROOT / "backend"))
    from llm import llm_complete

    prompt = JUDGE_PROMPT.format(
        question=question,
        expected=expected[:800],
        generated=generated[:600],
    )
    out = llm_complete(prompt, max_tokens=20, label="judge", model=judge_llm)
    if not out:
        return "Error"

    out = out.strip().lower()
    if "good" in out:
        return "Good"
    if "okay" in out:
        return "Okay"
    if "bad" in out:
        return "Bad"
    return "Error"


def _rate(question: str, expected: str, answerable: bool,
          generated: str, abstained: bool, judge_llm: str) -> str:
    """Rate a single Q&A pair.  Auto-rates clear-cut cases; delegates
    ambiguous ones to the LLM judge."""
    # Correct abstention on unanswerable → Good
    if not answerable and abstained:
        return "Good"
    # Answered an unanswerable question → Bad (hallucination)
    if not answerable and not abstained:
        return "Bad"
    # Failed to answer an answerable question → Bad
    if answerable and abstained:
        return "Bad"
    # Answerable + got an answer → LLM judge
    return _llm_rate(question, expected, generated, judge_llm)


# ── runner ─────────────────────────────────────────────────────────────

def _sweep(llms: list[str], embeds: list[str], questions: list[dict],
           report_id: int, judge_llm: str, label: str) -> dict:
    """Run all (llm × embed) combinations.  Returns nested dict of results."""
    results: dict = {}  # key: (llm, embed) → list of (answer, rating)

    for emb in embeds:
        print(f"\n{'█'*70}")
        print(f"█  EMBEDDING: {emb}")
        print(f"{'█'*70}")
        _set_env("EMBED_MODEL", emb)
        _reingest(report_id)
        time.sleep(2)

        for llm in llms:
            key = (llm, emb)
            results[key] = []
            print(f"\n  ┌─ LLM: {llm}")

            for i, q in enumerate(questions):
                print(f"  │ [{i+1}/{len(questions)}] "
                      f"{q['question'][:90]}…", end=" ", flush=True)
                try:
                    res = _run_query(q["question"], report_id, llm)
                    generated = res["answer"]
                    abstained = res.get("abstained", False)
                except Exception as e:
                    print(f"❌ {e}")
                    results[key].append(("ERROR: " + str(e), "Error"))
                    continue

                rating = _rate(q["question"], q["expected"],
                               q["answerable"], generated, abstained,
                               judge_llm)
                results[key].append((generated, rating))
                print(f"→ {rating}")

            # Running tally
            goods = sum(1 for _, r in results[key] if r == "Good")
            okays = sum(1 for _, r in results[key] if r == "Okay")
            bads  = sum(1 for _, r in results[key] if r == "Bad")
            errs  = sum(1 for _, r in results[key] if r == "Error")
            total = len(results[key])
            pts = goods * 2 + okays * 1
            max_pts = (total - errs) * 2
            pct = round(pts / max_pts * 100) if max_pts > 0 else 0
            print(f"  └─ {llm} × {emb}: "
                  f"G={goods} O={okays} B={bads} E={errs} → {pct}%")

    return results


# ── query (single DB session) ─────────────────────────────────────────

# Module-level cache — db is initialised once by main().
_glob = {"report_cache": {}}


def _run_query(question: str, report_id: int, model: str) -> dict:
    """Run one question through the RAG pipeline.  DB pool must be open."""
    sys.path.insert(0, str(ROOT / "backend"))
    import retrieval
    from db import query

    if report_id not in _glob["report_cache"]:
        r = query(
            "SELECT id, company, broker, file_name, status "
            "FROM reports WHERE id=%s", (report_id,), one=True,
        )
        if not r:
            raise RuntimeError(f"Report {report_id} not found")
        if r["status"] != "ready":
            raise RuntimeError(f"Report status is '{r['status']}'")
        _glob["report_cache"][report_id] = r

    return retrieval.answer(
        _glob["report_cache"][report_id], question, [],
        deep_search=False, model=model, verbose=False,
    )


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--report-id", type=int, required=True,
                        help="Report ID in the database to test against")
    parser.add_argument("--base-llm", default="llama3.1:8b",
                        help="LLM for Phase 1 (default: llama3.1:8b)")
    parser.add_argument("--judge-llm", default="",
                        help="LLM for judging (default: same as base-llm)")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run only Phase 1, skip Phase 2")
    args = parser.parse_args()

    judge_llm = args.judge_llm or args.base_llm

    questions = _read_questions()
    if not questions:
        print("No questions found in CSV — aborting.")
        sys.exit(1)
    print(f"Loaded {len(questions)} questions "
          f"({sum(1 for q in questions if q['answerable'])} answerable, "
          f"{sum(1 for q in questions if not q['answerable'])} unanswerable)")
    print(f"Judge LLM: {judge_llm}")

    # ── Init DB once ──────────────────────────────────────────────────
    sys.path.insert(0, str(ROOT / "backend"))
    from db import init_db, close_db
    init_db()
    try:
        # ── Phase 1: fix LLM, sweep embedding models ──────────────────
        print(f"\n{'═'*70}")
        print(f"PHASE 1 — Find best embedding model")
        print(f"  Base LLM: {args.base_llm}")
        print(f"  Embedding models: {', '.join(EMBED_MODELS)}")
        print(f"{'═'*70}")

        p1 = _sweep([args.base_llm], EMBED_MODELS, questions,
                    args.report_id, judge_llm, "Phase 1")

        # Pick the best embedding model
        best_embed = None
        best_score = -1
        for (llm, emb), vals in p1.items():
            goods = sum(1 for _, r in vals if r == "Good")
            okays = sum(1 for _, r in vals if r == "Okay")
            bads  = sum(1 for _, r in vals if r == "Bad")
            errs  = sum(1 for _, r in vals if r == "Error")
            total = len(vals)
            pts = goods * 2 + okays * 1
            max_pts = (total - errs) * 2
            pct = round(pts / max_pts * 100) if max_pts > 0 else 0
            print(f"\n  {emb}: G={goods} O={okays} B={bads} → {pct}%")
            if pct > best_score:
                best_score = pct
                best_embed = emb

        print(f"\n✅ Best embedding model: {best_embed} ({best_score}%)")

        all_results = dict(p1)

        if not args.phase1_only:
            # ── Phase 2: best embed, sweep LLMs ───────────────────────
            print(f"\n{'═'*70}")
            print(f"PHASE 2 — Find best LLM")
            print(f"  Embedding: {best_embed}")
            print(f"  LLMs: {', '.join(LLMS)}")
            print(f"{'═'*70}")

            p2 = _sweep(LLMS, [best_embed], questions,
                       args.report_id, judge_llm, "Phase 2")
            all_results.update(p2)

        # ── Summary ───────────────────────────────────────────────────
        print(f"\n{'═'*70}")
        print(f"FINAL RESULTS")
        print(f"{'═'*70}")
        print(f"{'LLM':<35} {'Embed':<30} {'G':>4} {'O':>4} {'B':>4} {'E':>4} {'Score':>7}")
        print(f"{'-'*35} {'-'*30} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*7}")
        for (llm, emb), vals in sorted(all_results.items()):
            goods = sum(1 for _, r in vals if r == "Good")
            okays = sum(1 for _, r in vals if r == "Okay")
            bads  = sum(1 for _, r in vals if r == "Bad")
            errs  = sum(1 for _, r in vals if r == "Error")
            total = len(vals)
            pts = goods * 2 + okays * 1
            max_pts = (total - errs) * 2
            pct = round(pts / max_pts * 100) if max_pts > 0 else 0
            print(f"{llm:<35} {emb:<30} {goods:>4} {okays:>4} {bads:>4} {errs:>4} {pct:>6}%")

        _write_csv(all_results, questions, args.report_id)

    finally:
        close_db()


def _write_csv(results: dict, questions: list[dict], report_id: int) -> None:
    """Write results to a timestamped CSV with generated answers AND ratings."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / f"test_results_{stamp}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        combos = sorted(results.keys())  # (llm, embed)

        # Header: Question | Expected | for each combo: Generated | Rating
        header = ["Question", "Expected"]
        for llm, emb in combos:
            header.append(f"{llm} × {emb} — Answer")
            header.append(f"{llm} × {emb} — Rating")
        w.writerow(header)

        for i, q in enumerate(questions):
            row = [q["question"], q["expected"]]
            for llm, emb in combos:
                vals = results.get((llm, emb), [])
                if i < len(vals):
                    ans, rating = vals[i]
                    row.append(ans)
                    row.append(rating)
                else:
                    row.append("")
                    row.append("")
            w.writerow(row)

    print(f"\n📄 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
