# Recall evaluation harness

Measures how well cognee recall answers hand-written questions about a corpus,
so a change to ingestion or retrieval shows up as a number instead of three
cherry-picked answers.

For each `dataset x search_type x question` the harness asks a **running cognee
server** twice - once for the answer, once with `only_context` for the context
that produced it - then an LLM judge grades the answer against **gold facts**
written by hand from the source documents. Coverage, wrong claims, fabricated
claims and stance errors are rolled up per dataset and search type.

Nothing runs cognee in-process except the judge, so what you measure is the
deployment as an agent would see it.

## Files

| File | What it is |
| --- | --- |
| `eval_recall.py` | The CLI. Import-light: `--help` never imports cognee. |
| `recall_eval_lib.py` | The logic (question files, HTTP, judge, report). Unit tested. |
| `*_questions.json` | One question set per corpus. |
| `runs/<UTC timestamp>/` | Per-run output: `answers.jsonl`, `verdicts.jsonl`, `report.md`. Git-ignored. |

The judge prompts live with the other cognee prompts:
`cognee/infrastructure/llm/prompts/eval_judge_system.txt` and
`eval_judge_user.txt`.

Tests: `cognee/tests/unit/scripts/test_eval_recall.py` (no network, no LLM, no
`~/.cognee`).

## Question file schema

```json
{
  "corpus": "adams_family",
  "questions": [
    {
      "id": "adams-01",
      "category": "disputes",
      "question": "Which allegations in the complaint do the defendants deny?",
      "gold_facts": [
        {"fact": "Defendants deny the allegations of paragraph 12.", "source": "answer.pdf p.3"}
      ],
      "must_not_claim": ["the court has already ruled on the motion"],
      "notes": "Watch for allegation-treated-as-finding."
    }
  ]
}
```

- `id` - unique across every question file passed in one run.
- `category` - one of `summary`, `disputes`, `who_said_what`, `valuation`,
  `timeline`, `procedure`, `references`.
- `gold_facts` - at least one. Each needs a `fact` and a `source` locating it in
  the documents. These are the ground truth; the judge grades coverage against
  them and copies their text verbatim into its verdict.
- `must_not_claim` - optional traps: plausible claims the documents do **not**
  support. If one appears in the answer (case-insensitive, whitespace
  normalised) it is counted as a fabricated claim whether the judge noticed or
  not. This deterministic floor is why a question set is worth writing: the
  judge is itself an LLM, and a hand-written trap is a fact about the corpus
  that no grader can be talked out of.
- `notes` - optional grading hints, shown to the judge.

Validate before running anything:

```bash
python scripts/legal/eval/eval_recall.py --validate-only \
    --questions scripts/legal/eval/adams_questions.json
```

Exit code 0 means valid; 1 prints every problem in the file at once.

## Running

The server must already be up (the fork's launcher defaults to `:8011`) and the
datasets ingested.

```bash
python scripts/legal/eval/eval_recall.py \
    --questions scripts/legal/eval/adams_questions.json \
    --datasets adams_family_redevelopment,adams_family_legal \
    --search-types HYBRID_COMPLETION,GRAPH_COMPLETION,AUTO \
    --spot-check 0.2
```

Credentials come from `COGNEE_EVAL_USER` / `COGNEE_EVAL_PASSWORD` and default to
the local stack's `default_user@example.com` / `default_password`. Neither the
password nor the bearer token is printed or written to the run directory.

The judge calls go through `LLMGateway`, so it uses whatever `LLM_API_KEY`
(falling back to `OPENAI_API_KEY`) the other `scripts/legal/*.py` use.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--questions PATH` | - | Question file; repeat for several. Required. |
| `--datasets a,b` | - | Dataset names. Required for a run. |
| `--search-types ...` | `HYBRID_COMPLETION` | cognee `SearchType` names, plus `AUTO`. |
| `--top-k N` | `15` | `top_k` per search. |
| `--limit N` | all | Only the first N questions - use it for a smoke run. |
| `--base-url URL` | `http://127.0.0.1:8011` | The running server. |
| `--timeout SECONDS` | `180` | HTTP timeout; searches are slow. |
| `--out DIR` | `runs/<UTC timestamp>` | Run directory. |
| `--no-judge` | off | Answers only; no LLM calls, no verdicts. |
| `--judge-only RUN_DIR` | - | Re-grade a finished run's `answers.jsonl`; no search calls. |
| `--spot-check 0.2` | `0` | Print a seeded sample of verdicts for hand review. |
| `--seed N` | `0` | Seed for that sample. |
| `--validate-only` | off | Validate the question files and exit. |

`AUTO` is not a `SearchType`: it posts to `/api/v1/recall` with
`search_type: null` and lets the router choose. Everything else posts to
`/api/v1/search` with the type pinned.

A failed cell is recorded as a row with `error` and the run continues; transient
failures (timeout, connection, 5xx) get one retry first.

### Reading the output

`report.md` (also printed to stdout) is one row per dataset x search type:

| column | meaning |
| --- | --- |
| `n` | cells attempted |
| `mean coverage` | mean over graded questions of `covered / (covered + missed)` gold facts |
| `wrong claims` | total claims contradicting a gold fact or the retrieved context |
| `fabricated claims` | total specific claims supported by neither context nor gold facts (includes every triggered `must_not_claim`) |
| `stance errors` | total places the answer asserted as true something the documents negate, or the reverse |
| `errors` | cells that failed to answer or to grade |

Counts are totals, not rates - compare them at equal `n`.

`--spot-check 0.2` prints a seeded 20% of verdicts with question, gold facts,
answer and verdict. Read them. The judge is an LLM and a coverage number you
have never sanity-checked by hand is not evidence.

## Adding a corpus

1. Ingest it into one or more datasets on the running server.
2. Write `scripts/legal/eval/<corpus>_questions.json` to the schema above.
   Ten to twenty questions spread across the categories is enough to see a
   change; every gold fact must be checkable against a named source, and give
   the hard questions a `must_not_claim` trap.
3. `--validate-only` it.
4. Run with `--limit 2 --no-judge` first to confirm the datasets and search
   types answer at all, then the full judged run.
5. Keep the run directory out of git (it already is) and quote the report table
   in the PR instead.
