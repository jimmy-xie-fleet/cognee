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

The judge calls go through `LLMGateway`, so it uses whatever `LLM_API_KEY` the
server-side configuration uses; `OPENAI_API_KEY` is mapped onto it when set. The
harness never exports an *empty* `LLM_API_KEY` - an empty environment variable
beats the value in `.env` and makes cognee raise `LLMAPIKeyNotSetError`, which is
how the first live run answered 77 questions and graded none of them. If every
verdict comes back as `judge: LLMAPIKeyNotSetError`, the key is missing from the
environment and from `.env`; fix it and `--resume` the run.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--questions PATH` | - | Question file; repeat for several. Required. |
| `--datasets a,b` | - | Dataset names. Required for a run. |
| `--search-types ...` | `HYBRID_COMPLETION` | cognee `SearchType` names, plus `AUTO`. |
| `--top-k N` | `15` | `top_k` per search. |
| `--limit N` | all | Only the first N questions - use it for a smoke run. |
| `--base-url URL` | `http://127.0.0.1:8011` | The running server. |
| `--timeout SECONDS` | `600` | HTTP timeout. See "Long runs" below. |
| `--pause-seconds FLOAT` | `0` | Sleep between requests, to keep load off a shared server. |
| `--out DIR` | `runs/<UTC timestamp>` | Run directory. |
| `--no-judge` | off | Answers only; no LLM calls, no verdicts. |
| `--judge-only RUN_DIR` | - | Re-grade a finished run's `answers.jsonl`; no search calls. |
| `--resume RUN_DIR` | - | Re-run only that run's failed rows; re-judge only what changed. |
| `--spot-check 0.2` | `0` | Print a seeded sample of verdicts for hand review. |
| `--seed N` | `0` | Seed for that sample. |
| `--validate-only` | off | Validate the question files and exit. |

`AUTO` is not a `SearchType`: it posts to `/api/v1/recall` with
`search_type: null` and lets the router choose. Everything else posts to
`/api/v1/search` with the type pinned.

A failed cell is recorded as a row with `error` and the run continues.

## Long runs on a shared server

The first live run lost 94 of 171 rows, and the three ways it failed are the
three things the harness now handles:

- **The token expired** about an hour in and every row after that was a 401. A
  401 is now treated as "log in again and replay this request once"; only a
  second 401 with a fresh token is a real failure. The run reports how many times
  it had to re-authenticate.
- **The server timed out** on `HYBRID_COMPLETION` over a 5k-node graph. `--timeout`
  now defaults to **600 s**. Efficiency is explicitly not a goal of this
  evaluation: a slow answer is data, a timed-out answer is a hole in the table.
- **The server returned 500s** from its session-cache connection pool while the UI
  and the memory plugin were also hitting it. Transient failures (timeouts,
  connection errors, 5xx) now get up to **three attempts** with exponential
  jittered backoff (about 2 s then 8 s), and `--pause-seconds` slows the whole
  harness down so a long run does not saturate a server other people are using.
  `--pause-seconds 2` is a reasonable default when someone else is on the box.

Everything else (4xx other than 401) is still permanent and fails the row
immediately.

### Resuming a partial run

```bash
python scripts/legal/eval/eval_recall.py \
    --questions scripts/legal/eval/adams_questions.json \
    --resume scripts/legal/eval/runs/20260916T101500Z \
    --pause-seconds 2
```

`--resume` re-runs **only** the rows carrying an error, at the same
dataset / search type / question / `top_k`, merges them back in place, and then
judges only what changed: the rows it just re-answered, plus any row that was
never graded or whose verdict is itself an error (a `judge:` error graded
nothing, so it is a hole, not a grade). Successful rows and their verdicts are
never re-run. `answers.jsonl` and `verdicts.jsonl` are copied to `.bak` before
being overwritten; pass `--out` to write the merged run somewhere else instead.

### Reading the output

`report.md` (also printed to stdout) is one row per dataset x search type:

| column | meaning |
| --- | --- |
| `n` | cells attempted |
| `answered` | cells that came back without an error - check this first |
| `mean coverage` | mean over graded questions of `covered / (covered + missed)` gold facts |
| `wrong claims` | total claims contradicting a gold fact or the retrieved context |
| `fabricated claims` | total specific claims supported by neither context nor gold facts (includes every triggered `must_not_claim`) |
| `stance errors` | total places the answer asserted as true something the documents negate, or the reverse |
| `errors` | cells that failed to answer or to grade |

Counts are totals, not rates - compare them at equal `n`.

Read `answered` before anything else. A run where every cell failed and a run
with no gold facts both render coverage as `-`, and the first live run's report
looked plausible while not a single row had succeeded.

Under the table is an error histogram by class:

| class | what to do |
| --- | --- |
| `401` | the token expired mid-run - `--resume` the run |
| `timeout` | raise `--timeout`, or `--pause-seconds` to reduce load |
| `5xx` | the server was saturated - `--pause-seconds`, then `--resume` |
| `other` | read the `error` field in `answers.jsonl` |

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
