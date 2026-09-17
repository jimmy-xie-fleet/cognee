# Recall evaluation harness

Measures how well cognee recall answers hand-written questions about a corpus,
so a change to ingestion or retrieval shows up as a number instead of three
cherry-picked answers.

For each `dataset x search_type x question` the harness asks a **running cognee
server** twice - first with `only_context` for the retrieval context, then for
the answer - then an LLM judge grades the answer against **gold facts** written
by hand from the source documents. Coverage, wrong claims, fabricated claims and
stance errors are rolled up per dataset and search type.

Nothing runs cognee in-process except the judge, so what you measure is the
deployment as an agent would see it.

## Files

| File | What it is |
| --- | --- |
| `eval_recall.py` | The CLI. Import-light: `--help` never imports cognee. |
| `recall_eval_lib.py` | The logic (question files, HTTP, judge, report). Unit tested. |
| `*_questions.json` | One question set per corpus. |
| `runs/<UTC timestamp>/` | Per-run output: `run.json`, `answers.jsonl`, `verdicts.jsonl`, `report.md`, `attribution.jsonl` (with `--attribute` / `--analyze`), and the crash journals. A `--repeats` run holds `repeat-<i>/` children plus a pooled `report.md`. Git-ignored. |

The judge prompts live with the other cognee prompts:
`cognee/infrastructure/llm/prompts/eval_judge_system.txt` and
`eval_judge_user.txt`; the miss-attribution grader's are `eval_attribution_system.txt`
and `eval_attribution_user.txt` next to them.

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
      "must_not_claim": [
        {"claim": "There is a $9,000,000 appraisal of the property.", "match": ["$9,000,000"]},
        {"claim": "The court has already ruled on the motion.", "match": []}
      ],
      "notes": "Watch for allegation-treated-as-finding."
    }
  ]
}
```

- `id` - unique across every question file passed in one run.
- `category` - one of `summary`, `disputes`, `who_said_what`, `valuation`,
  `timeline`, `procedure`, `references`.
- `gold_facts` - at least one. Each needs a `fact` and a `source` locating it in
  the documents. These are the ground truth. **The judge is shown them numbered
  and answers with the numbers**, so it cannot shrink the denominator: coverage
  is always `covered / len(gold_facts)`, and a fact the judge classified neither
  way counts as missed. An index it invents, or a fact it puts on both lists, is
  dropped and counted in the report's `judge issues` column.
- `must_not_claim` - optional traps: plausible claims the documents do **not**
  support. Each is an object:
  - `claim` - the full sentence. The LLM judge reads this and stays the primary
    detector for the trap.
  - `match` - short literal spans, e.g. an amount, a date, a resolution number,
    or a distinctive phrase. The **span floor** adds the claim to
    `fabricated_claims` when one of these spans occurs in a sentence of the
    answer that carries no negation cue (`not`, `no`, `never`, `denies`,
    `denied`, `without`, `cannot`, `unsupported`). Leave `match` empty to make a
    trap judge-only.

  A bare string is still accepted and means "this claim, no spans".

  The floor is a **conservative** backstop, not a second grader. It catches the
  fabrication class a judge is most likely to wave through - an invented amount
  or number - and stays quiet everywhere else. It is quiet by design: matching
  the full trap sentence would never fire (no real answer contains it verbatim),
  and matching a span without the negation check would flag "the memo does not
  state that conflicts were identified" as a fabrication.

  Write spans that only a *wrong* answer can contain. A span that also appears
  in one of the corpus's own gold facts is the wrong span - the falsehood there
  is the attribution, not the literal, and the LLM judge should handle it.
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
| `--spot-check 0.2` | `0` | Print a seeded sample of verdicts for hand review (must be in [0, 1]). |
| `--judge-concurrency N` | `4` | Judge calls in flight at once; `1` grades one row at a time. Verdicts are independent, so this only changes wall-clock time. |
| `--seed N` | `0` | Seed for that sample. |
| `--label TEXT` | - | Free text for the manifest, e.g. the server's code version. |
| `--validate-only` | off | Validate the question files and exit. |
| `--repeats N` | `1` | Run the whole matrix N times into `RUN_DIR/repeat-<i>/` on fresh sessions and report the mean and spread per dataset x search type. See "Repeats". |
| `--attribute` | off | After judging, attribute every missed gold fact to retrieval or generation (one extra LLM call per cell with misses). See "Miss attribution". |
| `--analyze RUN_DIR` | - | Run miss attribution on a finished, judged run; no search calls. Writes `attribution.jsonl` and re-renders `report.md` in place. |
| `--per-question` | off | Add a per-question table (one line per graded cell) to the report. |

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
| `judge issues` | indices the judge invented or put on both lists; above zero means read those rows |
| `mean coverage` | mean over graded questions of `covered / all of the question's gold facts` - the judge answers with indices, so it cannot shrink the denominator |
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
| `pending` | the matrix row was written but the cell was never attempted (the run died) - `--resume` |
| `empty` | the server answered with no text or the "memory still warming up" marker - `--resume` later |
| `other` | read the `error` field in `answers.jsonl` |

`--spot-check 0.2` prints a seeded 20% of verdicts with question, gold facts,
answer and verdict. Read them. The judge is an LLM and a coverage number you
have never sanity-checked by hand is not evidence.

Under the main table the report always adds a `## By category` table: the same
columns per dataset x search type x question category, so a five-point move on
the main table can be traced to the disputes questions or the timeline
questions. `--per-question` adds the raw grid under it.

### Repeats

One run of a 19-question set graded by an LLM is one draw, and both the
answering model and the judge are nondeterministic. A delta between two
branches is a result only if it is larger than the spread between two runs of
the *same* branch. `--repeats 3` answers and judges the whole matrix three
times, each into `RUN_DIR/repeat-<i>/` with its own manifest, answers, verdicts
and report, on sessions namespaced per repeat so no two repeats share one. The
parent `RUN_DIR/report.md` then carries the pooled main table plus a
`## Repeats` table:

| column | meaning |
| --- | --- |
| `coverage mean` | mean of the per-repeat mean coverages |
| `stdev` | sample standard deviation of those means, in percentage points |
| `min` / `max` | the lowest and highest repeat |
| `wrong / run` etc. | the claim counts averaged per repeat, so they stay comparable to a single run's table |

Rule of thumb: a delta smaller than about two `stdev` is the noise floor, not a
change. `--repeats` needs a fresh, empty `--out`, and needs the judge (it
measures graded coverage), so it cannot be combined with `--no-judge`,
`--judge-only` or `--resume`.

### Miss attribution

Coverage says how many gold facts the answer missed; it does not say why. A
missed fact was either never retrieved (the context the answer was generated
from did not contain it - a **retrieval miss**, fixed in ingestion, lanes or
budgets) or was retrieved and then left out of the answer (a **generation
miss**, fixed in the prompt or the rendering). Those are different bugs.

`--attribute` on a run, or `--analyze RUN_DIR` on a finished one, asks a second
grader one narrow question per cell with misses: of these numbered missed
facts, which does the retrieved context contain? It answers with the facts'
gold indices (the same index protocol as the judge, so it cannot drop a fact),
and each fact becomes one row of `attribution.jsonl`:

| field | meaning |
| --- | --- |
| `fact_index`, `fact` | which gold fact, 1-based in the question's list |
| `in_context` | `true` = generation miss, `false` = retrieval miss, `null` = the grader did not classify it (see `notes`) |
| `literals`, `literals_in_context` | the amounts, dates, paragraph and docket numbers in the fact, and whether every one of them occurs in the context - a deterministic cross-check recorded beside the grader's answer, never in place of it |
| `error` | the grader call failed for this cell; the row is counted under `errors` |

A cell that retrieved no context at all is attributed without a grader call:
every miss is a retrieval miss by definition. The report gains a
`## Miss attribution` table with a total per dataset x search type followed by
the per-category breakdown. Its `literal disagreements` column counts facts
where the literal check and the grader point different ways; a handful is
normal (a context can paraphrase a date), a large number means read those rows
before trusting the split.

`--analyze` works on runs written before the index fields existed: the missed
fact texts are mapped back to the question file, which is why it needs
`--questions` too. It never opens an HTTP client.

## What a run directory contains

| File | What it is |
| --- | --- |
| `run.json` | The manifest, written **before** the first request. |
| `answers.jsonl` | One row per cell, in matrix order. |
| `verdicts.jsonl` | One row per graded cell. |
| `report.md` | The table also printed to stdout. |
| `answers.partial.jsonl`, `verdicts.partial.jsonl` | Crash journals, present only while a run is in flight. |
| `*.bak` | The previous copy, kept when `--resume` or `--judge-only` overwrites in place. |

### The manifest

`run.json` records what the numbers mean: the UTC start time, the base URL, the
**harness's own** `git rev-parse HEAD` (`harness_checkout_commit` - this process
cannot see the server's code, which is what `--label` is for), a sha256 per
question file, the datasets, search types, `top_k`, timeout, pause, and the two
protocol flags below. Credentials are not inputs and never appear in it.

A coverage number is not comparable to another one without this: gold files get
edited, and a report that does not name the gold it graded against is a number
with no denominator.

### The protocol flags

- `session_per_cell: true` - every cell sends its own `session_id`
  (`eval-<run dir>-<dataset>-<search type>-<question id>`) on both calls. Both
  endpoints fall back to the caller's *default* session when none is given, so
  without this question 19 is answered against the conversation questions 1-18
  left behind, and two runs of the same matrix are not comparable.
- `context_before_answer: true` - the `only_context` call goes first. Both calls
  record a QA turn on the session, so asking for the answer first means the
  context the judge grades was shaped by the answer itself.

### Crash safety

- The **full matrix** is written to `answers.jsonl` before the first request,
  every row marked `pending: not attempted`. A run killed in its first pass is
  therefore always resumable; the journal, not the matrix, says what is done.
- Every row is appended to its journal the moment it exists, flushed. `--resume`
  and `--judge-only` both fold the journals into the main files **before** doing
  anything else, and the journals are deleted only once the main files hold
  their rows.
- Main files are written to a temp file in the same directory and `os.replace`d,
  so a crash mid-write leaves the previous file rather than a truncated one.
- A journal's last line can be torn by the crash that ended the run; it is
  skipped with a warning. A torn line anywhere else is an error, because
  silently dropping a middle row would quietly shrink the matrix.

### Non-answers

An answer the server returns as empty, or as a system marker
(`memory_warming_up`, `build_failed`), is recorded as an `empty` **error row**,
not graded. Marker text reads like prose, so grading it would report a real zero
for a dataset that was simply not built yet.

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
