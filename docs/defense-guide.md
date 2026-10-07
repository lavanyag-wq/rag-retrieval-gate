# Defense guide: rag-retrieval-gate

What to say about this repository in an interview, with the command that
proves each claim. Every number here is guarded by `tools/check_numbers.py`,
so if this document and the code ever disagree, the build fails.

## The 30 second version

"RAG eval scores have two structural blind spots: an end-to-end answer score
cannot tell you whether the retriever or the generator failed, and a small
golden set cannot tell a real regression from noise. I built a retrieval gate
that isolates the retriever against labelled relevance judgments, compares
runs per query with an exact sign test, and, this is the part I care about,
refuses to rule when the set is too small to support any verdict. It returns
UNDERPOWERED as its own exit code. I also measured the rule most teams use
instead, a one-point mean threshold, and it fired on 84.5% of resampled
identical-system golden sets at size 8."

Then stop. The question that usually comes next is "what do you mean the set
is too small to know", and the answer is the resolution table: an exact sign
test at alpha 0.05 needs a minimum of 5 flips in one direction, so a 36-query
set cannot call anything that moves fewer queries than that, which is 13.9%
of the 36-query set.

## The claims, each with how it is proved

### 1. The gate produces all three verdicts from real runs

Command: `python experiments/exp_regression.py`

Numbers: the v2 upgrade moved mean recall by 4.2 points, with 4 queries lost
and 1 improved, at p = 0.1875: a real-looking drop the set cannot confirm, so
the verdict is PASS. The dim=64 downsize has a mean delta of 6.9 points with
all 5 moved queries worse, p = 0.03125 on the exact sign test: REGRESSION.

If pushed ("so the upgrade was fine?"): no, and that is the point. All 4
worse queries are acronym-style, which is a concentrated loss the mean
understates and an end-to-end score would bury entirely. The gate surfaces
the per-style transition table; the honest conclusion is "grow the set", not
"ship it".

### 2. The resolution arithmetic is exact and published

Command: `rag-gate resolution --queries 36 --target-rate 0.1`

Numbers: at alpha 0.05, at least 5 queries must move in one direction before
any verdict is possible. Detecting a regression that hits 10% of queries with
80% power needs 66 queries; at 5% it would take 134 queries to detect.

If pushed ("where do these come from?"): direct binomial summation, about ten
lines in `stats.py`, no scipy. 0.5 to the fifth power is 0.03125, which is
the entire significance machinery at its floor.

### 3. The naive rule most teams use is noise at small sizes

Command: `python experiments/exp_noise_floor.py`

Numbers: an identical system compared across resampled golden subsets, and
the one-point mean rule fired on 84.5% of trials at size 8 and fires 100% of
the time at size 18. The gate under its sign-flip null fired 2.5% of the time
against a nominal 5%.

If pushed ("why does the naive rule get worse as the set grows?"): recall on
a two-relevant-document corpus is quantized to halves, so subset means move
in steps of about 2.8 points at size 18; any difference at all clears a one
point threshold. The naive rule's threshold sits below the metric's own
quantization, which is a category of mistake worth naming in a design review.

### 4. The numbers cannot silently go stale

Command: `make verify` (or break a README figure and watch it fail)

`tools/collect_metrics.py` reads the junit XML, the coverage JSON and the
experiment JSON; nothing is typed by hand except one labelled historical
value. `tools/check_numbers.py` then matches anchor phrases, not bare digits,
across this document, the README and ADR-004. The suite is 136 tests at 98%
line coverage, and CI re-runs the experiments from scratch.

If pushed ("one hand-set value?"): the 78.5% false positive reading from the
pre-fix noise-floor experiment. It is recorded history, not a measurement of
current code, and it is pinned so the war story cannot drift. ADR-004 is the
writeup.

## Questions that are meant to be hard

**Isn't this just a wrapper around a stats formula?** The formula is ten
lines. The engineering is everything around it: the paired run format that
remembers ranked lists so losses are attributable to documents, the three-way
verdict with exit codes CI can act on, the tie-break that makes rankings a
total order, the receipts pipeline, and the experiment that caught its own
broken null. A wrapper does not find its own bugs.

**The corpus is synthetic. Does any of this transfer?** The absolute recall
levels do not transfer, and ADR-001 says so in bold terms. What transfers is
everything the tool claims: calibration, resolution, attribution, and the
false positive behaviour of mean thresholds, which are properties of the
comparison machinery. Point the run pipeline at a real labelled set and the
same gate applies unchanged.

**What is the weakest part?** Binary relevance with exactly two relevant
documents per query. Real relevance is graded and ragged, and nDCG with
binary labels on a two-document relevant set is close to redundant with
recall here. Second weakest: the sign test ignores magnitudes, which is
defended in ADR-002 but is a real power cost on sets with meaningful deltas.

**What would you do differently with more time?** Graded relevance in the
corpus generator, a Wilcoxon option with a seeded, documented power
simulation beside the closed-form sign test, and a mode that ingests a real
system's run files (the JSON format already supports this; the generator
does not produce graded labels to exercise it).

**Why didn't you use a real embedding model?** Determinism buys the receipts
pipeline, and the gate is the subject, not the embedder. ADR-001. The honest
cost: absolute levels are synthetic, so the repository never quotes them as
findings.

**Why is UNDERPOWERED not just a warning string?** Because pipelines act on
exit codes, not log lines. A warning string inside a green build is invisible
by design review number three. ADR-005.

**Your false positive rate for the gate is 2.5%, not 5%. Is that a bug?** No:
discreteness. With 5 moved queries the only significant outcome under the
null has probability 0.03125, so the exact test is conservative by
construction. The measured 2.5% over 2000 sign-flip trials is consistent
with that.

## Things to say and things not to say

Say: "the set cannot support a verdict below its resolution floor", "all
losses concentrated in acronym-style queries", "measured 2.5% against a
nominal 5% because the exact test is conservative", "absolute recall here is
synthetic and I never quote it as a finding".

Do not say: "the gate proves there is no regression" (it finds no evidence at
the set's resolution, which is weaker and true). Do not say "zero false
positives". Do not quote mean recall 0.819 as if it described a real system.
Do not call the sign test "the most powerful test" (it is the most
defensible one here, which is a different claim, and ADR-002 prices it).

## The live demo script

```bash
rag-gate generate --seed 7 --out /tmp/corpus.json
rag-gate run --corpus /tmp/corpus.json --embedder v1 --out /tmp/base.json
rag-gate run --corpus /tmp/corpus.json --embedder v2 --out /tmp/cand.json
rag-gate gate --baseline /tmp/base.json --candidate /tmp/cand.json ; echo "exit $?"
rag-gate resolution --queries 36 --target-rate 0.1
```

The fourth command prints the per-style transition table and exits 0 with the
nearly-significant PASS; swap the candidate for `--dim 64` to watch exit 1,
or regenerate with `--topics 4` to watch exit 2.
