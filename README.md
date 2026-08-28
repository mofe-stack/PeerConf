# peerconf

warmup-free confidence filtering for parallel llm reasoning. the threshold comes from
the traces that finish inside the run itself instead of a separate warm-up phase, so
filtering starts at the first finisher. completion probes stop a trace once it has
committed to its answer, and that answer becomes its vote.

`peerconf/` is the method. `deepconf/` is the baseline, reimplemented from the paper
since adaptive sampling was not in the official release. that is the consensus stop,
which ends a question once the leading answer holds 95% of the confidence-weighted
vote, `V(a) / sum(V) >= 0.95`, where a trace's weight is its lowest window confidence.
both run against a stock vllm server and compute confidence client-side, so the
serving setup is identical across arms.

## what you need

a gpu, and:

```
pip install vllm transformers requests numpy
pip install git+https://github.com/hao-ai-lab/Dynasor.git
```

## running it

two cells, in order. cell 1 starts the server and waits for it to come up. cell 2 does
the run.

```
python peerconf/cell1_start_server.py     # leave this running
python peerconf/cell2_run.py
```

same for `deepconf/`. cell 1 shards across every gpu it finds, so set `TP` at the top
of that file if you want fewer.
## settings

everything you would want to change sits at the top of cell 2, in the block marked
control panel. the ones that matter most:

- `DATASET`: which benchmark to run. aime25, math500, hmmt25 or gsm8k, all read from
  `benchmarks/`
- `QIDS`: which questions from it
- `SEATS` and `MAX_TRACES`: how many attempts run at once, and how many the run is
  allowed in total. 16 and 32 by default
- `BAR_KEEP_TOP`: how strict the cutoff is. 10 keeps only the most confident tenth of
  finished attempts, which is what we call peerconf-low. 90 is far more forgiving
- `WINDOW`: how many recent tokens the confidence score averages over. 2048 on aime25
  and hmmt25, 256 on math500, whose answers are much shorter
- `PROBE_EVERY`: how many tokens between completion probes. 4096 on aime25 and hmmt25,
  512 on math500. set it to 0 to turn probes off entirely

## what comes out

one pickle file per question, written to `OUT_DIR`. each one holds every attempt's full
text, its confidence over time, any probes it fired, and the final vote worked out seven
different ways so you can compare them.

if a question's pickle already exists it gets skipped. so a sweep you had to kill halfway
picks up where it stopped when you rerun it. to redo a question, delete its pickle first.
