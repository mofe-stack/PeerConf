# peerconf

warmup-free confidence filtering for parallel llm reasoning. the threshold comes from
the traces that finish inside the run itself instead of a separate warm-up phase, so
filtering and agreement checks start at the first finisher. completion probes stop a trace once it has
committed to its answer, and that answer becomes its vote.

`peerconf/` is the method. `deepconf/` is the baseline which is reimplemented from the paper
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

two cells, in order. cell 1 starts the server and waits for it to come up while cell 2 does
the run.

```
python peerconf/cell1_start_server.py     # leave this running
python peerconf/cell2_run.py
```

same for `deepconf/`. cell also 1 shards across every gpu it finds, so set `TP` at the top
of that file if you want to use fewer gpus.
## settings

everything you would want to change sits at the top of cell 2 in the control panel

- `DATASET`: which benchmark to run. aime25, math500, hmmt25 or gsm8k, all read from
  `benchmarks/`
- `QIDS`: which questions from it
- `SEATS` and `MAX_TRACES`: how many attempts run at once, and how many the run is
  allowed in total. 16 and 32 by default
- `REPLACEMENT_SEATS`: how many replacements can run at once. one starts each time an
  attempt leaves, and once the opening batch has all gone it fills up to this number
- `BAR_KEEP_TOP`: how strict the cutoff is. 10 keeps only the most confident tenth of
  finished attempts the traces worst moments which is what we call peerconf-low. 90 keeps the most confident
  90% of the traces worst moments which we call peerconf-high 
- `WINDOW`: how many recent tokens the confidence score averages over. 2048 on aime25
  and hmmt25, 256 on math500, whose answers are much shorter
- `PROBE_EVERY`: how many tokens between completion probes. 4096 on aime25 and hmmt25,
  512 on math500. set it to 0 to turn probes off entirely

## what comes out

one pickle file per question which is written to `OUT_DIR`. each one holds every attempt's full
text, its confidence over time, any probes it fired, and the final vote with seven different voting methods that deepconf use.
we use the lowest group confidence (min_window_weighted) which is the same that deepconf online uses.

if a question's pickle is already in `OUT_DIR` it gets skipped so in order to redo a question you have to delete its pickle first.
