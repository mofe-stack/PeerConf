# peerconf

warmup-free confidence filtering for parallel llm reasoning. the threshold comes from
the traces that finish inside the run itself instead of a separate warm-up phase, so
filtering starts at the first finisher. completion probes stop a trace once it has
committed to its answer, and that answer becomes its vote.

`peerconf/` is the method. `deepconf/` is the baseline, reimplemented from the paper
since adaptive sampling was not in the official release. both run against a stock
vllm server and compute confidence client-side, so the serving setup is identical
across arms.

## what you need

a gpu, and:

```
pip install vllm transformers requests numpy
pip install git+https://github.com/hao-ai-lab/Dynasor.git
```

dynasor is only there for `math_equal`, which decides whether two answers are the same.
both arms use it, so they grade identically.

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

everything lives at the top of cell 2 under the control panel. the ones worth knowing:

- `DATASET` — aime25, math500, hmmt25 or gsm8k, read from `benchmarks/`
- `QIDS` — which questions to run
- `SEATS` and `MAX_TRACES` — 16 traces at once, 32 launched in total
- `BAR_KEEP_TOP` — 10 is peerconf-low, 90 is peerconf-high
- `WINDOW` — 2048 on aime25 and hmmt25, 256 on math500
- `PROBE_EVERY` — tokens between completion probes, 0 turns them off

## output

one pickle per question in `OUT_DIR`, holding each trace's text, its confidence over
time, the probes it fired, and the vote under seven different methods. a question whose
pickle already exists is skipped, so a sweep that gets killed picks up where it stopped.
delete a pickle to redo that question.
