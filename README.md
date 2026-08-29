# peer think with confidence (peerconf) 

warmup-free confidence filtering for parallel llm reasoning.

peerconf is a method for making parallel test-time reasoning more compute-efficient by
using confidence estimates from the reasoning traces generated within the current run.

the central idea is simple: when we sample many reasoning traces in parallel, some
traces become sufficiently confident in their answer well before they exhaust their
token budget. at the same time, the traces that finish early provide useful information
about what level of confidence is typical among successful completions.

peerconf uses these two observations to decide which traces are worth continuing. unlike
approaches that require a separate warm-up phase to estimate a confidence threshold,
peerconf estimates this threshold online from traces that have already finished for the
current problem. confidence filtering and consensous checks can therefore begin as soon as the first trace
finishes.

## how peerconf works

for each problem, we launch multiple reasoning traces in parallel.

as traces finish, peerconf uses their confidence scores to construct an online
confidence threshold. active traces are periodically probed to determine whether they
have effectively committed to an answer.

a trace can terminate early when:

- its confidence crosses the current peerconf threshold, and
- the model has completed its reasoning and committed to an answer.

that answer then becomes the trace's vote.

when a trace leaves an active seat, peerconf can launch a replacement trace, subject to
the configured sampling budget, and that replacement is judged against the threshold as
it generates. this allows computation to be redirected away from traces
that have already reached a sufficiently confident conclusion or dropped below the
threshold and toward additional independent attempts.

consensous across the finished traces is checked every time a trace deaprts instead of after a seperate warmup phase, so a question
can end as soon as the model's traces agree on an answer.

the goal is to retain the benefits of parallel reasoning and confidence-weighted voting
while reducing unnecessary generation.

## comparison with deepconf

the repository contains two implementations:

```
peerconf/   peerconf
deepconf/   deepconf baseline
```

the deepconf implementation is based on the original method, with adaptive sampling
added because we did not find this component in the official release repository.

for deepconf, adaptive sampling acts as a consensus-based stopping rule. sampling for a
question terminates when the leading answer accounts for at least 95% of the total
confidence-weighted vote:

```
V(a*) / sum(V) >= 0.95
```

where `a*` is the current leading answer. each trace is weighted using its lowest window
confidence, following the confidence measure used by deepconf for online inference. we use the same consensus-based early stopping rule
as deepconf and only apply it when at least 3 traces have finished.

both run against a stock vllm server and compute confidence client-side, so server is set up the same way
for both methods. 

## what you need

a gpu and:

```
pip install vllm transformers requests numpy
pip install git+https://github.com/hao-ai-lab/Dynasor.git
```

## running it

two cells run in order. cell 1 starts the server and waits for it to come up while cell 2 does
the run.

```
python peerconf/cell1_start_server.py     # leave this running
python peerconf/cell2_run.py
```

same for `deepconf/`. cell 1 also shards across every gpu it finds, so set `TP` at the
top of that file to the number of gpus you want to use.

## settings

everything that can be changed is at the top of cell 2 in the control panel

- `DATASET`: which benchmark to run. aime25, math500, hmmt25 or gsm8k, all in
  `benchmarks/`
- `QIDS`: which questions from the benchmarks
- `SEATS` and `MAX_TRACES`: how many attempts run at once, and how many the run is
  allowed in total. 16 and 32 by default
- `REPLACEMENT_SEATS`: how many replacements can run at once. one starts every time a
  trace departs, and once there are none left to replace it keeps launching more traces
  until this many replacements are running.
- `BAR_KEEP_TOP`: how strict the cutoff is. 10 keeps only the most confident 10% of
  the traces' worst moments which we call peerconf-low. 90 keeps the most confident
  90% of the traces' worst moments which we call peerconf-high
- `WINDOW`: how many recent tokens the confidence score averages over. 2048 on aime25
  and hmmt25, 256 on math500 because its traces are much shorter
- `PROBE_EVERY`: how many tokens between completion probes. 4096 on aime25 and hmmt25,
  512 on math500. the probe uses deepseek and qwen's end of thinking marker `</think>`, so running gpt-oss means changing it
  to its own marker `<|end|>` in cell 2. set it to 0 to turn probes off entirely 

## what comes out

one pkl file for every question which is written to `OUT_DIR`. each file holds every trace's full
text, its confidence over time, all the probes, and the final vote with seven different voting methods that deepconf use.
for both methods the confidence-weighed vote uses the lowest group confidence (min_window_weighted) which is the same
voting method that deepconf online uses.

if a question's pkl is already in `OUT_DIR` it gets skipped so in order to redo a question you have to delete its pkl first.
