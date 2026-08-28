# peerconf

## the problem

one of the most reliable ways to make a language model better at hard maths is to ask it
the same question many times and take whichever answer comes up most often. it works
well, and it is expensive. every attempt runs all the way to the end, so an attempt that
made a mistake in its first paragraph costs exactly as much as one that solves the
problem.

deepconf's idea was to watch each attempt while it writes. the model gives a confidence
score for every token it produces, and an attempt that starts going wrong tends to get
less confident. so you set a cutoff, and any attempt whose confidence drops below it gets
stopped early.

the catch is where the cutoff comes from. deepconf runs 16 attempts all the way to the
end first, looks at how confident those were, and puts the cutoff at a percentile of
them. that first batch is called the warm-up. nothing can be stopped while it is running,
so on a 32-attempt budget you have already paid for half the run before the method does
anything at all.

## what peerconf does instead

there is no warm-up. the cutoff comes from the attempts that finish during the run
itself. as soon as the first one finishes you have a cutoff, and every attempt that
finishes after that updates it. so filtering starts almost immediately, and the cutoff
keeps improving as the run goes on rather than being fixed once at the start.

the second idea is completion probes. models often work out the answer and then keep
going for thousands more tokens, checking themselves and restating things. every few
thousand tokens we take a copy of what an attempt has written so far, add a short line
asking it to give its final answer, and let it write about twenty more tokens. if it
answers confidently and closes off its own reasoning while doing so, we take that answer
as the attempt's vote and stop it there.

the "closes off its own reasoning" part matters. without it the model will happily give
a confident answer while it is still halfway through working, and that answer is often
wrong.

## what is in here

`peerconf/` is the method. `deepconf/` is the baseline. we reimplemented deepconf from
its paper because the part that ends a run early, which they call adaptive sampling, was
not in their released code. that is the rule that stops a question once the leading
answer holds 95% of the confidence-weighted vote.

both of them talk to an ordinary vllm server and work out confidence on the client side.
the server is set up identically for both, so the method being tested is the only thing
that differs between them.

each folder has two files. cell 1 starts the server. cell 2 does the run.

## what you need

a gpu, and:

```
pip install vllm transformers requests numpy
pip install git+https://github.com/hao-ai-lab/Dynasor.git
```

dynasor gives us `math_equal`, which decides whether two answers are the same thing. this
matters more than it sounds. one attempt might write `\dfrac{1}{2}` where another writes
`\frac{1}{2}`, and if you compare those as plain text you count one answer as two and
split the vote between them. the method and the baseline use the same comparison, so they
are graded the same way.

## running it

two steps, in order.

```
python peerconf/cell1_start_server.py     # leave this running
python peerconf/cell2_run.py
```

cell 1 loads the model and waits until the server answers. the first time this takes a
few minutes because it has to download the weights. leave it running and start cell 2 in
another terminal. the same two steps work for `deepconf/`.

cell 1 splits the model across every gpu it can find. if you want it to use fewer, set
`TP` at the top of that file to the number you want.

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
