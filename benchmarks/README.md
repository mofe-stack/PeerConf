# Benchmarks

One jsonl per benchmark, one question per line, always the same two fields:

```json
{"question": "Find the sum of all integer bases $b>9$ ...", "answer": "70"}
```

| file | questions | source |
|------|-----------|--------|
| `aime25.jsonl` | 30 | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) — `default/test` |
| `hmmt25.jsonl` | 30 | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) — `default/train` |
| `math500.jsonl` | 500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) — `default/test` |
| `gsm8k.jsonl` | 1319 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) — `main/test` |

GSM8K is the only one of the four that ships a train split as well; we use its
test split (1319 questions), the held-out set results are normally reported on.

Answers are stored the way the model is expected to box them. GSM8K's are
reduced to the number after `####`, dropping the worked solution. MATH-500 and
HMMT keep the dataset's own answer string, since `math_equal` does the grading
and handles LaTeX.

## Switching benchmark

Both run scripts read the `DATASET` env var and default to `aime25`:

```bash
DATASET=math500 python code/current/cell2_run.py        # PeerConf
DATASET=gsm8k   python deepconf/cell2_deepconf.py       # DeepConf
```

On Modal it is a flag:

```bash
modal run code/current/modal_run.py --qids "0,1,2" --dataset hmmt25
modal run code/current/modal_peerconf.py --qids "0,1" --dataset math500 --arms deepconf,peerconf
```

Adding a fifth benchmark means dropping a `<name>.jsonl` here in the same schema
and passing `DATASET=<name>` — no code change.

## Which questions run

| where | syntax |
|-------|--------|
| `cell2_run.py`, `cell2_deepconf.py` | `QIDS = range(30)` for a span, `QIDS = [6, 9]` for a set |
| `cell2-stopped.py` | `QID = 6`, one question per run |
| Modal `--qids` | `"6"`, `"6,9"`, `"0-29"`, or `"0-9,17,25-27"` — ranges include both ends |

Asking for a question the benchmark doesn't have stops the run outright —
nothing generates, and the error names the valid range:

```
SystemExit: hmmt25 has 30 questions (0-29); QIDS asks for [30, 99]
```

The Modal wrappers run the same check locally, so a bad `--qids` or `--dataset`
fails before a GPU is booked.

## Where results are saved

`OUT_DIR` is the folder cell2 writes one pkl per question into and cell3 reads
back. It defaults to `peerconf_out` / `deepconf_out` next to wherever you ran the
script. The resume check reads it too, so it has to outlive the machine — point
it at storage that survives the container or instance:

```bash
OUT_DIR=/out                                   # Modal: a mounted Volume
OUT_DIR=/mnt/results                           # AWS: an attached EBS volume or mounted S3
OUT_DIR=/content/drive/MyDrive/peerconf_out    # Colab: mounted Drive
```

On ephemeral storage the results die with the machine and every rerun starts
from zero.

## Output filenames

Every result is named for the benchmark it came from — `aime25_q6_bar_cs_looph.pkl`,
`math500_q6_bar_cs_looph.pkl` — so benchmarks can share one output directory
without colliding or falsely skipping each other.

Runs saved before the prefix existed are bare (`q6_bar_cs_looph.pkl`). Those
still count as done for `aime25`, so a rerun skips them instead of regenerating
work you already paid for, and the cell3 viewers still read them as aime25.
