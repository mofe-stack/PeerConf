# ================= CELL 2 — DEEPCONF BASELINE (online, streaming) =================
# DeepConf low/high as the paper runs it (CONFIDENCE_PERCENTILE picks: 10 = low,
# 90 = high): 16 warmup traces run to completion, the bar freezes at the
# percentile of their worst window scores, then the online wave streams token by
# token and a trace is cut the instant its sliding-window confidence drops below
# the bar. Consensus stop (tau) from the paper included.
import json, re, time, pickle, os, queue, threading
import numpy as np, requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from dynasor.core.evaluator import math_equal

MODEL   = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
SERVER  = "http://localhost:8000"
OUT_DIR = globals().get("OUT_DIR") or os.environ.get("OUT_DIR", "deepconf_out")
DATASET = os.environ.get("DATASET", "aime25")

QIDS           = range(30)
WARMUP_TRACES  = 16
TOTAL_BUDGET   = 32
CONFIDENCE_PERCENTILE = 10
WINDOW         = 2048
CONSENSUS      = 0.95

TEMPERATURE    = 0.6
TOP_P          = 0.95
TOP_K          = -1
LOGPROBS       = 20
MAX_TOK_TRACE  = 64000

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

_bench = f"{DATASET}.jsonl"
_bench_path = next((p for p in (_bench, f"../benchmarks/{_bench}",
                                f"benchmarks/{_bench}") if os.path.exists(p)), None)
if _bench_path is None:
    raise FileNotFoundError(f"{_bench} not found — add it to benchmarks/ "
                            f"(one question per line, same fields as aime25.jsonl)")
with open(_bench_path, encoding="utf-8") as f:
    data = [json.loads(l) for l in f]

QIDS = list(QIDS)
_bad = [q for q in QIDS if not 0 <= q < len(data)]
if _bad:
    raise SystemExit(f"{DATASET} has {len(data)} questions (0-{len(data) - 1}); "
                     f"QIDS asks for {_bad}")

DS_TAG = f"{DATASET}_"
print(f"Benchmark: {DATASET} — {len(data)} questions from {_bench_path}, "
      f"running {len(QIDS)}")

import re

SESSION = requests.Session()

_WS = re.compile(r"\s+")

def tidy_tex(a):
    s = str(a)
    for x, y in ((r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"),
                 (r"\left", ""), (r"\right", ""),
                 (r"\!", ""), (r"\,", " "), (r"\;", " "), (r"\ ", " ")):
        s = s.replace(x, y)
    return _WS.sub(" ", s).strip()

def same_answer(a, b):
    if a is None or b is None:
        return False
    a, b = tidy_tex(a), tidy_tex(b)
    if a == b or _WS.sub("", a) == _WS.sub("", b):
        return True
    try:
        return bool(math_equal(a, b))
    except Exception:
        return False

def ballot_key(piles, ans):
    for k in piles:
        if same_answer(ans, k):
            return k
    return str(ans)

def is_correct(ans, gt):
    if ans is None: return False
    return same_answer(ans, gt)

def extract_answer(text):
    if "boxed" not in text:
        return None
    ans = text.split("boxed")[-1]
    if not ans:
        return None
    if ans[0] == "{":
        stack, a = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a.strip() or None

class Trace:
    def __init__(self, tid, phase):
        self.id       = tid
        self.phase    = phase
        self.gen_text = ""
        self.confs    = []
        self.toks_gen = 0
        self.kill     = threading.Event()
        self.pending  = False
        self.status   = "running"
        self.answer   = None
    @property
    def min_conf(self):
        return min(self.confs) if self.confs else 0.0

def run_stream(t, prompt_text, max_toks):
    body = {"model": MODEL, "prompt": prompt_text, "max_tokens": int(max_toks),
            "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "logprobs": LOGPROBS, "stream": True}
    try:
        r = SESSION.post(f"{SERVER}/v1/completions", json=body, stream=True, timeout=7200)
        r.raise_for_status()
        win, s = deque(maxlen=WINDOW), 0.0
        fin = None
        for raw in r.iter_lines():
            if t.kill.is_set():
                r.close()
                break
            if not raw or not raw.startswith(b"data: "): continue
            payload = raw[6:]
            if payload == b"[DONE]": break
            ch = json.loads(payload)["choices"][0]
            text = ch.get("text") or ""
            scores = []
            for d in (ch.get("logprobs") or {}).get("top_logprobs") or []:
                if not d: continue
                vals = sorted(d.values(), reverse=True)[:LOGPROBS]
                c = -float(np.mean(vals))
                if len(win) == WINDOW: s -= win[0]
                win.append(c); s += c
                if len(win) == WINDOW:
                    scores.append(s / WINDOW)
            if ch.get("finish_reason"):
                fin = ch["finish_reason"]
            events.put((t, {"kind": "tok", "text": text, "scores": scores,
                            "ntok": len((ch.get("logprobs") or {}).get("top_logprobs") or [])}, None))
        events.put((t, {"kind": "end", "finish": fin or "stop"}, None))
    except Exception as e:
        events.put((t, None, e))

def launch(t):
    t.kill = threading.Event()
    t.pending = False
    inflight.add(t.id)
    budget = MAX_TOK_TRACE - t.toks_gen
    executor.submit(run_stream, t, PROMPT + t.gen_text, max(budget, 1))

def land(t, fin):
    ans = extract_answer(t.gen_text)
    if run_over and t.status == "running" and not t.pending:
        t.status = "abandoned"
    elif t.pending:
        t.status = "stopped"
        print(f"Trace {t.id}: cut at the bar {conf_bar:.3f} "
              f"(its low {t.min_conf:.3f}, {t.toks_gen} tokens)")
    elif ans is not None:
        t.status, t.answer = "finished", ans
        print(f"Trace {t.id} ({t.phase}): finished at {t.toks_gen} tokens | answer={ans}")
    else:
        t.status = "truncated"
        print(f"Trace {t.id} ({t.phase}): truncated at {t.toks_gen} tokens")

def voters_now():
    v = []
    for t in traces:
        if t.status != "finished" or t.answer is None or not t.confs: continue
        if t.phase == "warmup" and conf_bar is not None and t.min_conf < conf_bar: continue
        v.append(t)
    return v

def consensus_check():
    piles = {}
    for t in voters_now():
        _k = ballot_key(piles, t.answer)
        piles[_k] = piles.get(_k, 0.0) + t.min_conf
    if not piles: return None, 0.0
    a = max(piles, key=piles.get)
    return a, piles[a] / sum(piles.values())

def drain(phase_traces):
    global run_over
    while any(t.id in inflight for t in phase_traces):
        t, payload, err = events.get()
        if err:
            inflight.discard(t.id)
            t.status = "truncated"
            print(f"Trace {t.id}: stream failed -> truncated ({err})")
        elif payload["kind"] == "tok":
            t.gen_text += payload["text"]; t.toks_gen += payload["ntok"]
            t.confs += payload["scores"]
            if (t.phase == "online" and not t.pending and not run_over
                    and conf_bar is not None
                    and any(sc < conf_bar for sc in payload["scores"])):
                t.pending = True
                t.kill.set()
            continue
        else:
            inflight.discard(t.id)
            land(t, payload["finish"])
            if CONSENSUS <= 1.0 and not run_over and t.status == "finished":
                lead, share = consensus_check()
                if lead is not None and share >= CONSENSUS and conf_bar is not None:
                    run_over = True
                    live = [x for x in traces if x.id in inflight]
                    for x in live: x.kill.set()
                    print(f"Consensus: '{lead}' holds {share:.0%} — "
                          f"ending {len(live)} in-flight streams")

# ---------------- voting helpers (used once per question) ----------------
def trace_measures(t):
    c = np.array(t.confs) if t.confs else np.array([0.0])
    k = max(1, int(len(c) * 0.10))
    return {"mean_conf":     float(c.mean()),
            "tail_conf":     float(c[-1]),
            "bottom_window": float(np.sort(c)[:k].mean()),
            "min_window":    float(c.min())}

def weighted_vote(weight_key=None, only_traces=None):
    pool = only_traces if only_traces is not None else voters
    piles = {}
    for t in pool:
        w = 1.0 if weight_key is None else M[t.id][weight_key]
        _k = ballot_key(piles, t.answer)
        piles[_k] = piles.get(_k, 0.0) + float(w)
    if not piles: return None, None, 0
    a = max(piles.keys(), key=lambda x: piles[x])
    conf = float(np.mean([M[t.id][weight_key] for t in pool])) if weight_key else None
    return a, conf, len(pool)

def top_filtered(measure, top_percent=0.1):
    if not voters: return None, None, 0
    thr = np.percentile([M[t.id][measure] for t in voters], (1 - top_percent) * 100)
    return weighted_vote(measure, [t for t in voters if M[t.id][measure] >= thr])

executor = ThreadPoolExecutor(max_workers=TOTAL_BUDGET + 4)
os.makedirs(OUT_DIR, exist_ok=True)
t_sweep = time.time()

# ==================== THE SWEEP: one run per question ====================
for QID in QIDS:
    _name = f"q{QID}_deepconf_p{CONFIDENCE_PERCENTILE}_c{int(CONSENSUS*100)}.pkl"
    save_path = f"{OUT_DIR}/{DS_TAG}{_name}"
    _legacy = f"{OUT_DIR}/{_name}" if DATASET == "aime25" else None
    _done = next((p for p in (save_path, _legacy) if p and os.path.exists(p)), None)
    if _done:
        print(f"Q{QID}: already saved ({_done}) — skipping")
        continue

    print(f"\n{'=' * 60}\n### Q{QID}  ({(time.time() - t_sweep) / 60:.0f} min into the sweep)\n{'=' * 60}")
    question, ground_truth = data[QID]["question"], str(data[QID]["answer"]).strip()
    print(f"Q{QID}: {question[:80]}...\nGround truth: {ground_truth}\n")
    PROMPT = tok.apply_chat_template([{"role": "user", "content": question}],
                                     tokenize=False, add_generation_prompt=True)

    # -------- fresh run state --------
    events   = queue.Queue()
    inflight = set()
    t_start  = time.time()
    conf_bar = None
    run_over = False

    # ---- phase 1: warmup, never judged ----
    print(f"Warmup: {WARMUP_TRACES} traces, streaming per token")
    traces = [Trace(i, "warmup") for i in range(WARMUP_TRACES)]
    for t in traces: launch(t)
    drain(traces)

    wmins = [t.min_conf for t in traces if t.confs]
    conf_bar = float(np.percentile(wmins, 100 - CONFIDENCE_PERCENTILE))
    print(f"\nWarmup done: bar frozen at {conf_bar:.3f} "
          f"(keep top {CONFIDENCE_PERCENTILE}% of {len(wmins)} warmup minima)")

    # ---- phase 2: online wave, judged per token at the frozen bar ----
    if CONSENSUS <= 1.0:
        lead, share = consensus_check()
        if lead is not None and share >= CONSENSUS:
            run_over = True
            print(f"Consensus already held at warmup ('{lead}' at {share:.0%}) — online wave skipped")
    if not run_over:
        wave = [Trace(WARMUP_TRACES + i, "online")
                for i in range(TOTAL_BUDGET - WARMUP_TRACES)]
        traces += wave
        print(f"Online wave: {len(wave)} traces at the bar")
        for t in wave: launch(t)
        drain(wave)

    # -------- voting: the repo's seven methods over the voting pool --------
    voters = voters_now()
    M = {t.id: trace_measures(t) for t in voters}

    voting_results = {
        "majority":                     weighted_vote(None),
        "mean_confidence_weighted":     weighted_vote("mean_conf"),
        "tail_confidence_weighted":     weighted_vote("tail_conf"),
        "bottom_window_weighted":       weighted_vote("bottom_window"),
        "min_window_weighted":          weighted_vote("min_window"),
        "top10_tail_filtered":          top_filtered("tail_conf", 0.10),
        "top10_bottom_window_filtered": top_filtered("bottom_window", 0.10),
    }

    total_tokens = sum(t.toks_gen for t in traces)
    n_status = lambda s: sum(1 for x in traces if x.status == s)

    print(f"\n=== DeepConf Summary (online, streaming) ===")
    print(f"Bar: {conf_bar:.3f} (keep top {CONFIDENCE_PERCENTILE}%) | "
          f"warmup {WARMUP_TRACES} | budget {TOTAL_BUDGET} | tau {CONSENSUS}")
    print(f"Traces: finished {n_status('finished')} | stopped {n_status('stopped')} "
          f"| truncated {n_status('truncated')} | abandoned {n_status('abandoned')}")
    print(f"Voting pool: {len(voters)} (warmup above bar + online finishers)")
    print(f"Total tokens: {total_tokens} | time: {time.time()-t_start:.2f}s")

    print("\n" + "-" * 70)
    print(f"{'Method':<30}{'Answer':<12}{'Votes':<7}{'Confidence':<12}{'Correct'}")
    print("-" * 70)
    for m, (a, conf, v) in voting_results.items():
        ok = "✓" if is_correct(a, ground_truth) else "✗"
        cs = f"{conf:.3f}" if conf is not None else "-"
        print(f"{m:<30}{str(a):<12}{v:<7}{cs:<12}{ok}")
    marks = ", ".join(f"{m}:{'✓' if is_correct(r[0], ground_truth) else '✗'}"
                      for m, r in voting_results.items())
    print(f"Q{QID} done | gt={ground_truth} | {marks}")

    with open(save_path, "wb") as f:
        pickle.dump({"qid": QID, "gt": ground_truth,
                     "config": {"MODE": "deepconf-online-stream", "MODEL": MODEL,
                                "DATASET": DATASET,
                                "CONFIDENCE_PERCENTILE": CONFIDENCE_PERCENTILE,
                                "WARMUP_TRACES": WARMUP_TRACES,
                                "TOTAL_BUDGET": TOTAL_BUDGET,
                                "WINDOW": WINDOW, "CONSENSUS": CONSENSUS,
                                "MAX_TOK_TRACE": MAX_TOK_TRACE,
                                "conf_bar": conf_bar},
                     "voting": voting_results, "tokens": total_tokens,
                     "time_s": round(time.time() - t_start, 2),
                     "warmup_min_confs": wmins, "conf_bar": conf_bar,
                     "traces": [{"id": t.id, "phase": t.phase, "status": t.status,
                                 "answer": t.answer, "toks_gen": t.toks_gen,
                                 "confs": t.confs, "text": t.gen_text,
                                 "prompt": PROMPT} for t in traces]}, f)
    print(f"Saved to {save_path}")

print(f"\nSweep over: {len(list(QIDS))} questions requested, "
      f"{(time.time() - t_sweep) / 60:.0f} min total")
