# ======================= CELL 2 — THE LIVE RUN (streaming) =======================
import json, re, time, pickle, os, queue, threading
import numpy as np, requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from dynasor.core.evaluator import math_equal

# ==================== PeerConf CONTROL PANEL ====================

MODEL   = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
SERVER  = "http://localhost:8000"
OUT_DIR = os.environ.get("OUT_DIR", "peerconf_out")   # where results are saved
DATASET = os.environ.get("DATASET", "aime25")         # aime25 | math500 | gsm8k | hmmt25
                                                      # = benchmarks/<name>.jsonl; any file
                                                      # of {"question","answer"} lines works

QIDS           = range(30)  # which questions to run — the first 30, or a set like [6, 9]
SEATS          = 16         # how many traces run at once
WAVE           = SEATS      # how many of those seats the opening wave takes
MAX_TRACES     = 32         # total launch cap across the run
REPLACEMENT_SEATS = SEATS   # how many of those seats may hold replacements at once

# ----- the bar (PeerConf-low/high, from DeepConf-low/high) -----
# Self-calibrating: the run's own finishers are the warmup. Wave 1 (the first
# WAVE traces) runs bar-free; each finisher votes and sends its lifetime-worst
# window score to the calibration set. The bar = keep top BAR_KEEP_TOP% of
# those minima, updated on every new finisher and applied instantly to every replacement
BAR_KEEP_TOP        = 10  # 10 = PeerConf-low, 90 = PeerConf-high 
BAR_MIN_CALIBRATORS = 1   # the first finisher arms the bar (its worst moment IS
                          # the bar); every later finisher refines it

WINDOW         = 2048     # sliding window: a token's score = avg confidence of its
                          # last 2048 tokens. FULL windows only.

# ----- the loop guard (text repetition) -----
LOOP_ACTION      = "cut"      # "off" | "cut" = end the stuck trace on the spot
                            
LOOP_CHECK_EVERY = 256        # tokens between checks
LOOP_UNIT_CHARS  = 120        # repeat unit: the trace's last this-many chars
LOOP_TAIL_CHARS  = 2400       # ...searched within this much trailing text
LOOP_REPEATS     = 3          # fire at this many exact copies in the tail

# ----- the completion probe (CoDE-Stop-style forced answer, fixed schedule) -----
PROBE_EVERY      = 4096       # probe each trace every this-many tokens. 0 = off
PROBE_TEXT       = "\n**Final Answer**\n\nThe final answer is \\boxed"
PROBE_MAX_TOK    = 20         # greedy tokens per probe
PROBE_MIN_TOKS   = 2048       # no probes before the first full window
COMMIT_CONF        = 0.95       # commit on ONE probe: answer-token conf >= this...
COMMIT_EWT         = True       # ...that also reached </think> or <|end|> (</think> is used in this code)

# ----- the margin certificate: if (leader − runner-up) > (live + unlaunched) the question ends

# ----- consensous-based early stopping -----
CONSENSUS      = 0.95     # checked after EVERY finished trace; if the leading answer
                          # holds this share of the weighted votes among finished
                          # traces, stop launching AND end the in-flight streams
                          # on the spot.
                          # Anything > 1.0 = DISABLED (a fraction of the votes can never beat 1.0;
                          # exactly 1.0 = stop only on unanimity).
CONSENSUS_MIN_FINISHERS = 3   # how many traces must have finished before the rule
                              # may fire at all. 
# ----- generation -----
TEMPERATURE    = 0.6
TOP_P          = 0.95
TOP_K          = -1
LOGPROBS       = 20       # top-20 candidates per token — the confidence data
MAX_TOK_TRACE  = 64000    # total generation cap per trace, counted in tokens GENERATED
# ================================================================

tok    = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

_bench = f"{DATASET}.jsonl"
_bench_paths = [_bench, os.path.join("benchmarks", _bench),
                os.path.join("..", "benchmarks", _bench)]
try:
    _here = os.path.dirname(os.path.abspath(__file__))
    _bench_paths[:0] = [os.path.join(_here, *([".."] * up), "benchmarks", _bench)
                        for up in (1, 2)]
except NameError:
    pass
_bench_path = next((p for p in _bench_paths if os.path.exists(p)), None)
if _bench_path is None:
    raise FileNotFoundError(f"{_bench} not found — looked in {_bench_paths}")
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

def render_chat(user_text):
    return tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True)

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
    def __init__(self, tid):
        self.id = tid
        self.prompt_text = BASE_PROMPT
        self.gen_text  = ""
        self.confs     = []
        self.win       = None
        self.toks_gen  = 0
        self.judge_min = float("inf")
        self.kill      = threading.Event()
        self.pending   = None
        self.probes    = []
        self.probe_toks = 0
        self.probe_fails = 0
        self.probing   = False
        self.last_probe_at = 0
        self.last_loop_check = 0
        self.commit_answer = None
        self.looped    = False
        self.committed = False
        self.status    = "running"
        self.answer    = None

def update_bar():
    global bar
    mins = [x.judge_min for x in traces
            if x.status == "finished" and x.confs]
    if len(mins) < BAR_MIN_CALIBRATORS:
        return
    new = float(np.percentile(mins, 100 - BAR_KEEP_TOP))
    if bar is None:
        print(f"Bar ARMED at {new:.3f} ({len(mins)} finishers calibrating, "
              f"keep top {BAR_KEEP_TOP}%)")
    bar = new
    LINE_HISTORY.append({"tid": -1, "line": bar, "n_traces": len(mins),
                         "t": time.time() - t_start})

def consensus_check():
    piles = {}
    wave_out = all(x.status != "running" for x in traces if x.id < WAVE)
    for t in traces:
        if t.status == "finished" and t.answer is not None and t.confs:
            if wave_out and bar is not None and t.id < WAVE and min(t.confs) < bar:
                continue
            _k = ballot_key(piles, t.answer)
            piles[_k] = piles.get(_k, 0.0) + min(t.confs)
    if not piles: return None, 0.0
    a = max(piles, key=piles.get)
    return a, piles[a] / sum(piles.values())

def certificate():
    piles = {}
    for x in traces:
        if x.status == "finished" and x.answer is not None:
            _k = ballot_key(piles, x.answer)
            piles[_k] = piles.get(_k, 0) + 1
    if not piles:
        return None
    ranked = sorted(piles.values(), reverse=True)
    margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0)
    outstanding = len(inflight) + (MAX_TRACES - launched)
    if margin > outstanding:
        return max(piles, key=piles.get)
    return None

def looping(text):
    unit = text[-LOOP_UNIT_CHARS:]
    if len(unit) < LOOP_UNIT_CHARS:
        return False
    return text[-LOOP_TAIL_CHARS:].count(unit) >= LOOP_REPEATS

def run_probe(t, prompt, at_toks):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": PROBE_MAX_TOK,
            "temperature": 0.0, "logprobs": 1, "stop": ["</think>"]}
    try:
        r = SESSION.post(f"{SERVER}/v1/completions", json=body, timeout=90)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        lp = ch.get("logprobs") or {}
        toks = lp.get("tokens") or []
        tlps = lp.get("token_logprobs") or []
        text = ch.get("text") or ""
        blended = float(np.exp(np.mean(tlps))) if tlps else 0.0
        i0, i1 = text.find("{"), text.find("}", max(text.find("{"), 0))
        ans_lps, pos = [], 0
        for tk, l in zip(toks, tlps):
            s, e = pos, pos + len(tk)
            pos = e
            if i0 >= 0 and i1 > i0 and e > i0 + 1 and s < i1 and l is not None:
                ans_lps.append(l)
        conf = float(np.exp(np.mean(ans_lps))) if ans_lps else blended
        events.put((t, {"kind": "probe", "at": at_toks, "text": text,
                        "conf": conf, "blended": blended, "ntok": len(tlps),
                        "toks": toks, "tlps": tlps,
                        "conf_src": "answer" if ans_lps else "blended",
                        "ewt": ch.get("finish_reason") == "stop"}, None))
    except Exception as e:
        events.put((t, {"kind": "probe", "at": at_toks, "failed": str(e),
                        "text": "", "conf": 0.0, "blended": 0.0,
                        "ntok": 0, "toks": [], "tlps": [],
                        "conf_src": "none", "ewt": False}, None))

def launch_probe(t):
    t.probing = True
    t.last_probe_at = t.toks_gen
    executor.submit(run_probe, t, t.prompt_text + t.gen_text + PROBE_TEXT, t.toks_gen)

def run_stream(t, prompt_text, max_toks):
    body = {"model": MODEL, "prompt": prompt_text,
            "max_tokens": int(max_toks), "temperature": TEMPERATURE,
            "top_p": TOP_P, "top_k": TOP_K, "logprobs": LOGPROBS, "stream": True}
    try:
        r = SESSION.post(f"{SERVER}/v1/completions", json=body, stream=True, timeout=7200)
        r.raise_for_status()
        win = t.win if t.win is not None else deque(maxlen=WINDOW)
        s = sum(win)
        b_scores, b_text, b_toks = [], [], 0
        fin = None
        for raw in r.iter_lines():
            if t.kill.is_set():
                r.close()
                fin = fin or "killed"
                break
            if not raw: continue
            if not raw.startswith(b"data: "): continue
            payload = raw[6:]
            if payload == b"[DONE]": break
            ch = json.loads(payload)["choices"][0]
            b_text.append(ch.get("text") or "")
            lps = (ch.get("logprobs") or {}).get("top_logprobs") or []
            for d in lps:
                if not d: continue
                vals = sorted(d.values(), reverse=True)[:LOGPROBS]
                c = -float(np.mean(vals))
                if len(win) == WINDOW: s -= win[0]
                win.append(c); s += c
                b_toks += 1
                if len(win) == WINDOW:
                    b_scores.append(s / len(win))
            if ch.get("finish_reason"):
                fin = ch["finish_reason"]
            events.put((t, {"kind": "batch", "text": "".join(b_text),
                            "scores": b_scores, "ntok": b_toks}, None))
            b_scores, b_text, b_toks = [], [], 0
        t.win = win
        events.put((t, {"kind": "end", "text": "".join(b_text),
                        "scores": b_scores, "ntok": b_toks,
                        "finish": fin or "stop"}, None))
    except Exception as e:
        events.put((t, None, e))

def launch(t):
    budget = MAX_TOK_TRACE - t.toks_gen
    t.kill = threading.Event()
    t.pending = None
    inflight.add(t.id)
    executor.submit(run_stream, t, t.prompt_text + t.gen_text, max(budget, 1))

def judge(t, scores):
    if scores and min(scores) < t.judge_min:
        t.judge_min = min(scores)
    if t.id < WAVE or bar is None:
        return None
    for sc in scores:
        if sc < bar:
            return "cut"
    return None

# ---------------- voting helpers (used once per question, after its run) ----------------
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
    if not voters:
        return None, None, 0
    confs = [M[t.id][measure] for t in voters]
    thr   = np.percentile(confs, (1 - top_percent) * 100)
    elite = [t for t in voters if M[t.id][measure] >= thr]
    return weighted_vote(measure, only_traces=elite)

executor = ThreadPoolExecutor(max_workers=WAVE + REPLACEMENT_SEATS + 4)
os.makedirs(OUT_DIR, exist_ok=True)
t_sweep = time.time()

# ==================== THE SWEEP: one run per question ====================
for QID in QIDS:
    _extras = (("_cs" if PROBE_EVERY > 0 else "")
               + (f"_loop{LOOP_ACTION[0]}" if LOOP_ACTION != "off" else ""))
    save_path = f"{OUT_DIR}/{DS_TAG}q{QID}_bar{_extras}.pkl"
    _legacy = f"{OUT_DIR}/q{QID}_bar{_extras}.pkl" if DATASET == "aime25" else None
    _done = next((p for p in (save_path, _legacy) if p and os.path.exists(p)), None)
    if _done:
        print(f"Q{QID}: already saved ({_done}) — skipping")
        continue

    print(f"\n{'=' * 60}\n### Q{QID}  ({(time.time() - t_sweep) / 60:.0f} min into the sweep)\n{'=' * 60}")
    question, ground_truth = data[QID]["question"], str(data[QID]["answer"]).strip()
    print(f"Q{QID}: {question[:80]}...\nGround truth: {ground_truth}\n")
    BASE_PROMPT = render_chat(question)

    # -------- fresh run state (mutated ONLY by the main thread) --------
    traces    = [Trace(i) for i in range(SEATS)]
    launched  = SEATS
    bar       = None
    LINE_HISTORY = []
    t_start   = time.time()
    events    = queue.Queue()
    inflight  = set()
    run_over = False
    n_events  = 0

    # -------- the run loop: stream in, judge live --------
    print(f"Run start: {SEATS} seats (cap {MAX_TRACES}), streaming "
          f"| wave 1 line-free; replacements face the "
          f"self-calibrating bar (keep top {BAR_KEEP_TOP}%, arms at "
          f"{BAR_MIN_CALIBRATORS} finishers, updates per finisher)"
          + (f" | probes every {PROBE_EVERY} (commit at {COMMIT_CONF})"
             if PROBE_EVERY > 0 else "")
          + (f" | loop guard ON ({LOOP_ACTION})" if LOOP_ACTION != "off" else "")
          + f" | close: landslide ({CONSENSUS:.0%}) OR certificate")
    for t in traces:
        launch(t)

    while inflight:
        t, payload, err = events.get()
        n_events += 1

        # ---- completion-probe verdicts ----
        if payload is not None and payload.get("kind") == "probe":
            t.probing = False
            t.probe_toks += payload["ntok"]
            if "failed" in payload:
                t.probe_fails += 1
                print(f"Trace {t.id}: probe failed at {payload['at']} tokens "
                      f"({payload['failed'][:60]}) — skipped")
                continue
            p_ans = extract_answer("\\boxed" + payload["text"])
            rec = {"at": payload["at"], "conf": payload["conf"],
                   "blended": payload["blended"],
                   "ewt": payload["ewt"], "answer": p_ans,
                   "text": payload["text"], "ntok": payload["ntok"],
                   "toks": payload["toks"], "tlps": payload["tlps"],
                   "conf_src": payload["conf_src"]}
            t.probes.append(rec)
            if (t.pending is None and not run_over and t.status == "running"
                    and rec["conf"] >= COMMIT_CONF and (rec["ewt"] or not COMMIT_EWT)
                    and p_ans is not None):
                t.commit_answer = p_ans
                t.pending = "commit"
                t.kill.set()
                print(f"Trace {t.id}: COMMITTED at {payload['at']} tokens — "
                      f"conf {rec['conf']:.3f}, answer {p_ans}")
            continue

        if err:
            inflight.discard(t.id)
            t.status = "truncated"
            print(f"Trace {t.id}: stream failed -> truncated ({err})")
        elif payload["kind"] == "batch":
            t.gen_text += payload["text"]; t.toks_gen += payload["ntok"]
            t.confs += payload["scores"]
            if t.pending is None and not run_over:
                verdict = judge(t, payload["scores"])
                if verdict is not None:
                    t.pending = verdict
                    t.kill.set()
            if (LOOP_ACTION != "off" and t.pending is None and not run_over
                    and t.toks_gen - t.last_loop_check >= LOOP_CHECK_EVERY):
                t.last_loop_check = t.toks_gen
                if looping(t.gen_text):
                    t.looped = True
                    t.pending = "cut"
                    t.kill.set()
                    print(f"Trace {t.id}: LOOPING at {t.toks_gen} tokens "
                          f"(tail unit repeats >= {LOOP_REPEATS}x) -> ended, no ballot")
            if (PROBE_EVERY > 0 and not t.probing and t.pending is None
                    and not run_over and t.toks_gen >= PROBE_MIN_TOKS
                    and t.toks_gen - t.last_probe_at >= PROBE_EVERY):
                launch_probe(t)
            continue
        else:
            inflight.discard(t.id)
            t.gen_text += payload["text"]; t.toks_gen += payload["ntok"]
            fin = payload["finish"]
            t.confs += payload["scores"]
            if t.pending is None and not run_over and payload["scores"]:
                verdict = judge(t, payload["scores"])
                if verdict is not None:
                    t.pending = verdict

            ans = extract_answer(t.gen_text)

            if run_over:
                t.status = "abandoned"
            elif t.pending == "commit":
                t.status, t.answer = "finished", t.commit_answer
                t.committed = True
                print(f"Trace {t.id}: finished EARLY at {t.toks_gen} tokens "
                      f"(completion probe) | answer={t.answer}")
            elif t.pending == "cut":
                t.status = "stopped"
                if t.looped:
                    print(f"Trace {t.id}: loop ended at {t.toks_gen} tokens (no ballot)")
                else:
                    print(f"Trace {t.id}: cut at the bar "
                          + (f"{bar:.3f} " if bar is not None else "")
                          + f"(its low {t.judge_min:.3f}, {t.toks_gen} tokens)")
            elif ans is not None:
                t.status, t.answer = "finished", ans
                print(f"Trace {t.id}: finished at {t.toks_gen} tokens | answer={ans}")
            elif fin == "stop":
                t.status = "truncated"
                print(f"Trace {t.id}: ended without a boxed answer at {t.toks_gen} "
                      f"tokens -> truncated")
            else:
                t.status = "truncated"
                print(f"Trace {t.id}: truncated at {t.toks_gen} tokens (cap)")

        # ---- this trace departed: per-trace pit stop ----
        if t.status == "finished":
            update_bar()
        if CONSENSUS <= 1.0 and not run_over:
            lead, share = consensus_check()
            n_fin = sum(1 for x in traces if x.status == "finished")
            if (lead is not None and share >= CONSENSUS
                    and n_fin >= CONSENSUS_MIN_FINISHERS):
                run_over = True
                live_ids = [x.id for x in traces if x.id in inflight]
                for x in traces:
                    if x.id in inflight:
                        x.kill.set()
                print(f"Consensus after {n_fin} finishers: '{lead}' holds {share:.0%} — "
                      f"killing {len(live_ids)} in-flight streams, no new launches")
        if not run_over:
            winner = certificate()
            if winner is not None:
                run_over = True
                live_ids = [x.id for x in traces if x.id in inflight]
                for x in traces:
                    if x.id in inflight:
                        x.kill.set()
                print(f"CERTIFICATE: '{winner}' cannot be caught "
                      f"(margin exceeds all {len(live_ids)} outstanding) — run over")
        reps_live = sum(1 for x in traces if x.id >= WAVE and x.id in inflight)
        wave_done = all(x.status != "running" for x in traces if x.id < WAVE)
        target = REPLACEMENT_SEATS if wave_done else min(reps_live + 1, REPLACEMENT_SEATS)
        while (not run_over and launched < MAX_TRACES and reps_live < target):
            nt = Trace(launched); traces.append(nt)
            print(f"Trace {launched}: seated"
                  + (f" | bar {bar:.3f}" if bar is not None else " | bar unarmed"))
            launched += 1
            launch(nt)
            reps_live += 1

    # -------- results + voting table --------
    done   = traces
    voters = [t for t in done if t.status == "finished" and t.answer is not None
              and not (bar is not None and t.id < WAVE
                       and t.confs and min(t.confs) < bar)]
    M = {t.id: trace_measures(t) for t in voters}

    print(f"\nBasic voting candidates: {len(voters)}")
    print(f"Sample voting answers: {[t.answer for t in voters][:8]}")
    voting_results = {
        "majority":                     weighted_vote(None),
        "mean_confidence_weighted":     weighted_vote("mean_conf"),
        "tail_confidence_weighted":     weighted_vote("tail_conf"),
        "bottom_window_weighted":       weighted_vote("bottom_window"),
        "min_window_weighted":          weighted_vote("min_window"),
        "top10_tail_filtered":          top_filtered("tail_conf", 0.10),
        "top10_bottom_window_filtered": top_filtered("bottom_window", 0.10),
    }

    probe_tokens = sum(t.probe_toks for t in done)
    total_tokens = sum(t.toks_gen for t in done) + probe_tokens
    n_status     = lambda s: sum(1 for x in done if x.status == s)
    n_committed  = sum(1 for t in done if t.committed)
    n_looped     = sum(1 for t in done if t.looped)
    print("\n=== PeerConf Summary (streaming) ===")
    print(f"Final bar: {bar if bar is None else f'{bar:.3f}'} "
          f"(keep top {BAR_KEEP_TOP}% of finishers' minima) | "
          f"seats {SEATS} | launched {launched}/{MAX_TRACES} | events: {n_events}")
    print(f"Traces: finished {n_status('finished')} | stopped {n_status('stopped')} "
          f"| truncated {n_status('truncated')} | abandoned {n_status('abandoned')}")
    if PROBE_EVERY > 0:
        n_pfail = sum(t.probe_fails for t in done)
        print(f"Completion probes: {sum(len(t.probes) for t in done)} fired "
              f"({probe_tokens} probe tokens) | committed early: {n_committed}"
              + (f" | {n_pfail} failed" if n_pfail else ""))
    if LOOP_ACTION != "off":
        print(f"Loop guard: {n_looped} loops caught and ended (no ballots cast)")
    print(f"Valid answers for voting: {len(voters)}")
    print(f"Final answer: {voting_results['majority'][0]}   | ground truth: {ground_truth}")
    print(f"Total tokens generated (traces + probes, incl. discarded): {total_tokens}"
          + (f" (of which probes: {probe_tokens})" if probe_tokens else ""))
    print(f"Total time: {time.time()-t_start:.2f}s")

    print("\n=== Voting Results Summary ===")
    for m, (a, conf, v) in voting_results.items():
        cs = f" (conf: {conf:.3f})" if conf is not None else ""
        print(f"  {m}: {a}{cs} [{v} votes]")

    print("\n=== Detailed Voting Results ===")
    print("-" * 70)
    print(f"{'Method':<30}{'Answer':<12}{'Votes':<7}{'Confidence':<12}{'Correct'}")
    print("-" * 70)
    for m, (a, conf, v) in voting_results.items():
        ok = "✓" if is_correct(a, ground_truth) else "✗"
        cs = f"{conf:.3f}" if conf is not None else "-"
        print(f"{m:<30}{str(a):<12}{v:<7}{cs:<12}{ok}")
    marks = ", ".join(f"{m}:{'✓' if is_correct(r[0], ground_truth) else '✗'}" for m, r in voting_results.items())
    print(f"Q{QID} done | gt={ground_truth} | {marks}")

    # -------- save (results + the confidence-timeline game tape) --------
    with open(save_path, "wb") as f:
        pickle.dump({"qid": QID, "gt": ground_truth,
                     "config": {"MODE": "server-stream", "MODEL": MODEL,
                                "DATASET": DATASET,
                                "BAR_KEEP_TOP": BAR_KEEP_TOP,
                                "BAR_MIN_CALIBRATORS": BAR_MIN_CALIBRATORS,
                                "WINDOW": WINDOW,
                                "LOOP_ACTION": LOOP_ACTION,
                                "LOOP_CHECK_EVERY": LOOP_CHECK_EVERY,
                                "LOOP_UNIT_CHARS": LOOP_UNIT_CHARS,
                                "LOOP_TAIL_CHARS": LOOP_TAIL_CHARS,
                                "LOOP_REPEATS": LOOP_REPEATS,
                                "PROBE_EVERY": PROBE_EVERY,
                                "PROBE_MAX_TOK": PROBE_MAX_TOK,
                                "PROBE_MIN_TOKS": PROBE_MIN_TOKS,
                                "COMMIT_CONF": COMMIT_CONF,
                                "COMMIT_EWT": COMMIT_EWT,
                                "SEATS": SEATS, "WAVE": WAVE, "MAX_TRACES": MAX_TRACES,
                                "REPLACEMENT_SEATS": REPLACEMENT_SEATS,
                                "MAX_TOK_TRACE": MAX_TOK_TRACE,
                                "CONSENSUS": CONSENSUS,
                                "CONSENSUS_MIN_FINISHERS": CONSENSUS_MIN_FINISHERS,
                                "final_bar": bar},
                     "voting": voting_results, "tokens": total_tokens,
                     "time_s": round(time.time() - t_start, 2),
                     "probe_tokens": probe_tokens,
                     "launched": launched,
                     "line_history": LINE_HISTORY,
                     "mins": {t.id: t.judge_min for t in done if t.confs},
                     "traces": [{"id": t.id, "status": t.status, "answer": t.answer,
                                 "toks_gen": t.toks_gen,
                                 "confs": t.confs,
                                 "looped": t.looped,
                                 "committed": t.committed,
                                 "probes": t.probes,
                                 "probe_toks": t.probe_toks,
                                 "probe_fails": t.probe_fails,
                                 "text": t.gen_text,
                                 "prompt": t.prompt_text} for t in done]}, f)
    print(f"Saved to {save_path}")

print(f"\nSweep over: {len(list(QIDS))} questions requested, "
      f"{(time.time() - t_sweep) / 60:.0f} min total")
