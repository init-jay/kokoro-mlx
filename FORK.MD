# Implementation brief: word timestamps for `kokoro-mlx`

**Target repo:** https://github.com/init-jay/kokoro-mlx (fork of
`gabrimatic/kokoro-mlx`, v0.1.2)

**Consumer:** this repo, `lva-wakeword-trainer`. Nothing here needs changing to
*read* the feature - `train/oww/train.py` already consumes exactly this shape from
Kokoro-FastAPI. The fork is what lets us swap the engine.

---

## Why

Kokoro TTS corpus generation is the largest stage of an openWakeWord run on Apple
Silicon: **~50 of ~75 minutes**. Benchmarked on an M1 Max, `kokoro-mlx` is **~3x
faster per clip** than the Kokoro-FastAPI server we use today:

| | Kokoro-FastAPI (best config) | kokoro-mlx |
|---|---|---|
| plain `"hey seeree"`, single | 110 ms (mps) | **61.7 ms** |
| plain, batch of 10 joined | 88 ms (cpu) | **28.8 ms** |
| run-on, single | 143 ms (mps) | **99.6 ms** |
| run-on, batch of 10 joined | 229 ms (cpu) | **71.9 ms** |

That is worth ~1.65x on the whole training run (corpus 50 -> 19 min).

**One thing blocks the swap: word timestamps.** They are not a nice-to-have. The
trainer cuts run-on positives - `"hey seeree what is on tonight"` truncated just
after the wake word - at a boundary taken from per-word times. Without them the
fallback estimate measured **+153 ms late** against a `RUNON_TAIL_MS` of 150-300 ms,
which is the entire tail. That is why microWakeWord has no run-on positives at all,
and why its run-on detection sits at 21-44% while openWakeWord's went from 5% to the
80s once run-ons were added. Losing them to gain 3x would be a bad trade.

**The good news: `kokoro-mlx` already computes the exact quantity and discards it.**

---

## The mechanism, which is exact rather than heuristic

Kokoro-FastAPI does not use forced alignment or energy detection. It reads the
model's own **predicted per-phoneme durations** (`pred_dur`). The audio is rendered
from precisely those durations, so word boundaries are correct by construction.

From `api/src/inference/kokoro_v1.py` in Kokoro-FastAPI:

```python
_ESPEAK_TS_SCALE = 2.0 / 80.0  # pred_dur unit -> seconds (matches KPipeline.join_timestamps)

# "the model predicts a duration for every phoneme in every language (the audio is
#  rendered from exactly those durations). pred_dur[0] is BOS and pred_dur[1 + i]
#  covers phonemes[i], so summing per-character durations and splitting at the
#  phoneme string's spaces yields exact word times."
```

`kokoro-mlx` computes the same tensor in `kokoro_mlx/model.py`, inside
`KokoroModel.forward()` (v0.1.2 line numbers, verify against the fork):

```python
# ~line 477
duration = self.predictor.duration_proj(x)                        # (1, T, 50)
duration = mx.sigmoid(duration).sum(axis=-1) / speed              # (1, T)
pred_dur = mx.clip(mx.round(duration), 0, None).astype(mx.int32).squeeze(0)
pred_dur = mx.maximum(pred_dur, mx.ones_like(pred_dur))
mx.eval(pred_dur)
pred_dur_np = np.array(pred_dur.tolist())                         # <- what we need
...
# ~line 525
return audio.squeeze()                                            # <- durations dropped here
```

`pred_dur_np` and `phonemes` are both in scope at that point. **The work is plumbing
them out and applying the arithmetic, not deriving anything new.**

Note `/ speed` happens before rounding, so durations already account for the speed
multiplier. Do not scale again.

---

## The call chain to thread through

Three files, v0.1.2 line numbers:

| File | Location | Change |
|---|---|---|
| `kokoro_mlx/model.py` | `KokoroModel.forward()`, def ~439, returns ~525 | also return `pred_dur_np` |
| `kokoro_mlx/generate.py` | `generate()`, def ~33, calls `model.forward(...)` ~70 | capture and propagate; `phonemes` already local |
| `kokoro_mlx/kokoro.py` | `KokoroTTS.generate()`, ~94-128; `TTSResult`, ~24-30 | derive timestamps, add field |

`TTSResult` today is `audio, sample_rate, duration, voice`.

Keep the existing return shapes working. A `return_timestamps: bool = False`
argument, or a second return value only when asked for, is preferable to changing
`forward()`'s signature unconditionally - `generate_stream()` also calls
`model.forward()` (`generate.py:121`) and must not break.

---

## Required output contract

This repo reads timestamps in `train/oww/train.py`:

```python
end = timestamps[len(expected) - 1].get("end_time")     # phrase_end_sample()
got = [str(t.get("word", "")).lower() for t in timestamps[:len(expected)]]
```

So each entry must be a mapping (or an object supporting `.get`) with:

| key | type | meaning |
|---|---|---|
| `word` | `str` | the grapheme word, in input order |
| `start_time` | `float` | seconds from clip start |
| `end_time` | `float` | seconds from clip start |

Kokoro-FastAPI rounds to 3 decimals. One entry per **whitespace-separated word of the
input text**, in order. `len(timestamps) == len(text.split())` is the invariant the
consumer's word-matching check depends on.

**Return `None` rather than a guess when the mapping cannot be trusted.** The
consumer treats `None` as "fall back" and the fallback is degraded but honest; wrong
times are silently wrong audio, which is far worse. This repo has already been bitten
by a wrong cut once.

---

## The arithmetic to port

From `_espeak_word_timestamps` in Kokoro-FastAPI. Reproduced because the fork's
implementer will not have that repo to hand:

```python
words = graphemes.split()
durations = [dur * (2.0 / 80.0) for dur in pred_dur.tolist()]

groups, now, start = [], durations[0], None      # durations[0] is BOS
for i, ch in enumerate(phonemes):
    if ch.isspace():
        if start is not None:
            groups.append((start, now)); start = None
    elif start is None:
        start = now
    now += durations[1 + i]
if start is not None:
    groups.append((start, now))
```

Then reconcile group count against word count:

* `len(groups) == len(words)` - done, zip them.
* Otherwise, some words phonemise to several space-separated groups (numbers,
  abbreviations). Re-phonemise each word individually, count its groups, and merge
  runs accordingly. If the totals still disagree, **return `None`**.

Guard the whole thing in `try/except` and return `None` on any failure.

---

## Acceptance tests

**1. Word count and monotonicity.** For `"hey seeree what is on tonight"`, six
entries, times non-decreasing, `end_time` of the last within a few ms of
`len(audio)/sample_rate`.

**2. Agreement with Kokoro-FastAPI — the one that matters.** Render the same text,
same voice (`af_bella`), same speed through both engines and compare `end_time` for
word 2 (`"seeree"`), which is the run-on cut point. They will not be identical - the
weights are bf16 here and the phonemiser differs - but they should agree to within a
few tens of ms. **A disagreement of ~150 ms means the cut is as bad as the fallback
this exists to replace, and the fork has no value.** To produce the reference:

```bash
curl -s localhost:8880/dev/captioned_speech -H 'content-type: application/json' \
  -d '{"model":"kokoro","voice":"af_bella","input":"hey seeree what is on tonight",
       "response_format":"wav","speed":1.0,"stream":false,"return_timestamps":true}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['timestamps'])"
```

**3. Audible check.** Cut the clip at word 2's `end_time` and listen. It should say
`"hey seeree"` and nothing of the command. This is how the original bug was caught;
the measurements that were supposed to catch it did not.

**4. No regression.** `generate()` and `generate_stream()` unchanged when timestamps
are not requested, and no measurable slowdown - `pred_dur` is already computed and
already evaluated to numpy.

---

## Out of scope, and why

**A FastAPI server.** This repo will call the library in-process. The HTTP layer is
what we are removing: batching exists to amortise a ~119 ms per-request cost that an
in-process call does not have.

**Energy-based splitting.** Considered and rejected. Durations give exact boundaries
for free; energy detection cannot find the run-on cut point at all, because there is
no pause between the wake word and the command.

**Matching Kokoro-FastAPI's audio.** Out of scope for the fork, but note for whoever
integrates: `mlx-community/Kokoro-82M-bf16` and `misaki` G2P will not produce
byte-identical audio to the server's. Whether the corpus is *equivalent* is an eval
question for this repo - it already excludes voices that mispronounce the wake word,
so G2P differences are not cosmetic here. Test 2 above bounds the timing risk; it
says nothing about pronunciation.

---

## Environment note

`misaki[en]` pulls `espeakng-loader`, whose wheel hardcodes the path it was built at.
Without a real espeak-ng install it fails with a GitHub Actions runner path:

```
Error processing file '/Users/runner/work/espeakng-loader/.../phontab':
    No such file or directory
```

Fix: `brew install espeak-ng`, then

```bash
export ESPEAK_DATA_PATH=/opt/homebrew/share/espeak-ng-data
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
```

It surfaces *after* the model loads, so it looks like an MLX problem and is not.
