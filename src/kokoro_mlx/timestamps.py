# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Word timestamps derived from the model's predicted phoneme durations.

Kokoro renders audio from exactly the durations its duration predictor emits,
so summing those durations and splitting at the phoneme string's spaces gives
word boundaries that are exact by construction rather than estimated.

The arithmetic on top of those sums follows ``KPipeline.join_timestamps``
upstream, which is what Kokoro-FastAPI reports for English voices: each word
runs to the midpoint of the silence that follows it, not to its last phoneme.
See :func:`_groups`.

Every helper here returns ``None`` rather than a guess when the phoneme groups
cannot be mapped onto the input words.  A caller that falls back to its own
estimate is degraded but honest; silently wrong boundaries cut audio in the
wrong place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

# pred_dur unit -> seconds.  Matches KPipeline.join_timestamps upstream.
TS_SCALE = 2.0 / 80.0

WordTimestamp = dict[str, object]


def phoneme_units(phonemes: str, vocab: Mapping[str, int]) -> str:
    """Return the phoneme characters the model actually saw.

    ``KokoroModel.forward`` drops characters missing from the vocabulary before
    building ``input_ids``, so ``pred_dur`` is indexed against this filtered
    string rather than the raw phoneme string.
    """
    return "".join(c for c in phonemes if c in vocab)


def _groups(units: str, durations: Sequence[float]) -> list[tuple[float, float]]:
    """Start/end seconds of every space-separated run of phonemes.

    ``durations[0]`` is the BOS pad and ``durations[1 + i]`` covers
    ``units[i]``, so the running total is already the clip-relative time.

    The gap between two runs is silence that belongs to neither phoneme, and
    upstream's ``KPipeline.join_timestamps`` -- the path Kokoro-FastAPI uses for
    English voices -- splits it down the middle: a word ends halfway into the
    gap that follows it and the next word starts from that same midpoint.
    Ending a word at its last phoneme instead hands the whole gap to nobody and
    cuts every word short by half of it, which is 12-50 ms here, always in the
    same direction, and lands squarely in a caller's tail budget.

    The gap before the first run and the one after the last are not split:
    those are the BOS pad and the trailing silence, which no word owns.
    """
    raw: list[tuple[float, float]] = []
    now = durations[0]
    start: float | None = None

    for i, ch in enumerate(units):
        if ch.isspace():
            if start is not None:
                raw.append((start, now))
                start = None
        elif start is None:
            start = now
        now += durations[1 + i]

    if start is not None:
        raw.append((start, now))

    return [
        (
            (raw[i - 1][1] + start) / 2.0 if i > 0 else start,
            (end + raw[i + 1][0]) / 2.0 if i + 1 < len(raw) else end,
        )
        for i, (start, end) in enumerate(raw)
    ]


def _merge_groups(
    words: Sequence[str],
    groups: Sequence[tuple[float, float]],
    vocab: Mapping[str, int],
    phonemize_word: Callable[[str], str],
) -> list[tuple[float, float]] | None:
    """Fold multi-group words (numbers, abbreviations) back into one span each.

    Some words phonemise to several space-separated groups.  Re-phonemising
    each word on its own says how many groups it owns; if those counts do not
    add up to the groups we actually have, the mapping is untrustworthy.
    """
    counts = [len(phoneme_units(phonemize_word(word), vocab).split()) for word in words]
    if any(count == 0 for count in counts) or sum(counts) != len(groups):
        return None

    spans: list[tuple[float, float]] = []
    index = 0
    for count in counts:
        spans.append((groups[index][0], groups[index + count - 1][1]))
        index += count
    return spans


def word_timestamps(
    text: str,
    phonemes: str,
    pred_dur: Iterable[float],
    vocab: Mapping[str, int],
    phonemize_word: Callable[[str], str],
    offset: float = 0.0,
) -> list[WordTimestamp] | None:
    """Map *text*'s whitespace-separated words onto times in the rendered clip.

    Args:
        text: The graphemes this chunk was synthesized from.
        phonemes: The phoneme string handed to ``KokoroModel.forward``.
        pred_dur: The model's predicted per-token durations, BOS first.
        vocab: The model vocabulary, used to drop characters ``forward`` drops.
        phonemize_word: Phonemizes a single word, for group reconciliation.
        offset: Seconds to add, for a chunk that does not start the clip.

    Returns:
        One ``{"word", "start_time", "end_time"}`` mapping per word, in input
        order, with seconds rounded to milliseconds -- or ``None`` if the
        phoneme groups could not be mapped onto the words.
    """
    try:
        words = text.split()
        if not words:
            return []

        units = phoneme_units(phonemes, vocab)
        durations = [float(dur) * TS_SCALE for dur in pred_dur]
        # BOS + one per unit + EOS.
        if len(durations) < len(units) + 2:
            return None

        groups = _groups(units, durations)
        if len(groups) == len(words):
            spans: list[tuple[float, float]] | None = list(groups)
        else:
            spans = _merge_groups(words, groups, vocab, phonemize_word)
        if spans is None:
            return None

        return [
            {
                "word": word,
                "start_time": round(offset + start, 3),
                "end_time": round(offset + end, 3),
            }
            for word, (start, end) in zip(words, spans)
        ]
    except Exception:
        return None
