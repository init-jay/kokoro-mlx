# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Tests for word timestamp derivation. No model weights required."""

from __future__ import annotations

from kokoro_mlx.timestamps import TS_SCALE, phoneme_units, word_timestamps

# A toy vocab: every phoneme character used below, plus the space that
# separates words.  '?' is deliberately absent so the filtering path is tested.
_VOCAB = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}


def _durations(*per_unit: int) -> list[int]:
    """BOS + one duration per phoneme unit + EOS, all in pred_dur units."""
    return [0, *per_unit, 0]


def _identity_phonemes(word: str) -> str:
    return word


class TestPhonemeUnits:
    def test_drops_characters_outside_the_vocab(self):
        assert phoneme_units("ab?c", _VOCAB) == "abc"

    def test_keeps_spaces(self):
        assert phoneme_units("ab cd", _VOCAB) == "ab cd"

    def test_empty_string(self):
        assert phoneme_units("", _VOCAB) == ""


class TestWordTimestamps:
    def test_one_entry_per_word(self):
        result = word_timestamps("ab cd", "ab cd", _durations(1, 1, 1, 1, 1), _VOCAB, _identity_phonemes)
        assert [entry["word"] for entry in result] == ["ab", "cd"]

    def test_times_follow_the_durations(self):
        # units "ab cd": a=2, b=2, space=1, c=3, d=3 (BOS=0).
        result = word_timestamps("ab cd", "ab cd", _durations(2, 2, 1, 3, 3), _VOCAB, _identity_phonemes)
        assert result[0]["start_time"] == 0.0
        # The space is split down the middle, as KPipeline.join_timestamps does:
        # "ab" runs half a space past its last phoneme and "cd" starts there.
        assert result[0]["end_time"] == round(4.5 * TS_SCALE, 3)
        assert result[1]["start_time"] == round(4.5 * TS_SCALE, 3)
        assert result[1]["end_time"] == round(11 * TS_SCALE, 3)

    def test_words_meet_at_the_midpoint_of_every_gap(self):
        # A long space between two words: neither owns it, both reach halfway.
        result = word_timestamps("ab cd", "ab cd", _durations(1, 1, 8, 1, 1), _VOCAB, _identity_phonemes)
        assert result[0]["end_time"] == round(6 * TS_SCALE, 3)
        assert result[1]["start_time"] == round(6 * TS_SCALE, 3)

    def test_leading_and_trailing_silence_belong_to_no_word(self):
        # The BOS pad and the tail past the last phoneme are not split: the
        # first word starts after the whole pad and the last ends at its own
        # final phoneme.
        result = word_timestamps("ab cd", "ab cd", _durations(1, 1, 2, 1, 1), _VOCAB, _identity_phonemes)
        assert result[0]["start_time"] == 0.0
        assert result[1]["end_time"] == round(6 * TS_SCALE, 3)

    def test_bos_duration_shifts_the_first_word(self):
        durations = [4, 1, 1, 0]
        result = word_timestamps("ab", "ab", durations, _VOCAB, _identity_phonemes)
        assert result[0]["start_time"] == round(4 * TS_SCALE, 3)

    def test_times_are_monotonic(self):
        result = word_timestamps(
            "ab cd ef", "ab cd ef", _durations(1, 2, 1, 3, 1, 1, 2, 4), _VOCAB, _identity_phonemes
        )
        times = [t for entry in result for t in (entry["start_time"], entry["end_time"])]
        assert times == sorted(times)

    def test_offset_shifts_every_entry(self):
        durations = _durations(1, 1, 1, 1, 1)
        plain = word_timestamps("ab cd", "ab cd", durations, _VOCAB, _identity_phonemes)
        shifted = word_timestamps("ab cd", "ab cd", durations, _VOCAB, _identity_phonemes, offset=1.5)
        for before, after in zip(plain, shifted):
            assert after["start_time"] == round(before["start_time"] + 1.5, 3)
            assert after["end_time"] == round(before["end_time"] + 1.5, 3)

    def test_characters_outside_the_vocab_do_not_shift_times(self):
        # '?' never reached the model, so it owns no duration.
        result = word_timestamps("ab cd", "ab? cd", _durations(1, 1, 1, 1, 1), _VOCAB, _identity_phonemes)
        assert result[1]["start_time"] == round(2.5 * TS_SCALE, 3)

    def test_empty_text_returns_empty_list(self):
        assert word_timestamps("", "", [0, 0], _VOCAB, _identity_phonemes) == []

    def test_rounds_to_milliseconds(self):
        result = word_timestamps("ab", "ab", _durations(7, 7), _VOCAB, _identity_phonemes)
        for entry in result:
            assert entry["start_time"] == round(entry["start_time"], 3)
            assert entry["end_time"] == round(entry["end_time"], 3)


class TestGroupReconciliation:
    def test_word_spanning_several_groups_is_merged(self):
        # "ten" phonemises to two space-separated groups, as numbers do.
        def phonemize(word: str) -> str:
            return "aa bb" if word == "ten" else word

        result = word_timestamps("ten cd", "aa bb cd", _durations(1, 1, 1, 1, 1, 1, 1, 1), _VOCAB, phonemize)
        assert [entry["word"] for entry in result] == ["ten", "cd"]
        # The merged span runs from the first group's start to the last's end,
        # so the gap inside "ten" is absorbed and only the gap before "cd" splits.
        assert result[0]["start_time"] == 0.0
        assert result[0]["end_time"] == round(5.5 * TS_SCALE, 3)
        assert result[1]["start_time"] == round(5.5 * TS_SCALE, 3)

    def test_returns_none_when_counts_still_disagree(self):
        # Re-phonemising says one group per word, but there are three groups.
        result = word_timestamps("ab cd", "aa bb cc", _durations(1, 1, 1, 1, 1, 1, 1, 1), _VOCAB, _identity_phonemes)
        assert result is None

    def test_returns_none_when_a_word_phonemises_to_nothing(self):
        result = word_timestamps("ab cd", "aa", _durations(1, 1), _VOCAB, lambda word: "")
        assert result is None

    def test_returns_none_when_durations_are_too_short(self):
        assert word_timestamps("ab cd", "ab cd", [0, 1, 1], _VOCAB, _identity_phonemes) is None

    def test_returns_none_rather_than_raising(self):
        # A group/word mismatch sends this through the phonemizer, which blows up.
        def phonemize(word: str) -> str:
            raise RuntimeError("g2p exploded")

        assert word_timestamps("ab", "aa bb", _durations(1, 1, 1, 1, 1), _VOCAB, phonemize) is None
