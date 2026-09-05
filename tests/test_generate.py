# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Tests for the generate pipeline and playback utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from kokoro_mlx.generate import SAMPLE_RATE

_MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16/snapshots/a71e4d38b236d968966a2002c4c895dbd12b1c3c"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    from kokoro_mlx.model import KokoroModel

    return KokoroModel.from_pretrained(_MODEL_PATH)


@pytest.fixture(scope="module")
def config():
    from kokoro_mlx.config import KokoroConfig

    return KokoroConfig.from_pretrained(_MODEL_PATH)


@pytest.fixture(scope="module")
def voice_manager():
    from kokoro_mlx.voices import VoiceManager

    return VoiceManager(_MODEL_PATH)


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGenerate:
    def test_returns_float32_array(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("Hello, world.", model, config, voice_manager)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32

    def test_audio_has_samples(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("Hello.", model, config, voice_manager)
        assert audio.shape[0] > 0

    def test_empty_text_returns_empty_array(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("", model, config, voice_manager)
        assert isinstance(audio, np.ndarray)
        assert audio.shape[0] == 0

    def test_whitespace_only_returns_empty_array(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("   ", model, config, voice_manager)
        assert isinstance(audio, np.ndarray)
        assert audio.shape[0] == 0

    def test_no_nan_in_output(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("Testing audio quality.", model, config, voice_manager)
        assert not np.any(np.isnan(audio))

    def test_speed_affects_length(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        text = "The quick brown fox jumps over the lazy dog."
        slow = generate(text, model, config, voice_manager, speed=0.5)
        fast = generate(text, model, config, voice_manager, speed=2.0)
        # Slower speech produces more frames.
        assert slow.shape[0] > fast.shape[0]


# ---------------------------------------------------------------------------
# _resample()
# ---------------------------------------------------------------------------


class TestResample:
    """No model weights needed: this is arithmetic on arrays."""

    @staticmethod
    def _tone(freq: float, secs: float, rate: int) -> np.ndarray:
        t = np.arange(int(secs * rate)) / rate
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    def test_length_follows_the_rate_ratio(self):
        from kokoro_mlx.generate import _resample

        audio = self._tone(440, 1.0, 24000)
        assert len(_resample(audio, 24000, 16000)) == 16000
        assert len(_resample(audio, 24000, 48000)) == 48000
        assert len(_resample(audio, 24000, 22050)) == 22050

    def test_same_rate_is_a_passthrough(self):
        from kokoro_mlx.generate import _resample

        audio = self._tone(440, 0.1, 24000)
        np.testing.assert_array_equal(_resample(audio, 24000, 24000), audio)

    def test_downsampling_preserves_the_tone(self):
        from kokoro_mlx.generate import _resample

        out = _resample(self._tone(440, 0.5, 24000), 24000, 16000)
        # The 440 Hz peak must survive at the same frequency and amplitude.
        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.argmax(spectrum) * 16000 / len(out)
        assert abs(peak_hz - 440) < 5
        assert 0.9 < np.abs(out[1000:-1000]).max() < 1.1

    def test_downsampling_does_not_alias(self):
        from kokoro_mlx.generate import _resample

        # 10 kHz is above the 8 kHz Nyquist of a 16 kHz signal. Truncating the
        # spectrum must remove it, not fold it down to 6 kHz.
        out = _resample(self._tone(10000, 0.5, 24000), 24000, 16000)
        assert np.abs(out[500:-500]).max() < 0.05

    def test_upsampling_matches_the_previous_2x_path(self):
        from kokoro_mlx.generate import _resample

        audio = self._tone(440, 0.25, 24000)
        n = len(audio)
        spectrum = np.fft.rfft(audio)
        padded = np.zeros(n + 1, dtype=spectrum.dtype)
        padded[: len(spectrum)] = spectrum
        expected = np.fft.irfft(padded, n=n * 2).astype(np.float32) * 2.0
        np.testing.assert_allclose(_resample(audio, 24000, 48000), expected, atol=1e-6)

    def test_empty_input(self):
        from kokoro_mlx.generate import _resample

        out = _resample(np.array([], dtype=np.float32), 24000, 16000)
        assert out.shape[0] == 0

    def test_output_is_float32(self):
        from kokoro_mlx.generate import _resample

        assert _resample(self._tone(440, 0.1, 24000), 24000, 16000).dtype == np.float32


class TestSampleRateValidation:
    def test_rejects_non_positive(self):
        from kokoro_mlx.generate import _validate_sample_rate

        for bad in (0, -24000):
            with pytest.raises(ValueError):
                _validate_sample_rate(bad)

    def test_rejects_non_integer(self):
        from kokoro_mlx.generate import _validate_sample_rate

        for bad in (24000.0, "24000", None):
            with pytest.raises(ValueError):
                _validate_sample_rate(bad)

    def test_accepts_ordinary_rates(self):
        from kokoro_mlx.generate import _validate_sample_rate

        for good in (8000, 16000, 22050, 24000, 44100, 48000):
            _validate_sample_rate(good)


# ---------------------------------------------------------------------------
# generate(return_timestamps=True)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGenerateTimestamps:
    TEXT = "hey seeree what is on tonight"

    def test_returns_audio_and_timestamps(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio, timestamps = generate(self.TEXT, model, config, voice_manager, return_timestamps=True)
        assert isinstance(audio, np.ndarray)
        assert timestamps is not None

    def test_one_entry_per_word_in_order(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        _, timestamps = generate(self.TEXT, model, config, voice_manager, return_timestamps=True)
        assert [entry["word"] for entry in timestamps] == self.TEXT.split()

    def test_times_are_non_decreasing(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        _, timestamps = generate(self.TEXT, model, config, voice_manager, return_timestamps=True)
        times = [t for entry in timestamps for t in (entry["start_time"], entry["end_time"])]
        assert times == sorted(times)

    def test_last_word_ends_within_the_clip(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio, timestamps = generate(self.TEXT, model, config, voice_manager, return_timestamps=True)
        duration = len(audio) / SAMPLE_RATE
        end = timestamps[-1]["end_time"]
        # The clip ends with the EOS pad's silence, so the last word ends a
        # little before it -- but never after.
        assert end <= duration
        assert duration - end < 0.5

    def test_durations_account_for_the_whole_clip(self, model, config, voice_manager):
        """The predicted durations the times come from must sum to the audio."""
        from kokoro_mlx.phonemize import Phonemizer
        from kokoro_mlx.timestamps import TS_SCALE

        phonemizer = Phonemizer(config.vocab, language="en-us")
        phonemes, token_ids = phonemizer.phonemize(self.TEXT)
        voice_array = voice_manager.load_voice("af_bella")
        style = voice_manager.get_style(voice_array, len(token_ids))

        audio, pred_dur = model.forward(phonemes, style, 1.0, return_pred_dur=True)
        samples = np.array(audio.tolist(), dtype=np.float32)
        assert abs(float(pred_dur.sum()) * TS_SCALE - len(samples) / SAMPLE_RATE) < 1e-3

    def test_words_spanning_several_phoneme_groups(self, model, config, voice_manager):
        """Numbers and abbreviations phonemise to multiple groups; still one entry each."""
        from kokoro_mlx.generate import generate

        text = "call 555 1234 now"
        _, timestamps = generate(text, model, config, voice_manager, return_timestamps=True)
        assert timestamps is not None
        assert [entry["word"] for entry in timestamps] == text.split()

    def test_multi_chunk_text_offsets_each_chunk(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        # Long enough to exceed the 510-token window and split into chunks.
        text = " ".join(f"The quick brown fox jumps over the lazy dog number {i}." for i in range(12))
        audio, timestamps = generate(text, model, config, voice_manager, return_timestamps=True)
        assert timestamps is not None
        assert [entry["word"] for entry in timestamps] == text.split()
        starts = [entry["start_time"] for entry in timestamps]
        assert starts == sorted(starts)
        assert timestamps[-1]["end_time"] <= len(audio) / SAMPLE_RATE

    def test_speed_is_already_in_the_durations(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio, timestamps = generate(self.TEXT, model, config, voice_manager, speed=1.5, return_timestamps=True)
        assert timestamps[-1]["end_time"] <= len(audio) / SAMPLE_RATE

    def test_times_stay_in_seconds_at_every_rate(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        native = generate(self.TEXT, model, config, voice_manager, return_timestamps=True)[1]
        for rate in (16000, 48000):
            other = generate(self.TEXT, model, config, voice_manager, sample_rate=rate, return_timestamps=True)[1]
            assert native == other

    def test_timestamps_fit_the_clip_at_every_rate(self, model, config, voice_manager):
        """The bug this guards: 16 kHz used to return 24 kHz samples relabelled."""
        from kokoro_mlx.generate import generate

        for rate in (16000, 24000, 48000):
            audio, timestamps = generate(
                self.TEXT, model, config, voice_manager, sample_rate=rate, return_timestamps=True
            )
            assert timestamps[-1]["end_time"] <= len(audio) / rate

    def test_empty_text_returns_no_timestamps(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio, timestamps = generate("", model, config, voice_manager, return_timestamps=True)
        assert audio.shape[0] == 0
        assert timestamps == []

    def test_plain_call_still_returns_a_bare_array(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        assert isinstance(generate(self.TEXT, model, config, voice_manager), np.ndarray)

    def test_audio_is_unchanged_by_the_request(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        plain = generate(self.TEXT, model, config, voice_manager)
        with_ts, _ = generate(self.TEXT, model, config, voice_manager, return_timestamps=True)
        # The vocoder's excitation noise makes samples differ run to run, but
        # the durations that set the length are deterministic.
        assert plain.shape == with_ts.shape
        assert plain.dtype == with_ts.dtype


# ---------------------------------------------------------------------------
# generate_stream()
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGenerateStream:
    def test_yields_chunks(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate_stream

        chunks = list(generate_stream("Hello. World.", model, config, voice_manager))
        assert len(chunks) >= 1

    def test_each_chunk_is_float32_array(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate_stream

        for chunk in generate_stream("Hello. World.", model, config, voice_manager):
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32

    def test_empty_text_yields_nothing(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate_stream

        chunks = list(generate_stream("", model, config, voice_manager))
        assert chunks == []

    def test_concatenated_matches_generate(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate, generate_stream

        text = "One sentence only."
        full = generate(text, model, config, voice_manager)
        streamed = np.concatenate(list(generate_stream(text, model, config, voice_manager)))
        # Same text produces the same number of samples and the same dtype.
        assert full.shape == streamed.shape
        assert full.dtype == streamed.dtype


# ---------------------------------------------------------------------------
# save_wav()
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSaveWav:
    def test_creates_valid_wav_file(self, model, config, voice_manager):
        import soundfile as sf

        from kokoro_mlx.generate import generate
        from kokoro_mlx.playback import save_wav

        audio = generate("Saving audio to disk.", model, config, voice_manager)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.wav"
            save_wav(audio, path)

            assert path.exists()
            assert path.stat().st_size > 0

            data, sr = sf.read(str(path))
            assert sr == 24000
            assert len(data) == len(audio)

    def test_roundtrip_preserves_values(self, model, config, voice_manager):
        import soundfile as sf

        from kokoro_mlx.generate import generate

        audio = generate("Round trip test.", model, config, voice_manager)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rt.wav"
            # Write as 32-bit float to avoid PCM_16 quantization loss.
            sf.write(str(path), audio, 24000, subtype="FLOAT")
            data, _ = sf.read(str(path), dtype="float32")
            np.testing.assert_allclose(audio, data, atol=1e-6)
