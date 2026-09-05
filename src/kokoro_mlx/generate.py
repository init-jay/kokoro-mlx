# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Text-to-audio generation pipeline for Kokoro TTS."""

from __future__ import annotations

import numpy as np

from .config import KokoroConfig
from .model import KokoroModel
from .phonemize import Phonemizer
from .timestamps import WordTimestamp, word_timestamps
from .voices import VoiceManager

SAMPLE_RATE = 24000


def _validate_sample_rate(sample_rate: int) -> None:
    """Reject a rate we cannot honestly resample to."""
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio between two rates using the Fourier method.

    For a real signal of length N, the rfft has N//2+1 bins.  Zero-padding that
    spectrum interpolates; truncating it decimates, and the truncation is
    itself an ideal low-pass at the new Nyquist, so downsampling needs no
    separate anti-aliasing filter.  Numpy-only, no extra dependencies.
    """
    n = len(audio)
    if n == 0 or orig_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    out_len = int(round(n * target_rate / orig_rate))
    if out_len < 1:
        return np.array([], dtype=np.float32)

    spectrum = np.fft.rfft(audio)
    resized = np.zeros(out_len // 2 + 1, dtype=spectrum.dtype)
    keep = min(len(spectrum), len(resized))
    resized[:keep] = spectrum[:keep]

    # A real signal's Nyquist bin must be real. When decimating, the bin we
    # truncated at is an interior bin of the original spectrum and generally
    # is not.
    if out_len % 2 == 0 and len(spectrum) > len(resized):
        resized[-1] = resized[-1].real

    return np.fft.irfft(resized, n=out_len).astype(np.float32) * (out_len / n)


def generate(
    text: str,
    model: KokoroModel,
    config: KokoroConfig,
    voice_manager: VoiceManager,
    voice: str = "af_heart",
    speed: float = 1.0,
    phonemizer: Phonemizer | None = None,
    sample_rate: int = SAMPLE_RATE,
    return_timestamps: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[WordTimestamp] | None]:
    """Full text-to-audio pipeline.

    Args:
        text: Input text to synthesize.
        model: Loaded KokoroModel.
        config: KokoroConfig instance.
        voice_manager: VoiceManager instance.
        voice: Voice name to use for synthesis.
        speed: Speaking rate multiplier (>1 is faster, <1 is slower).
        phonemizer: Optional pre-built Phonemizer to avoid re-initializing.
        sample_rate: Output sample rate. The model renders at 24000; any other
            rate is resampled, so the returned audio is always genuinely at
            *sample_rate* and ``len(audio) / sample_rate`` is its true duration.
        return_timestamps: When True, return ``(audio, timestamps)`` where
            timestamps holds one ``{"word", "start_time", "end_time"}`` mapping
            per whitespace-separated word of *text*, or ``None`` if the words
            could not be mapped onto the phonemes.

    Returns:
        Float32 numpy array of audio samples at the requested sample rate, or
        ``(audio, timestamps)`` when *return_timestamps*.
    """
    _validate_sample_rate(sample_rate)

    if phonemizer is None:
        phonemizer = Phonemizer(config.vocab)

    chunks = phonemizer.phonemize_chunks(text)
    if not chunks:
        empty = np.array([], dtype=np.float32)
        return (empty, []) if return_timestamps else empty

    voice_array = voice_manager.load_voice(voice)

    audio_chunks = []
    timestamps: list[WordTimestamp] | None = [] if return_timestamps else None
    elapsed = 0.0

    for chunk_text, phonemes, token_ids in chunks:
        style = voice_manager.get_style(voice_array, len(token_ids))
        if return_timestamps:
            audio, pred_dur = model.forward(phonemes, style, speed, return_pred_dur=True)
        else:
            audio = model.forward(phonemes, style, speed)
        samples = np.array(audio.tolist(), dtype=np.float32)
        audio_chunks.append(samples)

        if timestamps is not None:
            chunk_timestamps = word_timestamps(
                text=chunk_text,
                phonemes=phonemes,
                pred_dur=pred_dur,
                vocab=model.vocab,
                phonemize_word=lambda word: phonemizer.phonemize(word)[0],
                offset=elapsed,
            )
            # One bad chunk poisons the whole clip: later times are still
            # right, but the word list no longer lines up with the input.
            timestamps = None if chunk_timestamps is None else timestamps + chunk_timestamps

        # Chunks are concatenated at the native rate, so this is the offset of
        # the next chunk in seconds regardless of any later upsampling.
        elapsed += len(samples) / SAMPLE_RATE

    result = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
    result = _resample(result, SAMPLE_RATE, sample_rate)

    if not return_timestamps:
        return result

    if timestamps is not None and len(timestamps) != len(text.split()):
        timestamps = None
    return result, timestamps


def generate_stream(
    text: str,
    model: KokoroModel,
    config: KokoroConfig,
    voice_manager: VoiceManager,
    voice: str = "af_heart",
    speed: float = 1.0,
    phonemizer: Phonemizer | None = None,
    sample_rate: int = SAMPLE_RATE,
):
    """Generate audio in chunks as they are produced.

    Yields float32 numpy arrays, one per phoneme chunk. Suitable for
    low-latency streaming playback.

    Args:
        text: Input text to synthesize.
        model: Loaded KokoroModel.
        config: KokoroConfig instance.
        voice_manager: VoiceManager instance.
        voice: Voice name to use for synthesis.
        speed: Speaking rate multiplier.
        phonemizer: Optional pre-built Phonemizer to avoid re-initializing.
        sample_rate: Output sample rate. The model renders at 24000; any other
            rate is resampled chunk by chunk.

    Yields:
        Float32 numpy arrays, one per sentence chunk.
    """
    _validate_sample_rate(sample_rate)

    if phonemizer is None:
        phonemizer = Phonemizer(config.vocab)

    chunks = phonemizer.phonemize_long(text)
    if not chunks:
        return

    voice_array = voice_manager.load_voice(voice)

    for phonemes, token_ids in chunks:
        style = voice_manager.get_style(voice_array, len(token_ids))
        audio = model.forward(phonemes, style, speed)
        chunk = np.array(audio.tolist(), dtype=np.float32)
        yield _resample(chunk, SAMPLE_RATE, sample_rate)
