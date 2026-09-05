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


def _resample_2x(audio: np.ndarray) -> np.ndarray:
    """Upsample audio by exactly 2x using FFT zero-padding.

    For a real signal of length N, the rfft has N//2+1 bins.  Padding the
    spectrum to 2x length and taking the irfft produces a perfectly
    bandlimited 2x-upsampled signal.  Numpy-only, no extra dependencies.
    """
    n = len(audio)
    spectrum = np.fft.rfft(audio)
    out_len = n * 2
    padded = np.zeros(out_len // 2 + 1, dtype=spectrum.dtype)
    padded[: len(spectrum)] = spectrum
    return np.fft.irfft(padded, n=out_len).astype(np.float32) * 2.0


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
        sample_rate: Output sample rate. 24000 (native) or 48000 (2x upsampled).
        return_timestamps: When True, return ``(audio, timestamps)`` where
            timestamps holds one ``{"word", "start_time", "end_time"}`` mapping
            per whitespace-separated word of *text*, or ``None`` if the words
            could not be mapped onto the phonemes.

    Returns:
        Float32 numpy array of audio samples at the requested sample rate, or
        ``(audio, timestamps)`` when *return_timestamps*.
    """
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

    if sample_rate == 48000 and len(result) > 0:
        result = _resample_2x(result)

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
        sample_rate: Output sample rate. 24000 (native) or 48000 (2x upsampled).

    Yields:
        Float32 numpy arrays, one per sentence chunk.
    """
    if phonemizer is None:
        phonemizer = Phonemizer(config.vocab)

    chunks = phonemizer.phonemize_long(text)
    if not chunks:
        return

    voice_array = voice_manager.load_voice(voice)
    upsample = sample_rate == 48000

    for phonemes, token_ids in chunks:
        style = voice_manager.get_style(voice_array, len(token_ids))
        audio = model.forward(phonemes, style, speed)
        chunk = np.array(audio.tolist(), dtype=np.float32)
        if upsample and len(chunk) > 0:
            chunk = _resample_2x(chunk)
        yield chunk
