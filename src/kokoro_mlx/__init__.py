# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""kokoro-mlx: Kokoro TTS inference on Apple Silicon via MLX."""

__version__ = "0.2.1"

from .config import ISTFTNetConfig, KokoroConfig, PLBertConfig
from .kokoro import KokoroTTS, TTSResult
from .phonemize import Phonemizer, language_from_voice, normalize_language
from .timestamps import WordTimestamp, word_timestamps
from .voices import DEFAULT_VOICE, VoiceManager

__all__ = [
    "__version__",
    "ISTFTNetConfig",
    "PLBertConfig",
    "KokoroConfig",
    "KokoroTTS",
    "TTSResult",
    "Phonemizer",
    "language_from_voice",
    "normalize_language",
    "WordTimestamp",
    "word_timestamps",
    "VoiceManager",
    "DEFAULT_VOICE",
]
