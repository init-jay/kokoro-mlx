# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.2] - 2026-09-06

### Fixed

- Word `end_time` no longer lands systematically early. The silence between two words was attributed to neither of them, so every word ended at its last phoneme. `KPipeline.join_timestamps` upstream — the path Kokoro-FastAPI reports for English voices — splits that gap down the middle instead, and now so do we: a word ends halfway into the following silence and the next word starts there. Measured on `"hey seeree what is on tonight"` across the eight voices of the report, the median disagreement with Kokoro-FastAPI goes from **-34 ms to -9 ms** and loses its consistent sign. The leading pad and the trailing silence are still owned by no word.

## [0.2.1] - 2026-09-05

### Fixed

- `sample_rate` now actually resamples instead of relabelling. Only 48000 was handled; every other value returned native 24 kHz samples tagged with the requested rate, so 16 kHz output played 1.5x slow and a fifth low, and `save()` wrote a WAV whose header did not match its samples. Resampling is Fourier-method and numpy-only; downsampling is anti-aliased by construction.
- `TTSResult.duration` is consequently the true wall-clock length at every rate, and no longer disagrees with `timestamps`.
- Reject a `sample_rate` that is not a positive integer, rather than returning mislabelled audio.

## [0.2.0] - 2026-09-05

### Added

- Word timestamps, derived from the model's own predicted phoneme durations rather than estimated: `tts.generate(..., return_timestamps=True)` populates `TTSResult.timestamps` with one `{"word", "start_time", "end_time"}` mapping per whitespace-separated word of the input.
- `kokoro_mlx.timestamps` with the duration arithmetic, exported as `word_timestamps`. Returns `None` rather than a guess when the phoneme groups cannot be mapped onto the input words.
- `KokoroModel.forward(..., return_pred_dur=True)` also returns the predicted per-token durations the audio was rendered from.
- `Phonemizer.phonemize_chunks`, which keeps each chunk's source text alongside its phonemes.

### Fixed

- Point the model-backed tests in `tests/test_model.py` at the current user's HuggingFace cache instead of a hardcoded home directory.

## [0.1.2] - 2026-05-12

### Changed

- Require MLX 0.31+ and align runtime dependencies with the current MLX audio ecosystem.
- Constrain Python support to 3.10-3.12, matching the current `misaki[en]` support range.
- Infer G2P language from Kokoro voice prefixes, with an explicit `language` override on `generate`, `generate_stream`, `speak`, and `save`.
- Add optional Japanese and Mandarin G2P extras: `kokoro-mlx[ja]`, `kokoro-mlx[zh]`, and `kokoro-mlx[multilingual]`.
- Update README API and language selection docs.

## [0.1.0] - 2026-02-28

### Added

- Pure MLX implementation of Kokoro-82M text-to-speech for Apple Silicon. No PyTorch, no transformers, no third-party ML frameworks.
- ALBERT-based text encoder with 3 hidden layers, 768-dim embeddings, shared parameters across layers.
- Prosody predictor with BiLSTM backbone, duration and F0 estimation per phoneme.
- iSTFTNet vocoder: multi-scale decoder with SineGen excitation, AdaIN residual blocks, inverse STFT synthesis to 24 kHz float32 audio.
- WeightNormConv1d layers throughout the decoder stack for stable training-weight loading.
- G2P frontend via misaki: English phonemization with automatic sentence chunking at the 510-token limit.
- 54 built-in voices (American English, British English, and additional languages) with style vector management.
- Speed control for adjusting speech rate.
- Streaming synthesis: sentence-by-sentence generation for long text inputs.
- Automatic long-text chunking at sentence boundaries.
- WAV export via soundfile (24 kHz, float32).
- Audio playback via sounddevice (optional dependency).
- `KokoroTTS` public API: context manager, `from_pretrained`, `generate`, thread-safe via internal lock.
- KokoroConfig, ISTFTNetConfig, PLBertConfig dataclasses for full model configuration.
- 82 tests covering all modules.
