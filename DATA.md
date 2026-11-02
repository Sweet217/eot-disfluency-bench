# Data

No audio is committed to this repository. `make data` regenerates or downloads
everything into `data/` (git-ignored). Only public or synthetic sources are used:
no customer, broadcast or private recordings, and no paid APIs.

## 1. EdAcc: real accented conversational English

* **Source:** The Edinburgh International Accents of English Corpus,
  Hugging Face release [`edinburghcstr/edacc`](https://huggingface.co/datasets/edinburghcstr/edacc).
  Paper: Sanabria et al., *The Edinburgh International Accents of English Corpus:
  Towards the Democratization of English ASR*, ICASSP 2023 (arXiv:2303.18110).
* **License:** CC BY-SA 4.0, as stated on the dataset card ("Public Domain, Creative
  Commons Attribution-ShareAlike International Public License"). This repository
  redistributes none of the audio or transcripts; it downloads them for local
  use. Derived per-item metadata stays in `data/` and is not committed.
  If you redistribute derived audio, the ShareAlike terms apply.
* **What is downloaded:** row group 2 (100 segments) of each of the 16 parquet
  shards (10 test + 6 validation), fetched with HTTP range requests:
  **586 MB** on disk, about 8% of the 6.95 GB release.
* **Items built:** 500 turn-final segments from 20 dyadic Zoom conversations,
  39 speakers, 0.9 h of turn speech (1.53 h with the appended room tone).
  21 distinct values of the corpus `l1` field (e.g. Nigerian English, Sinhalese, Southern British English,
  Arabic, Italian, Kenyan English, Urdu, Mandarin, Hebrew, Spanish, Vietnamese, Polish,
  Russian, Romanian, Lithuanian, Hindi).
  Accent groups use the corpus `l1` field: `edacc_L1_english` (an English variety is
  the L1; 181 items) vs `edacc_L2_english` (319 items).
* **Turn-end ground truth.** The HF release has one row per transcribed speaker
  segment in conversation order, without absolute times. A segment becomes an
  item only if it is not a backchannel, has at least 4 words, lasts 1.5 to 25 s, and
  the *next* segment is by the other speaker and is not a backchannel either
  (the floor really changed hands). Segments followed by a backchannel are
  dropped as ambiguous. Inside a kept segment, the turn end is the last speech
  frame and intra-turn pauses are internal low-energy runs of at least 150 ms,
  found by an offline, non-causal energy detector (10 ms frames, threshold
  `max(p2 + 12 dB, p98 - 40 dB)`, gaps < 60 ms closed, bursts < 50 ms removed)
  that shares nothing with the systems under test. 0.5 s of the segment's own
  room tone is prepended and 4 s appended.
* **Word timings for the oracle-text ablation.** The gold transcript is aligned
  to faster-whisper `small.en` word timestamps on the full segment
  (`difflib` matching; unmatched words linearly interpolated).
* **Split:** conversation-disjoint, hence speaker-disjoint. A deterministic search
  balances item count and accent group: train 250 items / 21 speakers,
  test 250 items / 18 speakers, each with 36% L1-English and 64% L2-English items.
* **Caveats:** labels come from an energy detector, so turn ends can be off by
  tens of milliseconds and breaths or partner crosstalk can be counted as speech.
  The appended tail is synthetic continuation of room tone; in the real
  conversation the partner started speaking there.

## 2. Synthetic controlled set (macOS `say`)

* **Generator:** `eotbench/synth.py`, sentences in `eotbench/sentences.py`
  (80 invented, generic assistant-style requests written for this project).
* **Construction:** each turn is `[0.5 s noise][part A][pause][part B][4 s noise]`.
  Part A is the prefix up to a marked split point, optionally followed by a filler
  (`um`, `uh`, `so`, `and then`, `like`, `you know`; 60% of items), synthesised
  with trailing continuation punctuation. The pause is 300, 500, 800, 1200, 1600 or
  2000 ms. 15% of items are controls without an inserted pause. Split points are
  either mid-clause (`|`, e.g. after a preposition) or at a clause boundary where
  the prefix is already a complete sentence (`#`, the hard case for text models).
  Pink noise at 20, 30 or 40 dB SNR; speech level -23 dBFS +- 3 dB.
* **Ground truth** is exact by construction: inserted pause boundaries and turn
  end are sample positions (10 ms trimming resolution); verified by
  `tests/test_synth.py`.
* **Voices and accent groups:** 43 macOS voices in three groups:
  `en_native` (16 US/UK/AU/IE/ZA voices), `en_IN` (3 Indian English voices) and
  `tts_nonnative_approx` (24 voices whose locale is not English, e.g. Italian,
  German, French, Spanish, Portuguese, Japanese, Mandarin, Korean, Russian,
  Polish, Turkish, Hindi, Vietnamese, Romanian, Hebrew, Arabic, Dutch, reading
  English text). **The last group is only an approximation of non-native accents:**
  it is a TTS engine applying another language's phonology, not L2 speech, and
  its pauses and prosody are produced by the same synthesiser.
* **Split:** voice-disjoint (every third voice per group is test) *and*
  sentence-disjoint (24 of 80 sentences are test-only), so neither voice nor
  wording leaks between train and test.
* **Size:** 1,000 train items (29 voices) + 480 test items (14 voices),
  3.74 h total, 482 MB as 16 kHz WAV plus 8 kHz copies of the test items.

## 3. Telephony variant

For every test item an 8 kHz copy is made (`eotbench/audio.py:telephony`):
4th-order 300-3400 Hz band-pass, resampling to 8 kHz, and 8-bit mu-law
(G.711-style) companding. All systems receive it upsampled back to 16 kHz.
Models are trained on 16 kHz audio only, so the 8 kHz results measure robustness
to a channel mismatch.

## 4. Models downloaded at run time

| model | use | license |
|---|---|---|
| Silero VAD v6 (`silero-vad` pip package) | VAD for silence baselines and all candidate pauses | MIT |
| `webrtcvad-wheels` (WebRTC VAD) | second silence baseline | BSD-style (WebRTC) / MIT (bindings) |
| `pipecat-ai/smart-turn-v3`, file `smart-turn-v3.2-cpu.onnx` | semantic/audio turn baseline, run on CPU with onnxruntime | BSD-2-Clause |
| faster-whisper `base.en` (int8, CPU) | prefix ASR for the proposed detector | MIT |
| faster-whisper `small.en` | word-time alignment of EdAcc gold transcripts (data prep only) | MIT |
| `distilgpt2` | sentence-end probability feature | Apache-2.0 |
