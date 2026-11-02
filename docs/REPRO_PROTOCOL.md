# Manual protocol: does a voice assistant end my turn too early?

This is a protocol a person can follow by hand to check whether a given voice
assistant (any product, any vendor) ends the user's turn while the user is
still speaking. It involves **no automated interaction with any product**: a
human speaks, a human observes, and a human writes down what happened. It makes
no claims about any particular product; it only describes how to collect
evidence in a way others can repeat.

## What you need

* The assistant under test, in its normal voice mode, on a device you own.
* A second device to record the session (a phone voice-memo app is enough),
  placed near the speaker, so pause lengths and cut-off moments can be measured
  later. Check the assistant's terms of use and local law before recording, and
  only record your own voice.
* A quiet room (and, optionally, a second run with steady background noise).
* A stopwatch or a metronome app to pace pauses (a silent visual timer works
  best; an audible click would be picked up by the assistant).
* The prompt sheet below, printed or on a separate screen.

## Prompt sheet

Each prompt is read aloud. `[P]` marks a pause you hold silently, without
fillers, for the length given in the session plan; `um`, `so...` and
`and then...` are spoken fillers. Prompts are harmless everyday requests.

| ID | Prompt | Pause type |
|----|--------|-----------|
| A1 | "I'd like a recipe for [P] a vegetarian lasagna that serves six." | mid-clause, after preposition |
| A2 | "Can you remind me to call [P] my sister when I get home?" | mid-clause, after verb |
| A3 | "Explain the difference between [P] weather and climate." | mid-clause, after preposition |
| B1 | "What's a good name for, um, [P] a small grey cat?" | filler + pause |
| B2 | "I need ideas for a party, so... [P] something for ten-year-olds." | conjunction filler + pause |
| B3 | "First I boiled the pasta and then... [P] what should I do with the sauce?" | "and then" + pause |
| C1 | "Set a timer for ten minutes. [P] And another one for twenty." | complete clause, then continuation |
| C2 | "Tell me a fun fact. [P] About octopuses, please." | complete clause, then continuation |
| D1 | "What's the capital of Australia?" | control, no pause |
| D2 | "Give me three tips for sleeping better." | control, no pause |

If English is not your first language, or you naturally pause to search for
words, read the prompts in your natural way as well; note this in the log
(speaker profile), because such speakers are exactly the population this
protocol is meant to cover.

## Session plan

1. Start the recording device, then say the session ID and date aloud.
2. Pause lengths: run every A/B/C prompt at **0.5 s, 1.0 s, 1.5 s and 2.5 s**
   (use the timer; start counting when your voice stops). Controls D1 and D2
   run once per block.
3. Randomise the order of prompts and pause lengths within each block (for
   example by shuffling index cards), so learning effects do not line up with
   pause length.
4. Run each (prompt, pause) pair **three times** in separate assistant
   sessions or after the assistant has finished replying, so one cut-off does
   not change the next trial.
5. Optionally repeat the whole block in a different condition (background
   noise, phone speaker vs. headset, a different device), recording the
   condition in the log.

## What to record per trial

| field | how to fill it |
|-------|----------------|
| trial ID, prompt ID, target pause (s) | from the plan |
| outcome | `complete` (assistant waited for the whole prompt), `cut_off` (assistant started replying, or visibly stopped listening, before the prompt was finished), `no_response` (nothing within 5 s after you finished), `other` |
| where it cut off | the last word you had said before the assistant took the turn |
| response delay (s) | from the end of your last word to the start of the assistant's reply, measured afterwards on the recording (a waveform editor such as Audacity works) |
| notes | anything unusual: you misspoke, background noise, assistant asked a clarification question, etc. |

A CSV template:

```
trial_id,prompt_id,target_pause_s,condition,outcome,last_word_before_cutoff,response_delay_s,notes
```

## Analysis

* **Cut-off rate per pause length**: `cut_off / (complete + cut_off)` for each
  target pause, with the number of trials next to it. Small numbers of trials
  give wide uncertainty; report counts, not just percentages (a Wilson interval
  is a good default).
* **Response delay** on `complete` trials: median and the slowest trial.
* Compare pause types (mid-clause vs. filler vs. complete-clause) at the same
  pause length: a system that uses only silence should behave the same across
  types; a system that uses content or prosody should treat mid-clause and
  filler pauses differently from complete-clause pauses.

## Reporting guidance

* Report exactly what was run: device, app version, date, condition, speaker
  profile, number of trials. Do not generalise beyond that.
* A cut-off observed in a few trials is evidence of behaviour under these
  conditions, not proof of a systematic defect; a product's behaviour can
  change between versions and settings.
* If you share results with a vendor, share the recordings of your own voice
  and the log so the behaviour can be reproduced; keep the tone factual.
