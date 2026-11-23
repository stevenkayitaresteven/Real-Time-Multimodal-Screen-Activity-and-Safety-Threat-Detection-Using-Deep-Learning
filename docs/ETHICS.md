# Responsible use, privacy and limitations

A system that watches a screen sees everything: messages, passwords being typed, medical and financial information.
Responsible design is not optional here. The same capability that protects a user from a phishing page could be
turned into stalkerware.

## Intended use
* **Self-protection**: a user runs it on their own device to get warnings about phishing, scams, malware lures and
  unwanted explicit content.
* **Transparent family or organisational safety**: the monitored person knows the tool is running and what it reports,
  following applicable law and organisational policy.
* **Research**: studying multimodal screen understanding on synthetic or consented data.

## Not intended for
* Covert monitoring of another person, an intimate partner, or employees without notice and consent. In many
  jurisdictions this is illegal.
* Making automated decisions with legal or disciplinary consequences.
* Detecting child sexual abuse material. That requires hash-matching services and legal reporting channels, not an
  NSFW classifier (see [DATASETS.md](DATASETS.md#nsfw-training)).

## Privacy-by-design defaults in this code
| Default | Where |
|---|---|
| All inference is on-device; no network calls | entire pipeline |
| Frames are never written to disk | `realtime/pipeline.py` (nothing saves images) |
| Event logs contain labels, scores, rule IDs and domains, **not screen text** | `JsonlSink` |
| Text logging is opt-in (`--log-text`) and PII-redacted (emails, phones, card numbers, wallets, passwords) | `privacy.py` |
| The monitor prints a visible notice when it starts | `cli.py monitor` |
| The NSFW head reports "unavailable" until trained on NSFW data, so it never emits uncalibrated guesses | `ScreenAnalyzer` |

## Known limitations
* **Synthetic-to-real gap.** Reported accuracies are on synthetic screens and are an upper bound. Expect lower numbers
  on real screens until you fine-tune on real data.
* **OCR dependence.** Threat detection relies heavily on text. Small fonts, unusual scripts, low contrast and
  non-English content reduce recall. The default model is trained on English text only.
* **Adversarial content.** Attackers can render text as images, use homoglyphs outside the normalisation table, or
  split cues across screens. The rule layer and lookalike detector are intentionally simple and inspectable, not
  complete.
* **Base rates.** Real threats are rare. A 1 % false-positive rate at 2 frames per second still produces many false
  alarms per day, which is why the alert hysteresis, cooldowns and change-based skipping matter. Tune thresholds for a
  target false-alarm budget.
* **Bias.** NSFW classifiers are known to show skin-tone and gender biases. Evaluate per subgroup before deployment.
  The synthetic generator does not cover this.
