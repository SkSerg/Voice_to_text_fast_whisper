# Voice to Text (Windows, local Faster-Whisper)

Local speech-to-text script for Windows that:
- captures audio from your microphone,
- transcribes speech with `faster-whisper` locally,
- types recognized text into the currently active app (Telegram, Word, VS Code, etc.).

The script supports:
- GPU inference (`CUDA`) when available,
- pause/resume with global hotkeys,
- automatic chunking by speech pauses,
- async processing to reduce audio overflows.

## 1. Requirements

- Windows 10/11
- Python 3.10+
- Working microphone
- (Optional, recommended) NVIDIA GPU + CUDA for speed

## 2. Installation

```powershell
git clone <your-repo-url>
cd <repo-folder>

python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run

```powershell
python main.py
```

At startup, script is paused:
- `F9` -> start/pause
- `F10` -> quit

## 4. How output works

- Default mode: `output_mode = "active_window"`
- The script types text into the app that currently has keyboard focus.
- By default, `add_newline = False`, so it does not press Enter automatically.

If you want console output for debugging:
- set `output_mode = "console"` in `main.py`.

## 5. Main config (main.py)

`Config` contains all key settings:

- Audio:
  - `sample_rate` (default `16000`)
  - `block_ms` (audio callback block size)
- Chunking:
  - `pause_sec` (silence duration to close chunk)
  - `min_utterance_sec` (minimum accepted utterance)
  - `min_emit_sec` (minimum chunk size before pause-based send)
  - `max_utterance_sec` (forced split for very long speech)
  - `max_split_overlap_sec` (overlap on forced split to reduce word loss)
- Whisper:
  - `model_size` (`medium`, `large-v3`, etc.)
  - `language` (for Russian use `"ru"`)
  - `beam_size`, `best_of`
  - `no_speech_threshold`, `log_prob_threshold`, `compression_ratio_threshold`
- VAD and cleanup:
  - `silence_rms_threshold` (external pause detector)
  - `vad_speech_pad_ms` (Silero VAD padding inside whisper)
  - `blacklist_phrases` (known hallucinations to remove)
- Hotkeys:
  - `hotkey_toggle = "f9"`
  - `hotkey_quit = "f10"`

## 6. GPU notes

The script uses:

```python
device = "cuda"
compute_type = "float16"
```

If GPU is not used, verify your CUDA stack and `ctranslate2`/`faster-whisper` environment.

## 7. Troubleshooting

### Hotkeys do not work
- Run terminal as Administrator.
- Avoid conflicts with keyboard vendor software.
- Try alternative combinations (for example `ctrl+shift+f9`).

### Text quality is poor / wrong phrases
- Keep microphone close and clean input signal.
- Keep `language = "ru"` for Russian-only speech.
- Increase model size (`large-v3`) for better accuracy.
- Tune chunking: larger `pause_sec` and `min_emit_sec` reduce fragmented chunks.

### Words disappear at chunk boundaries
- Increase `max_split_overlap_sec` (for example from `0.6` to `0.8`).
- Increase `vad_speech_pad_ms` (for example to `320`).

### Frequent very short chunks (1-3 sec)
- Increase `min_emit_sec`.
- Increase `pause_sec`.
- Lower `silence_rms_threshold` if silence detector is too aggressive.

## 8. Privacy

- Audio is processed locally on your machine.
- No cloud transcription is used by this script itself.

## 9. Limitations

- This is optimized for Windows desktop usage.
- Active-window typing may interfere with your own typing if both happen at once.
- Accuracy depends heavily on microphone quality and room noise.

## 10. Recommended repository structure

- `main.py` - main script
- `requirements.txt` - dependencies
- `README.md` - this documentation

Optionally add:
- `LICENSE`
- `.gitignore`
- `CHANGELOG.md`
