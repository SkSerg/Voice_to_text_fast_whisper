# Voice to Text (Windows, local Faster-Whisper)

Local speech-to-text script for Windows that:
- captures audio from your microphone,
- transcribes speech with `faster-whisper` locally,
- types recognized text into the currently active app (Telegram, Word, VS Code, etc.).

The script supports:
- GPU inference (`CUDA`) when available,
- push-to-talk recording with a global hotkey,
- transcription of the complete recording after the hotkey is released,
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

At startup, the script waits for push-to-talk:
- hold `F9` -> record audio
- release `F9` -> stop recording, transcribe the complete recording, and type it
- `F10` -> quit

Default transcription language on startup is Russian (`ru`).
You can switch language by saying a single command word (the command itself is not typed to output):
- switch to Russian (`ru`): `русский` / `russian` / `russisch`
- switch to English (`en`): `английский` / `english` / `englisch`
- switch to German (`de`): `немецкий` / `german` / `deutsch`

## 4. How output works

- Default mode: `output_mode = "active_window"`
- The script types text into the app that currently has keyboard focus.
- By default, `add_newline = False`, so it does not press Enter automatically.

If you want console output for debugging:
- set `output_mode = "console"` in `main.py`.

## 5. Push-to-talk recording

- The microphone audio is accumulated only while `F9` is physically held down.
- No partial transcription is started while recording.
- Releasing `F9` sends the whole accumulated recording to Whisper.
- The complete recognized text always ends with punctuation: Whisper's `?` is
  preserved for a question; otherwise the text ends with `.`.

## 6. Main config (main.py)

`Config` contains all key settings:

- Audio:
  - `sample_rate` (default `16000`)
  - `block_ms` (audio callback block size)
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
  - `hotkey_record = "f9"`
  - `hotkey_quit = "f10"`
  - `f9_release_debounce_sec = 0.25` (ignores short false release/press events
    produced by some keyboards while F9 is still physically held)

## 7. Run on a PC without GPU (CPU mode)

If you do not have an NVIDIA GPU, update `Config` in `main.py`:

```python
device = "cpu"
compute_type = "int8"
model_size = "small"  # or "base" for weaker CPUs
```

Recommendations for CPU mode:
- Start with `model_size = "small"` for a balance of speed/quality.
- If transcription is still slow, switch to `model_size = "base"`.
- `medium`/`large-v3` on CPU are usually much slower and may lag in real-time use.

## 8. Encoding (important for Russian text)

Use UTF-8 in terminal to avoid garbled Cyrillic output:

```powershell
chcp 65001
$env:PYTHONUTF8 = "1"
python main.py
```

Also ensure files are saved in UTF-8:
- `main.py`
- `README.md`

## 9. GPU notes

Default GPU config:

```python
device = "cuda"
compute_type = "float16"
cuda_gpu_name = "2080 Ti"
```

At startup the script finds the GPU by name through `nvidia-smi`, sets it as the
only visible CUDA device, and then loads Faster-Whisper. If RTX 2080 Ti cannot be
found, startup stops instead of silently using RTX 4090 or another GPU.

## 10. Troubleshooting

### Hotkeys do not work
- Run terminal as Administrator.
- Avoid conflicts with keyboard vendor software.
- Try alternative combinations (for example `ctrl+shift+f9`).

### Text quality is poor / wrong phrases
- Keep microphone close and clean input signal.
- For best accuracy, keep the active transcription language aligned with the spoken language (use voice commands above to switch).
- Increase model size (`large-v3`) for better accuracy.

## 11. Privacy

- Audio is processed locally on your machine.
- No cloud transcription is used by this script itself.

## 12. Limitations

- This is optimized for Windows desktop usage.
- Active-window typing may interfere with your own typing if both happen at once.
- Accuracy depends heavily on microphone quality and room noise.

## 13. Recommended repository structure

- `main.py` - main script
- `requirements.txt` - dependencies
- `README.md` - this documentation

Optionally add:
- `LICENSE`
- `.gitignore`
- `CHANGELOG.md`
