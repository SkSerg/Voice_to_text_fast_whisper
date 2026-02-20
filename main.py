import queue
import re
import sys
import threading
import time
from dataclasses import dataclass

import keyboard as kb
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput.keyboard import Controller


@dataclass
class Config:
    sample_rate: int = 16000
    block_ms: int = 100
    pause_sec: float = 1.4
    inactivity_pause_sec: float = 120.0
    min_utterance_sec: float = 0.9
    min_emit_sec: float = 3.5
    max_utterance_sec: float = 20.0
    max_split_overlap_sec: float = 0.6
    silence_rms_threshold: float = 0.005
    model_size: str = "large-v3"  # "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
    device: str = "cuda"  # "cpu" or "cuda"
    compute_type: str = "float16"  # "int8", "int8_float16", "float16", "float32"
    cuda_device: int = 0
    language: str = "ru"
    beam_size: int = 5
    best_of: int = 5
    initial_prompt: str = ""
    no_speech_threshold: float = 0.85
    log_prob_threshold: float = -1.5
    compression_ratio_threshold: float = 2.8
    vad_speech_pad_ms: int = 260
    use_segment_confidence_filter: bool = False
    segment_min_avg_logprob: float = -1.2
    segment_max_no_speech_prob: float = 0.7
    blacklist_phrases: tuple[str, ...] = (
        "Субтитры сделал DimaTorzok",
        "Продолжение следует...",
        "Продолжение следует....",
    )
    output_mode: str = "active_window"  # "console" or "active_window"
    hotkey_toggle: str = "f9"
    hotkey_quit: str = "f10"
    replace_ellipsis: bool = True
    strip_trailing_punctuation: bool = True
    skip_if_buffer_rms_below: float = 0.0035
    min_voiced_chunk_ratio: float = 0.12
    type_delay_sec: float = 0.01  # small delay between keypresses
    add_newline: bool = False


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.device == "cuda":
            try:
                import ctranslate2 as ct
                if hasattr(ct, "get_cuda_version"):
                    print(f"ctranslate2 CUDA: {ct.get_cuda_version()}", flush=True)
                if hasattr(ct, "get_supported_compute_types"):
                    print(f"compute types: {ct.get_supported_compute_types('cuda')}", flush=True)
                if hasattr(ct, "get_cuda_device_count"):
                    print(f"cuda devices: {ct.get_cuda_device_count()}", flush=True)
            except Exception as exc:
                print(f"CUDA check failed: {exc}", file=sys.stderr)
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
            device_index=cfg.cuda_device,
        )

    def set_language(self, language: str) -> None:
        self.cfg.language = language

    @staticmethod
    def _normalize_text_for_filtering(text: str) -> str:
        normalized = text.lower().replace("ё", "е")
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        return " ".join(normalized.split())

    def transcribe(self, pcm_i16: np.ndarray) -> str:
        if pcm_i16.size == 0:
            return ""
        audio = pcm_i16.astype(np.float32) / 32768.0
        t0 = time.time()
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            task="transcribe",
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": int(self.cfg.pause_sec * 1000),
                "speech_pad_ms": self.cfg.vad_speech_pad_ms,
            },
            beam_size=self.cfg.beam_size,
            best_of=self.cfg.best_of,
            temperature=0,
            condition_on_previous_text=False,
            initial_prompt=self.cfg.initial_prompt or None,
            no_speech_threshold=self.cfg.no_speech_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
        )
        dt = time.time() - t0
        audio_sec = pcm_i16.size / float(self.cfg.sample_rate)
        print(f"transcribe {audio_sec:.2f}s -> {dt:.2f}s", flush=True)
        selected_texts: list[str] = []
        for seg in segments:
            if self.cfg.use_segment_confidence_filter:
                avg_logprob = float(getattr(seg, "avg_logprob", -999.0))
                no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0))
                if avg_logprob < self.cfg.segment_min_avg_logprob:
                    continue
                if no_speech_prob > self.cfg.segment_max_no_speech_prob:
                    continue
            selected_texts.append(seg.text)
        text = "".join(selected_texts).strip()
        if self.cfg.replace_ellipsis:
            text = text.replace("…", " ").replace("...", " ")
        normalized_text = self._normalize_text_for_filtering(text)
        normalized_blacklist = {
            self._normalize_text_for_filtering(phrase) for phrase in self.cfg.blacklist_phrases
        }
        for phrase in normalized_blacklist:
            if phrase and phrase in normalized_text:
                return ""
        if "продолжение следует" in normalized_text:
            return ""
        if "субтитры" in normalized_text and "dimatorzok" in normalized_text:
            return ""
        text = " ".join(text.split())
        if self.cfg.strip_trailing_punctuation:
            text = text.rstrip(".,!?;:")
        return text


def main() -> int:
    cfg = Config()
    keyboard = Controller()
    running = threading.Event()
    running.clear()
    stop_requested = threading.Event()
    flush_requested = threading.Event()
    last_transcription_time = time.monotonic()

    block_samples = int(cfg.sample_rate * cfg.block_ms / 1000)
    block_bytes = block_samples * 2  # int16

    q: queue.Queue[bytes] = queue.Queue(maxsize=200)
    work_q: queue.Queue[tuple[bytes, bool] | None] = queue.Queue(maxsize=10)

    def audio_callback(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    stream = sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=1,
        dtype="int16",
        blocksize=block_samples,
        callback=audio_callback,
    )

    print(
        f"Состояние: ПАУЗА. Для запуска нажмите F9. Для выхода нажмите F10. "
        f"Автопауза: {int(cfg.inactivity_pause_sec)} сек без транскрибации.",
        flush=True,
    )

    min_utt_bytes = int(cfg.min_utterance_sec * cfg.sample_rate) * 2
    min_emit_bytes = int(cfg.min_emit_sec * cfg.sample_rate) * 2
    max_utt_bytes = int(cfg.max_utterance_sec * cfg.sample_rate) * 2
    max_overlap_bytes = int(cfg.max_split_overlap_sec * cfg.sample_rate) * 2
    silence_chunks_needed = max(1, int(cfg.pause_sec * 1000 / cfg.block_ms))

    transcriber = Transcriber(cfg)
    language_commands = {
        "ru": {"русский", "russian", "russisch"},
        "en": {"english", "английский", "englisch"},
        "de": {"немецкий", "german", "deutsch"},
    }
    language_names = {"ru": "русский", "en": "английский", "de": "немецкий"}

    def normalize_command(text: str) -> str:
        return text.strip().lower().strip(".,!?;:()[]{}\"'")

    def try_switch_language(text: str) -> bool:
        normalized = normalize_command(text)
        for language, commands in language_commands.items():
            if normalized in commands:
                transcriber.set_language(language)
                print(f"\nЯзык транскрибации: {language_names[language]} ({language}).", flush=True)
                return True
        return False

    def enqueue_audio(payload: bytes, add_sentence_dot: bool = False) -> None:
        if not payload:
            return
        while not stop_requested.is_set():
            try:
                work_q.put((payload, add_sentence_dot), timeout=0.2)
                return
            except queue.Full:
                continue

    def worker():
        nonlocal last_transcription_time
        while True:
            item = work_q.get()
            if item is None:
                return
            payload, add_sentence_dot = item
            pcm = np.frombuffer(payload, dtype=np.int16)
            pcm_f32 = pcm.astype(np.float32) / 32768.0
            buffer_rms = float(np.sqrt(np.mean(pcm_f32 * pcm_f32))) if pcm_f32.size else 0.0
            if buffer_rms < cfg.skip_if_buffer_rms_below:
                continue
            text = transcriber.transcribe(pcm)
            if text:
                last_transcription_time = time.monotonic()
                if add_sentence_dot and text[-1] not in ".!?":
                    text = f"{text}."
                if try_switch_language(text):
                    continue
                if cfg.output_mode == "active_window":
                    keyboard.type(text + " ")
                    if cfg.add_newline:
                        keyboard.type("\n")
                else:
                    print(text, end=" ", flush=True)
                    if cfg.add_newline:
                        print("", flush=True)
                time.sleep(cfg.type_delay_sec)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    def on_toggle():
        nonlocal last_transcription_time
        if running.is_set():
            running.clear()
            flush_requested.set()
            print("\nСостояние: ПАУЗА. Для запуска нажмите F9.", flush=True)
        else:
            running.set()
            last_transcription_time = time.monotonic()
            print(
                f"\nСостояние: ЗАПИСЬ. Автопауза через {int(cfg.inactivity_pause_sec)} сек без транскрибации.",
                flush=True,
            )

    def on_quit():
        stop_requested.set()
        flush_requested.set()
        running.set()
        try:
            work_q.put_nowait(None)
        except queue.Full:
            pass

    kb.add_hotkey(cfg.hotkey_toggle, on_toggle)
    kb.add_hotkey(cfg.hotkey_quit, on_quit)

    with stream:
        try:
            capture_buffer = bytearray()
            silence_chunks = 0
            while not stop_requested.is_set():
                if running.is_set() and (time.monotonic() - last_transcription_time >= cfg.inactivity_pause_sec):
                    running.clear()
                    capture_buffer.clear()
                    silence_chunks = 0
                    print(
                        f"\nСостояние: ПАУЗА. Нет транскрибации {int(cfg.inactivity_pause_sec)} сек. Для запуска нажмите F9.",
                        flush=True,
                    )

                if flush_requested.is_set():
                    if len(capture_buffer) >= min_utt_bytes:
                        enqueue_audio(bytes(capture_buffer))
                    capture_buffer.clear()
                    silence_chunks = 0
                    flush_requested.clear()

                if not running.is_set():
                    # Drop mic data while paused to avoid stale transcription.
                    while True:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                    time.sleep(0.05)
                    continue
                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if len(chunk) < block_bytes:
                    continue
                capture_buffer.extend(chunk)

                chunk_pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                chunk_rms = float(np.sqrt(np.mean(chunk_pcm * chunk_pcm)))
                if chunk_rms < cfg.silence_rms_threshold:
                    silence_chunks += 1
                else:
                    silence_chunks = 0

                if (
                    len(capture_buffer) >= min_utt_bytes
                    and len(capture_buffer) >= min_emit_bytes
                    and silence_chunks >= silence_chunks_needed
                ):
                    chunks_total = len(capture_buffer) // block_bytes
                    voiced_chunks = max(0, chunks_total - silence_chunks)
                    voiced_ratio = (voiced_chunks / chunks_total) if chunks_total > 0 else 0.0
                    if voiced_ratio < cfg.min_voiced_chunk_ratio:
                        capture_buffer.clear()
                        silence_chunks = 0
                        continue
                    enqueue_audio(bytes(capture_buffer), add_sentence_dot=True)
                    capture_buffer.clear()
                    silence_chunks = 0

                if len(capture_buffer) >= max_utt_bytes:
                    enqueue_audio(bytes(capture_buffer))
                    if max_overlap_bytes > 0 and len(capture_buffer) > max_overlap_bytes:
                        capture_buffer = bytearray(capture_buffer[-max_overlap_bytes:])
                    else:
                        capture_buffer.clear()
                    silence_chunks = 0
        except KeyboardInterrupt:
            print("\nStopped.")
            flush_requested.set()
            try:
                work_q.put_nowait(None)
            except queue.Full:
                pass
            kb.clear_all_hotkeys()
            return 0
        finally:
            kb.clear_all_hotkeys()


if __name__ == "__main__":
    raise SystemExit(main())
