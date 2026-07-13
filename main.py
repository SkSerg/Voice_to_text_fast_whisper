import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import keyboard as kb
import numpy as np
import sounddevice as sd
from pynput.keyboard import Controller


@dataclass
class Config:
    sample_rate: int = 16000
    block_ms: int = 100
    pause_sec: float = 1.4
    model_size: str = "large-v3"  # "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
    device: str = "cuda"  # "cpu" or "cuda"
    compute_type: str = "float16"  # "int8", "int8_float16", "float16", "float32"
    cuda_device: int = 0  # index inside CUDA_VISIBLE_DEVICES
    cuda_gpu_name: str = "2080 Ti"
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
    hotkey_record: str = "f9"
    hotkey_quit: str = "f10"
    f9_release_debounce_sec: float = 0.25
    replace_ellipsis: bool = True
    strip_trailing_punctuation: bool = False
    skip_if_buffer_rms_below: float = 0.0035
    type_delay_sec: float = 0.01  # small delay between keypresses
    add_newline: bool = False


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.device == "cuda":
            self._select_cuda_gpu(cfg.cuda_gpu_name)
            cfg.cuda_device = 0
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
        # Import only after CUDA_VISIBLE_DEVICES has been set. CTranslate2 reads
        # the variable while its CUDA runtime is initialized.
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
            device_index=cfg.cuda_device,
        )

    @staticmethod
    def _select_cuda_gpu(required_name: str) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "Не удалось определить GPU через nvidia-smi; запуск CUDA остановлен, "
                "чтобы случайно не использовать другую видеокарту."
            ) from exc

        available: list[tuple[str, str]] = []
        for line in result.stdout.splitlines():
            index, separator, name = line.partition(",")
            if separator:
                available.append((index.strip(), name.strip()))

        matches = [gpu for gpu in available if required_name.casefold() in gpu[1].casefold()]
        if not matches:
            detected = ", ".join(f"{index}: {name}" for index, name in available) or "нет"
            raise RuntimeError(
                f"GPU с именем '{required_name}' не найдена. Обнаружены: {detected}. "
                "Запуск остановлен, чтобы не использовать RTX 4090 или другую карту."
            )

        physical_index, gpu_name = matches[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_index
        print(
            f"Выбрана GPU {physical_index}: {gpu_name} "
            f"(CUDA device_index внутри процесса: 0).",
            flush=True,
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


def ensure_final_punctuation(text: str) -> str:
    """End the complete transcription with either '?' or '.'."""
    text = text.rstrip()
    is_question = text.endswith("?")
    text = text.rstrip(".,!?;: ")
    if not text:
        return ""
    return f"{text}{'?' if is_question else '.'}"


def main() -> int:
    cfg = Config()
    keyboard = Controller()
    recording = threading.Event()
    stop_requested = threading.Event()

    block_samples = int(cfg.sample_rate * cfg.block_ms / 1000)
    work_q: queue.Queue[tuple[bytes, bool] | None] = queue.Queue(maxsize=10)
    capture_buffer = bytearray()
    capture_lock = threading.Lock()
    release_generation = 0

    def audio_callback(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        with capture_lock:
            if recording.is_set():
                capture_buffer.extend(bytes(indata))

    stream = sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=1,
        dtype="int16",
        blocksize=block_samples,
        callback=audio_callback,
    )

    print(
        "Готово. Удерживайте F9 для записи; после отпускания начнётся "
        "распознавание. Для выхода нажмите F10.",
        flush=True,
    )

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
                if try_switch_language(text):
                    continue
                text = ensure_final_punctuation(text)
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

    def on_record_press(_event) -> None:
        nonlocal release_generation
        if stop_requested.is_set():
            return
        with capture_lock:
            # Some keyboards/drivers emit a short release/press pair while F9
            # is physically held. Invalidate a pending delayed release without
            # clearing the audio already collected.
            release_generation += 1
            if recording.is_set():
                return
            capture_buffer.clear()
            recording.set()
        print("\nСостояние: ЗАПИСЬ (F9 удерживается).", flush=True)

    def finish_recording(expected_generation: int | None = None) -> None:
        with capture_lock:
            if expected_generation is not None and expected_generation != release_generation:
                return
            if not recording.is_set():
                return
            recording.clear()
            payload = bytes(capture_buffer)
            capture_buffer.clear()
        if payload:
            print("\nСостояние: ОБРАБОТКА. Запись завершена.", flush=True)
            enqueue_audio(payload)

    def on_record_release(_event) -> None:
        nonlocal release_generation
        with capture_lock:
            if not recording.is_set():
                return
            release_generation += 1
            expected_generation = release_generation

        # Confirm that F9 really stayed released. A repeated press invalidates
        # this generation and recording continues in the same buffer.
        release_timer = threading.Timer(
            cfg.f9_release_debounce_sec,
            finish_recording,
            args=(expected_generation,),
        )
        release_timer.daemon = True
        release_timer.start()

    def on_quit():
        nonlocal release_generation
        with capture_lock:
            release_generation += 1
        finish_recording()
        stop_requested.set()
        try:
            work_q.put_nowait(None)
        except queue.Full:
            pass

    kb.on_press_key(cfg.hotkey_record, on_record_press, suppress=True)
    kb.on_release_key(cfg.hotkey_record, on_record_release, suppress=True)
    kb.add_hotkey(cfg.hotkey_quit, on_quit, suppress=True)

    with stream:
        try:
            while not stop_requested.is_set():
                stop_requested.wait(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
            finish_recording()
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
