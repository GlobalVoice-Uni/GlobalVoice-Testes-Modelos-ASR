"""
Realtime benchmark (terminal-only) with utterance-level commits.
Strategy:
"""
import argparse
import json
import math
import queue
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
from scipy.io.wavfile import write as wav_write
from scipy.signal import resample_poly
import sounddevice as sd
import torch
from faster_whisper import WhisperModel


class RealtimeTranscriberBenchmark:
    def __init__(self, model_size="base", device="cpu", language="en", context_window=5):
        self.model_size = model_size
        self.device_request = device
        self.language = "pt" if language == "pt-br" else language
        self.context_window = max(0, context_window)

        self.sample_rate = 16000
        self.step_duration = 0.2
        self.input_sample_rate = None
        self.step_samples = 0
        self.speech_peak_threshold = 0.0018
        self.max_silence_inside_utterance_s = 0.4
        self.max_utterance_s = 3.2
        self.min_utterance_s = 0.7
        self.boundary_overlap_s = 0.45
        self.tail_guard_words = 4


        self.input_device = None  # default system mic
        self.audio_queue = queue.Queue()
        self.audio_remainder = np.zeros(0, dtype=np.float32)

        self.live_words = deque(maxlen=1400)
        self.context_words = deque(maxlen=self.context_window)
        self.full_parts = []
        self.captured_chunks = []
        self.last_committed_norm = ""
        self.pending_tail_words = []

        self.start_time = None
        self.total_audio_duration = 0.0
        self.chunks_processed = 0
        self.latencies_ms = []

        self.whisper_device = self._resolve_device(self.device_request)
        self.model = self._load_model()

        print("Initializing Faster-Whisper...")
        print(f"  Model: {self.model_size}")
        print(f"  Device requested: {self.device_request}")
        print(f"  Device used: {self.whisper_device}")
        print(f"  Language: {self.language}")
        print(f"  Step duration: {self.step_duration:.1f}s")
        print(f"  Max utterance: {self.max_utterance_s:.1f}s")
        print(f"  Boundary overlap: {self.boundary_overlap_s:.2f}s")
        print(f"  Tail guard words: {self.tail_guard_words}")

    def _resolve_device(self, requested):
        if requested == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but CUDA is unavailable.")
            return "cuda"
        return "cpu"

    def _load_model(self):
        if self.whisper_device == "cuda":
            last_error = None
            for compute_type in ("float16", "int8_float16", "float32"):
                try:
                    print(f"Trying GPU compute_type={compute_type}...")
                    return WhisperModel(
                        self.model_size,
                        device="cuda",
                        compute_type=compute_type,
                        num_workers=1,
                    )
                except Exception as exc:
                    last_error = exc
                    print(f"Failed compute_type={compute_type}: {exc}")
            raise RuntimeError("Could not initialize Faster-Whisper on GPU.") from last_error

        return WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
            num_workers=1,
        )

    def _audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata[:, 0].copy())

    def _resample_to_16k(self, audio_data, source_sr):
        if int(source_sr) == self.sample_rate:
            return audio_data.astype(np.float32)

        g = math.gcd(int(source_sr), int(self.sample_rate))
        up = int(self.sample_rate // g)
        down = int(source_sr // g)
        return resample_poly(audio_data, up, down).astype(np.float32)

    def _get_step_audio(self):
        parts = []
        total = 0

        if len(self.audio_remainder) > 0:
            parts.append(self.audio_remainder)
            total += len(self.audio_remainder)
            self.audio_remainder = np.zeros(0, dtype=np.float32)

        deadline = time.time() + 2.0
        while total < self.step_samples and time.time() < deadline:
            timeout = max(0.05, deadline - time.time())
            try:
                chunk = self.audio_queue.get(timeout=timeout)
            except queue.Empty:
                break
            parts.append(chunk)
            total += len(chunk)

        if total == 0:
            return np.zeros(self.step_samples, dtype=np.float32), 0.0, 0.0

        merged = np.concatenate(parts).astype(np.float32)
        valid_len = min(len(merged), self.step_samples)

        if len(merged) >= self.step_samples:
            audio = merged[: self.step_samples]
            self.audio_remainder = merged[self.step_samples :]
        else:
            audio = np.pad(merged, (0, self.step_samples - len(merged)), mode="constant")

        valid = audio[:valid_len]
        rms = float(np.sqrt(np.mean(np.square(valid)))) if len(valid) else 0.0
        peak = float(np.max(np.abs(valid))) if len(valid) else 0.0
        audio_16k = self._resample_to_16k(audio, self.input_sample_rate)
        return audio_16k, rms, peak

    def _transcribe_audio(self, audio_data):
        context_prompt = " ".join(self.context_words) if self.context_words else None

        t0 = time.time()
        segments, _ = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=5,
            initial_prompt=context_prompt,
            condition_on_previous_text=False,
            no_speech_threshold=0.35,
            vad_filter=True,
            temperature=0.0,
        )
        transcribe_ms = (time.time() - t0) * 1000.0
        self.latencies_ms.append(transcribe_ms)

        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        return text, transcribe_ms

    def _extract_new_suffix(self, candidate_text):
        words = [w for w in candidate_text.split() if w.strip()]
        if not words:
            return []

        def _norm(token):
            return re.sub(r"[^\w]", "", token.lower())

        history = list(self.live_words)
        max_overlap = min(len(history), len(words), 30)
        overlap = 0
        for k in range(max_overlap, 0, -1):
            if [_norm(w) for w in history[-k:]] == [_norm(w) for w in words[:k]]:
                overlap = k
                break

        return words[overlap:]

    def _looks_like_loop(self, words):
        if len(words) < 28:
            return False

        lower_words = [w.lower() for w in words]
        uniq_ratio = len(set(lower_words)) / len(lower_words)
        if uniq_ratio < 0.20:
            return True

        bigrams = [f"{lower_words[i]} {lower_words[i+1]}" for i in range(len(lower_words) - 1)]
        counts = {}
        for bg in bigrams:
            counts[bg] = counts.get(bg, 0) + 1
        return max(counts.values()) >= 10 if counts else False

    def _append_words(self, new_words):
        if not new_words or self._looks_like_loop(new_words):
            return

        text = " ".join(new_words)
        norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()
        if norm and (norm == self.last_committed_norm or self.last_committed_norm.endswith(norm)):
            return

        self.full_parts.append(text)
        self.last_committed_norm = norm if norm else self.last_committed_norm

        for w in new_words:
            self.live_words.append(w)
            clean = w.rstrip(".,!?;:")
            if clean:
                self.context_words.append(clean)

        # Saida append-only: somente texto novo confirmado, sem reimprimir tudo.
        print(text + " ", end="", flush=True)

    def _merge_with_pending_tail(self, words):
        if not self.pending_tail_words:
            return words

        def _norm(token):
            return re.sub(r"[^\w]", "", token.lower())

        pending = self.pending_tail_words
        max_overlap = min(len(pending), len(words), self.tail_guard_words)
        overlap = 0
        for k in range(max_overlap, 0, -1):
            if [_norm(w) for w in pending[-k:]] == [_norm(w) for w in words[:k]]:
                overlap = k
                break

        merged = pending + words[overlap:]
        self.pending_tail_words = []
        return merged

    def _commit_transcribed_text(self, text, forced_split):
        words = [w for w in text.split() if w.strip()]
        if not words:
            return

        words = self._merge_with_pending_tail(words)

        if forced_split:
            if len(words) <= self.tail_guard_words:
                self.pending_tail_words = words
                return
            commit_words = words[:-self.tail_guard_words]
            self.pending_tail_words = words[-self.tail_guard_words:]
        else:
            commit_words = words

        if commit_words:
            new_words = self._extract_new_suffix(" ".join(commit_words))
            self._append_words(new_words)

    def start_recording(self, duration):
        self.start_time = time.time()

        try:
            default_input = sd.query_devices(kind="input")
            print(f"Using default input device: {default_input.get('name', 'unknown')}")
            self.input_sample_rate = int(default_input.get("default_samplerate", self.sample_rate))
        except Exception:
            print("Using default input device.")
            self.input_sample_rate = self.sample_rate

        self.step_samples = int(self.input_sample_rate * self.step_duration)
        print(f"Capture sample rate: {self.input_sample_rate} Hz -> ASR sample rate: {self.sample_rate} Hz")
        print("Starting microphone capture...")

        remaining = float(duration)
        speech_active = False
        speech_buffers = []
        silence_steps = 0
        carryover_audio = np.zeros(0, dtype=np.float32)

        max_silence_steps = max(1, int(self.max_silence_inside_utterance_s / self.step_duration))
        max_utt_samples = int(self.max_utterance_s * self.sample_rate)
        min_utt_samples = int(self.min_utterance_s * self.sample_rate)
        overlap_samples = int(self.boundary_overlap_s * self.sample_rate)

        with sd.InputStream(
            samplerate=self.input_sample_rate,
            channels=1,
            dtype="float32",
            latency="high",
            blocksize=0,
            callback=self._audio_callback,
            device=self.input_device,
        ):
            while remaining > 0:
                current_step = min(self.step_duration, remaining)
                audio, _, peak = self._get_step_audio()
                self.captured_chunks.append(audio.copy())

                self.chunks_processed += 1
                self.total_audio_duration += current_step

                has_speech = peak >= self.speech_peak_threshold

                if has_speech:
                    if not speech_active:
                        speech_buffers = [carryover_audio] if len(carryover_audio) > 0 else []
                        carryover_audio = np.zeros(0, dtype=np.float32)
                    speech_active = True
                    silence_steps = 0
                    speech_buffers.append(audio)
                elif speech_active:
                    silence_steps += 1
                    speech_buffers.append(audio)

                utter_len = sum(len(x) for x in speech_buffers) if speech_buffers else 0
                should_finalize = speech_active and (
                    silence_steps >= max_silence_steps or utter_len >= max_utt_samples
                )

                if should_finalize:
                    forced_split = utter_len >= max_utt_samples and silence_steps < max_silence_steps
                    utter = np.concatenate(speech_buffers).astype(np.float32)
                    speech_active = False
                    speech_buffers = []
                    silence_steps = 0

                    if forced_split and overlap_samples > 0 and len(utter) > overlap_samples:
                        carryover_audio = utter[-overlap_samples:].copy()
                    else:
                        carryover_audio = np.zeros(0, dtype=np.float32)

                    if len(utter) >= min_utt_samples:
                        text, transcribe_ms = self._transcribe_audio(utter)
                        if text:
                            self._commit_transcribed_text(text, forced_split=forced_split)

                remaining -= current_step

        if speech_buffers:
            utter = np.concatenate(speech_buffers).astype(np.float32)
            if len(utter) >= min_utt_samples:
                text, transcribe_ms = self._transcribe_audio(utter)
                if text:
                    self._commit_transcribed_text(text, forced_split=False)

        if self.pending_tail_words:
            tail_new_words = self._extract_new_suffix(" ".join(self.pending_tail_words))
            self._append_words(tail_new_words)
            self.pending_tail_words = []

        print("\n")

    def _save_results(self, elapsed_time, final_text, ram_mb):
        lat = sorted(self.latencies_ms)
        if lat:
            latency_obj = {
                "min": min(lat),
                "max": max(lat),
                "mean": float(np.mean(lat)),
                "p50": lat[int(0.50 * (len(lat) - 1))],
                "p95": lat[int(0.95 * (len(lat) - 1))],
                "p99": lat[int(0.99 * (len(lat) - 1))],
            }
        else:
            latency_obj = {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}

        timestamp_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = f"realtime-{self.model_size}-{self.device_request}-{self.language}-{self.context_window}ctx-{timestamp_tag}"

        out_dir = Path("realtime_tests") / "resultados"
        out_dir.mkdir(parents=True, exist_ok=True)

        debug_audio_path = None
        if self.captured_chunks:
            debug_audio = np.concatenate(self.captured_chunks).astype(np.float32)
            debug_audio = np.clip(debug_audio, -1.0, 1.0)
            debug_dir = out_dir / "realtime_debug_audio"
            debug_dir.mkdir(parents=True, exist_ok=True)
            wav_path = debug_dir / f"{stem}.wav"
            wav_write(str(wav_path), self.sample_rate, (debug_audio * 32767.0).astype(np.int16))
            debug_audio_path = str(wav_path)

        payload = {
            "model": "faster-whisper",
            "model_size": self.model_size,
            "device": self.device_request,
            "language": self.language,
            "context_window": self.context_window,
            "step_duration_s": self.step_duration,
            "boundary_overlap_s": self.boundary_overlap_s,
            "tail_guard_words": self.tail_guard_words,
            "timestamp": datetime.now().isoformat(),
            "total_time_s": elapsed_time,
            "audio_duration_s": self.total_audio_duration,
            "real_time_factor": elapsed_time / self.total_audio_duration if self.total_audio_duration > 0 else 0,
            "chunks_processed": self.chunks_processed,
            "latency_ms": latency_obj,
            "transcription": final_text,
            "ram_peak_mb": ram_mb,
            "debug_audio_path": debug_audio_path,
        }

        json_path = out_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"Saved results: {json_path}")

    def print_results(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0.0
        final_text = " ".join(self.full_parts).strip()

        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Elapsed time: {elapsed_time:.2f}s")
        print(f"Audio processed: {self.total_audio_duration:.2f}s")
        print(f"RTF: {(elapsed_time / self.total_audio_duration) if self.total_audio_duration > 0 else 0.0:.2f}x")
        print(f"Chunks processed: {self.chunks_processed}")

        if self.latencies_ms:
            lat = sorted(self.latencies_ms)
            print(
                f"Latency mean: {float(np.mean(lat)):.1f}ms | "
                f"p50: {lat[int(0.50 * (len(lat) - 1))]:.1f}ms | "
                f"p95: {lat[int(0.95 * (len(lat) - 1))]:.1f}ms"
            )

        ram_mb = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"RAM: {ram_mb:.1f} MB")
        print("FINAL:")
        print(final_text if final_text else "(vazio)")

        self._save_results(elapsed_time, final_text, ram_mb)


def main():
    parser = argparse.ArgumentParser(description="Realtime benchmark with Faster-Whisper (clean)")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], default="base")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--language", choices=["en", "pt-br"], default="en")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--context", type=int, default=5)

    args = parser.parse_args()

    bench = RealtimeTranscriberBenchmark(
        model_size=args.model,
        device=args.device,
        language=args.language,
        context_window=args.context,
    )
    bench.start_recording(duration=args.duration)
    bench.print_results()


if __name__ == "__main__":
    main()
