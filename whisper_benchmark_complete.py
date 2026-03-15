import argparse
import time
import json
import logging
import gc
import shutil
import subprocess
import threading
import re
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import sounddevice as sd
import psutil
from faster_whisper import WhisperModel
import whisper as openai_whisper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


class WhisperBenchmark:
    """Simple Whisper Benchmark - Compare different models with microphone or audio file input"""
    
    def __init__(self, model_name: str, use_gpu: bool = False, model_size: str = "base"):
        self.model_name = model_name
        self.model_size = model_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        self.process = psutil.Process()
        self._ram_peak_bytes = 0
        self._vram_peak_mb = 0.0
        self._gpu_memory_query_mode = "unavailable"
        self._gpu_total_baseline_mb = 0.0
        self._nvidia_smi_path = shutil.which("nvidia-smi") if self.use_gpu else None
        self.show_transcription_early = False
        self.measure_chunks = False
        self.language: Optional[str] = None
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Selected model: {self.model_name} ({self.model_size})")
        self._load_model()
    
    def _load_model(self):
        """Load the selected model"""
        logger.info(f"Loading {self.model_name} model...")
        
        if self.model_name == 'faster-whisper':
            if self.use_gpu:
                compute_candidates = ["float16", "int8_float16", "float32"]
                last_error = None
                for compute_type in compute_candidates:
                    try:
                        logger.info(f"Trying faster-whisper compute_type={compute_type} on GPU...")
                        self.model = WhisperModel(
                            self.model_size,
                            device=self.device,
                            compute_type=compute_type,
                        )
                        logger.info(f"Loaded faster-whisper with compute_type={compute_type}")
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Failed compute_type={compute_type}: {e}")

                if self.model is None:
                    raise last_error
            else:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="int8"
                )
        
        elif self.model_name == 'openai-whisper':
            self.model = openai_whisper.load_model(self.model_size, device=self.device)
        
        elif self.model_name == 'whisperx':
            try:
                import whisperx
                self.model = whisperx.load_model(self.model_size, device=self.device)
            except Exception as e:
                logger.error(f"Error loading whisperx: {e}")
                raise
        
        logger.info("Model loaded successfully!")
    
    def record_audio(self, duration: float, sample_rate: int = 16000) -> np.ndarray:
        """Record audio from microphone"""
        logger.info(f"Recording from microphone for {duration}s...")
        print(f"\n🎤 Speak now! Recording in progress... ({duration}s)\n")
        
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            latency='high'
        )
        sd.wait()
        
        logger.info("Recording complete!")
        return audio.flatten()

    def load_audio_file(self, file_path: str) -> np.ndarray:
        """Load audio file as mono float32 at 16kHz using Whisper's loader."""
        resolved_path = Path(file_path).expanduser().resolve()
        logger.info(f"Loading audio file: {resolved_path}")
        audio = openai_whisper.load_audio(str(resolved_path))
        return audio.astype(np.float32)

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _compute_wer(self, reference_text: str, hypothesis_text: str) -> Optional[float]:
        ref_tokens = self._normalize_text(reference_text).split()
        hyp_tokens = self._normalize_text(hypothesis_text).split()

        if not ref_tokens:
            return None

        rows = len(ref_tokens) + 1
        cols = len(hyp_tokens) + 1
        dp = [[0] * cols for _ in range(rows)]

        for i in range(rows):
            dp[i][0] = i
        for j in range(cols):
            dp[0][j] = j

        for i in range(1, rows):
            for j in range(1, cols):
                cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )

        return dp[-1][-1] / len(ref_tokens)

    def _clean_transcription_text(self, text: str) -> str:
        """Collapse obvious consecutive sentence repetitions from model output."""
        normalized = " ".join(text.strip().split())
        if not normalized:
            return ""

        parts = re.findall(r'[^.!?]+[.!?]?', normalized)
        cleaned_parts = []

        for part in parts:
            sentence = part.strip()
            if not sentence:
                continue

            sentence_key = re.sub(r"\s+", " ", sentence.lower())
            if cleaned_parts:
                last_key = re.sub(r"\s+", " ", cleaned_parts[-1].lower())
                # Only collapse long, identical consecutive sentences.
                if sentence_key == last_key and len(sentence.split()) >= 5:
                    continue

            cleaned_parts.append(sentence)

        if not cleaned_parts:
            return normalized
        return " ".join(cleaned_parts)

    def _monitor_ram_peak(self, stop_event: threading.Event):
        local_peak = 0
        while not stop_event.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > local_peak:
                    local_peak = rss
            except Exception:
                pass
            time.sleep(0.01)

        self._ram_peak_bytes = max(self._ram_peak_bytes, local_peak)

    def _query_nvidia_smi(self, query_scope: str, query_fields: str) -> str:
        if not self._nvidia_smi_path:
            return ""

        try:
            completed = subprocess.run(
                [
                    self._nvidia_smi_path,
                    f"--query-{query_scope}={query_fields}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return ""

        if completed.returncode != 0:
            return ""

        return completed.stdout.strip()

    def _query_process_vram_mb(self) -> Optional[float]:
        output = self._query_nvidia_smi("compute-apps", "pid,used_gpu_memory")
        if not output:
            return None

        total_used_mb = 0.0
        found_pid = False
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue

            try:
                pid = int(parts[0])
                used_mb = float(parts[1])
            except ValueError:
                continue

            if pid == self.process.pid:
                total_used_mb += used_mb
                found_pid = True

        if not found_pid:
            return None

        return total_used_mb

    def _query_total_vram_mb(self) -> Optional[float]:
        output = self._query_nvidia_smi("gpu", "memory.used")
        if not output:
            return None

        usage_values = []
        for line in output.splitlines():
            value = line.strip()
            if not value:
                continue

            try:
                usage_values.append(float(value))
            except ValueError:
                continue

        if not usage_values:
            return None

        return max(usage_values)

    def _prepare_vram_monitoring(self):
        self._vram_peak_mb = 0.0
        self._gpu_total_baseline_mb = 0.0

        if not self.use_gpu:
            self._gpu_memory_query_mode = "unavailable"
            return

        process_vram_mb = self._query_process_vram_mb()
        if process_vram_mb is not None:
            self._gpu_memory_query_mode = "process"
            self._vram_peak_mb = max(self._vram_peak_mb, process_vram_mb)
            return

        total_vram_mb = self._query_total_vram_mb()
        if total_vram_mb is not None:
            self._gpu_memory_query_mode = "total"
            self._gpu_total_baseline_mb = total_vram_mb
            return

        self._gpu_memory_query_mode = "unavailable"
        logger.warning("Unable to measure VRAM via nvidia-smi; vram_peak_mb will remain 0.")

    def _get_current_vram_mb(self) -> Optional[float]:
        if self._gpu_memory_query_mode == "process":
            return self._query_process_vram_mb()

        if self._gpu_memory_query_mode == "total":
            total_vram_mb = self._query_total_vram_mb()
            if total_vram_mb is None:
                return None
            return max(0.0, total_vram_mb - self._gpu_total_baseline_mb)

        return None

    def _monitor_vram_peak(self, stop_event: threading.Event):
        local_peak_mb = self._vram_peak_mb
        while not stop_event.is_set():
            current_vram_mb = self._get_current_vram_mb()
            if current_vram_mb is not None and current_vram_mb > local_peak_mb:
                local_peak_mb = current_vram_mb
            time.sleep(0.05)

        self._vram_peak_mb = max(self._vram_peak_mb, local_peak_mb)

    def _transcribe_once(self, audio: np.ndarray, collect_text: bool = True):
        transcription = ""
        language = "unknown"

        if self.model_name == 'faster-whisper':
            segments, info = self.model.transcribe(audio, language=self.language)
            if collect_text:
                transcription = " ".join([segment.text for segment in segments])
            else:
                for _ in segments:
                    pass
            language = self.language if self.language else (info.language if hasattr(info, 'language') else "unknown")

        elif self.model_name == 'openai-whisper':
            result = self.model.transcribe(audio, language=self.language)
            if collect_text:
                transcription = result.get("text", "")
            language = self.language if self.language else result.get("language", "unknown")

        elif self.model_name == 'whisperx':
            result = self.model.transcribe(audio, language=self.language)
            if isinstance(result, dict) and "segments" in result:
                if collect_text:
                    transcription = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
                language = self.language if self.language else result.get("language", "unknown")
            else:
                if collect_text:
                    transcription = result.get("text", "")
                language = self.language if self.language else result.get("language", "unknown")

        return transcription, language

    def _measure_chunk_latency(
        self,
        audio: np.ndarray,
        sample_rate: int,
        chunk_size_s: float,
    ) -> Dict:
        chunk_size_samples = max(1, int(chunk_size_s * sample_rate))
        chunk_latencies_ms = []

        for start in range(0, len(audio), chunk_size_samples):
            chunk = audio[start:start + chunk_size_samples]
            if len(chunk) == 0:
                continue

            chunk_start = time.time()
            self._transcribe_once(chunk, collect_text=False)
            chunk_latency_ms = (time.time() - chunk_start) * 1000.0
            chunk_latencies_ms.append(chunk_latency_ms)

        if not chunk_latencies_ms:
            return {
                "chunk_count": 0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        latencies = np.array(chunk_latencies_ms, dtype=np.float64)
        return {
            "chunk_count": int(len(chunk_latencies_ms)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
    
    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        reference_text: Optional[str] = None,
        chunk_size_s: float = 1.0,
    ) -> Dict:
        """Transcribe audio with the selected model"""
        logger.info("Starting transcription...")
        self._ram_peak_bytes = 0
        self._prepare_vram_monitoring()
        monitor_stop = threading.Event()
        monitor_thread = threading.Thread(target=self._monitor_ram_peak, args=(monitor_stop,), daemon=True)
        monitor_thread.start()
        gpu_monitor_thread = None
        if self.use_gpu:
            gpu_monitor_thread = threading.Thread(target=self._monitor_vram_peak, args=(monitor_stop,), daemon=True)
            gpu_monitor_thread.start()

        start_time = time.time()
        transcription = ""
        language = "unknown"
        processing_time = 0.0
        chunk_metrics = {
            "chunk_count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

        try:
            transcription, language = self._transcribe_once(audio, collect_text=True)
            transcription = self._clean_transcription_text(transcription)
            processing_time = time.time() - start_time

            if self.show_transcription_early:
                print("\n=== TRANSCRIPTION READY (before chunk metrics) ===")
                print(transcription.strip() if transcription.strip() else "(empty transcription)")
                print("=== END TRANSCRIPTION ===\n")

            if self.measure_chunks:
                chunk_metrics = self._measure_chunk_latency(
                    audio,
                    sample_rate,
                    chunk_size_s,
                )
            else:
                logger.info("Chunk latency measurement disabled (use --measure-chunks to enable).")
        finally:
            monitor_stop.set()
            monitor_thread.join(timeout=1.0)
            if gpu_monitor_thread is not None:
                gpu_monitor_thread.join(timeout=1.0)

        audio_duration = len(audio) / sample_rate
        # Standard RTF: processing_time / audio_duration (lower is better)
        rtf = processing_time / audio_duration if audio_duration > 0 else 0

        ram_peak_mb = self._ram_peak_bytes / (1024 * 1024)
        vram_peak_mb = 0.0
        if self.use_gpu:
            vram_peak_mb = self._vram_peak_mb

        wer = self._compute_wer(reference_text, transcription) if reference_text else None
        
        logger.info(f"Transcription complete! RTF (lower is better): {rtf:.2f}")
        
        return {
            "model": self.model_name,
            "model_size": self.model_size,
            "device": self.device,
            "audio_duration_s": round(audio_duration, 2),
            "processing_time_s": round(processing_time, 2),
            "real_time_factor": round(rtf, 2),
            "chunk_size_s": round(chunk_size_s, 2),
            "chunk_count": chunk_metrics["chunk_count"],
            "chunk_latency_ms_p50": round(chunk_metrics["p50_ms"], 2),
            "chunk_latency_ms_p95": round(chunk_metrics["p95_ms"], 2),
            "chunk_latency_ms_p99": round(chunk_metrics["p99_ms"], 2),
            "ram_peak_mb": round(ram_peak_mb, 2),
            "vram_peak_mb": round(vram_peak_mb, 2),
            "wer": round(wer, 4) if wer is not None else None,
            "transcription": transcription,
            "language": language
        }
    
    def print_result(self, result: Dict):
        """Print formatted result"""
        print(f"""
╔════════════════════════════════════════╗
║        BENCHMARK RESULT                ║
╟────────────────────────────────────────╢
║ Model:           {result['model']:<19} 
║ Device:          {result['device']:<19} 
║ Audio Duration:  {result['audio_duration_s']:<1} s 
║ Processing Time: {result['processing_time_s']:<1} s 
║ Real-Time Factor:{result['real_time_factor']:<1} x 
║ Language:        {result['language']:<19} 
╟────────────────────────────────────────
║ Transcription:                         
║ {result['transcription'][:2000]:<39} 
╚════════════════════════════════════════
        """)
        print(f"Chunk Size:       {result['chunk_size_s']} s")
        print(f"Chunk Count:      {result['chunk_count']}")
        print(f"Chunk Latency P50:{result['chunk_latency_ms_p50']} ms")
        print(f"Chunk Latency P95:{result['chunk_latency_ms_p95']} ms")
        print(f"Chunk Latency P99:{result['chunk_latency_ms_p99']} ms")
        print(f"RAM Peak:         {result['ram_peak_mb']} MB")
        print(f"VRAM Peak:        {result['vram_peak_mb']} MB")
        if result['wer'] is not None:
            print(f"WER:              {result['wer']}")
        else:
            print("WER:              N/A (use --reference-text or --reference-file)")
    
    def cleanup(self):
        """Clean up memory"""
        logger.info("Cleaning up memory...")
        if self.use_gpu:
            # Avoid explicit GPU teardown because some backends crash on process cleanup
            # for specific model/compute_type combinations.
            logger.info("Skipping explicit GPU cleanup to avoid backend termination crash.")
            return

        self.model = None
        gc.collect()
    
    def run(self, duration: float, input_file: Optional[str] = None) -> Dict:
        """Run complete benchmark"""
        try:
            if input_file:
                audio = self.load_audio_file(input_file)
            else:
                audio = self.record_audio(duration)

            result = self.transcribe_audio(
                audio,
                reference_text=self.reference_text,
                chunk_size_s=self.chunk_size_s,
            )
            self.print_result(result)
            return result
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
                description='Whisper Benchmark - Compare transcription models with microphone or audio file input',
        epilog="""
Examples:
  python whisper_benchmark_complete.py --list-models
  python whisper_benchmark_complete.py --model faster-whisper --duration 10
    python whisper_benchmark_complete.py --model faster-whisper --input-file audio.wav
  python whisper_benchmark_complete.py --model openai-whisper --duration 10
  python whisper_benchmark_complete.py --model whisperx --duration 10 --output results/result_wx.json
        """
    )
    
    parser.add_argument('--list-models', action='store_true', help='Show available models and exit')
    parser.add_argument('--model', type=str, choices=['faster-whisper', 'openai-whisper', 'whisperx'], help='Model to benchmark')
    parser.add_argument('--model-size', type=str, default='base', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'], help='Whisper model size (default: base)')
    parser.add_argument('--duration', type=float, default=10.0, help='Recording duration in seconds')
    parser.add_argument('--input-file', type=str, default=None, help='Path to an audio file (if set, microphone recording is skipped)')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--output', type=str, default=None, help='Output file for results (JSON format)')
    parser.add_argument('--chunk-size', type=float, default=1.0, help='Chunk size in seconds for latency percentiles')
    parser.add_argument('--measure-chunks', action='store_true', help='Enable chunk latency measurement (slower execution)')
    parser.add_argument('--reference-text', type=str, default=None, help='Reference text for WER calculation')
    parser.add_argument('--reference-file', type=str, default=None, help='Path to a file with reference text for WER')
    parser.add_argument('--show-transcription-early', action='store_true', help='Print full transcription before chunk simulation finishes')
    parser.add_argument('--language', type=str, default=None, choices=['en', 'pt'], help='Force transcription language, skips auto-detection (en/pt)')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\n📋 Available Models:")
        print("  • faster-whisper")
        print("  • openai-whisper")
        print("  • whisperx")
        print()
        return
    
    if not args.model:
        parser.error("--model is required (use --list-models to see options)")
    
    if args.duration <= 0:
        parser.error("Duration must be positive")
    if args.chunk_size <= 0:
        parser.error("Chunk size must be positive")
    if args.input_file and not Path(args.input_file).expanduser().exists():
        parser.error(f"Audio file not found: {args.input_file}")

    reference_text = args.reference_text
    if args.reference_file:
        with open(args.reference_file, 'r', encoding='utf-8') as f:
            file_reference = f.read().strip()
        if reference_text:
            reference_text = f"{reference_text} {file_reference}".strip()
        else:
            reference_text = file_reference
    
    try:
        benchmark = WhisperBenchmark(model_name=args.model, use_gpu=args.use_gpu, model_size=args.model_size)
        benchmark.reference_text = reference_text
        benchmark.chunk_size_s = args.chunk_size
        benchmark.measure_chunks = args.measure_chunks
        benchmark.show_transcription_early = args.show_transcription_early
        benchmark.language = args.language
        result = benchmark.run(duration=args.duration, input_file=args.input_file)
        
        if args.output and result:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"✅ Results saved to {args.output}")
    
    except Exception as e:
        logger.error(f"❌ Error during benchmark: {e}")
        raise


if __name__ == '__main__':
    main()