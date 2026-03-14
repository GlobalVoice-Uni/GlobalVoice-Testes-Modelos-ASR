import argparse
import time
import json
import logging
import gc
import warnings
import threading
import re
from typing import Dict, Optional
import numpy as np
import torch
import sounddevice as sd
import psutil

# Suppress torchcodec warnings
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhisperXBenchmark:
    """WhisperX Benchmark - Optimized transcription model with speaker diarization"""
    
    def __init__(self, use_gpu: bool = False, model_size: str = "base"):
        self.model_name = "whisperx"
        self.model_size = model_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        self.process = psutil.Process()
        self._ram_peak_bytes = 0
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Selected model: {self.model_name} ({self.model_size})")
        self._load_model()
    
    def _load_model(self):
        """Load WhisperX model"""
        logger.info(f"Loading {self.model_name} model...")
        
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

    def _transcribe_once(self, audio: np.ndarray, collect_text: bool = True):
        result = self.model.transcribe(audio)

        transcription = ""
        language = "unknown"
        if isinstance(result, dict) and "segments" in result:
            if collect_text:
                transcription = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
            language = result.get("language", "unknown")
        else:
            if collect_text:
                transcription = result.get("text", "")
            language = result.get("language", "unknown")

        return transcription, language

    def _measure_chunk_latency(self, audio: np.ndarray, sample_rate: int, chunk_size_s: float) -> Dict:
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
        """Transcribe audio with WhisperX"""
        logger.info("Starting transcription...")
        self._ram_peak_bytes = 0
        monitor_stop = threading.Event()
        monitor_thread = threading.Thread(target=self._monitor_ram_peak, args=(monitor_stop,), daemon=True)
        monitor_thread.start()

        if self.use_gpu:
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        transcription = ""
        language = "unknown"
        chunk_metrics = {
            "chunk_count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

        try:
            transcription, language = self._transcribe_once(audio, collect_text=True)
            processing_time = time.time() - start_time
            chunk_metrics = self._measure_chunk_latency(audio, sample_rate, chunk_size_s)
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
        finally:
            monitor_stop.set()
            monitor_thread.join(timeout=1.0)

        audio_duration = len(audio) / sample_rate
        # Standard RTF: processing_time / audio_duration (lower is better)
        rtf = processing_time / audio_duration if audio_duration > 0 else 0

        ram_peak_mb = self._ram_peak_bytes / (1024 * 1024)
        vram_peak_mb = 0.0
        if self.use_gpu:
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

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
║ Model:           {result['model']:<21} ║
║ Device:          {result['device']:<21} ║
║ Audio Duration:  {result['audio_duration_s']:<19} s ║
║ Processing Time: {result['processing_time_s']:<19} s ║
║ Real-Time Factor:{result['real_time_factor']:<19} x ║
║ Language:        {result['language']:<21} ║
╟────────────────────────────────────────╢
║ Transcription:                         ║
║ {result['transcription'][:38]:<38} ║
╚════════════════════════════════════════╝
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
        self.model = None
        gc.collect()
        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def run(self, duration: float) -> Dict:
        """Run complete benchmark"""
        try:
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
        description='WhisperX Benchmark - Optimized transcription with speaker diarization',
        epilog="""
Examples:
  python whisperx_benchmark.py --duration 10
  python whisperx_benchmark.py --duration 10 --use-gpu
  python whisperx_benchmark.py --duration 10 --output results/whisperx_result.json
        """
    )
    
    parser.add_argument('--model-size', type=str, default='base', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'], help='Whisper model size (default: base)')
    parser.add_argument('--duration', type=float, default=10.0, help='Recording duration in seconds')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--output', type=str, default=None, help='Output file for results (JSON format)')
    parser.add_argument('--chunk-size', type=float, default=1.0, help='Chunk size in seconds for latency percentiles')
    parser.add_argument('--reference-text', type=str, default=None, help='Reference text for WER calculation')
    parser.add_argument('--reference-file', type=str, default=None, help='Path to a file with reference text for WER')
    
    args = parser.parse_args()
    
    if args.duration <= 0:
        parser.error("Duration must be positive")
    if args.chunk_size <= 0:
        parser.error("Chunk size must be positive")

    reference_text = args.reference_text
    if args.reference_file:
        with open(args.reference_file, 'r', encoding='utf-8') as f:
            file_reference = f.read().strip()
        if reference_text:
            reference_text = f"{reference_text} {file_reference}".strip()
        else:
            reference_text = file_reference
    
    try:
        benchmark = WhisperXBenchmark(use_gpu=args.use_gpu, model_size=args.model_size)
        benchmark.reference_text = reference_text
        benchmark.chunk_size_s = args.chunk_size
        result = benchmark.run(duration=args.duration)
        
        if args.output and result:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"✅ Results saved to {args.output}")
    
    except Exception as e:
        logger.error(f"❌ Error during benchmark: {e}")
        raise


if __name__ == '__main__':
    main()