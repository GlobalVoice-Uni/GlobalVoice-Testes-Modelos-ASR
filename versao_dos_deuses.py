import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import librosa
import soundfile as sf
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Store benchmark results for a model"""
    model_name: str
    audio_duration: float
    processing_time: float
    inference_time: float
    memory_used_mb: float
    device: str
    chunk_count: int
    accuracy_score: Optional[float] = None
    
    def get_throughput(self) -> float:
        """Calculate audio-to-real-time ratio"""
        return self.audio_duration / self.processing_time if self.processing_time > 0 else 0


class WhisperBenchmark:
    """Benchmark different Whisper implementations"""
    
    MODELS = {
        'faster-whisper': 'faster_whisper',
        'openai-whisper': 'openai_whisper',
        'insanely-fast-whisper': 'insanely_fast_whisper',
        'whisperx': 'whisperx'
    }
    
    def __init__(self, use_gpu: bool = False, chunk_duration: float = 3.0):
        """Initialize benchmark environment"""
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.chunk_duration = chunk_duration
        self.results: List[BenchmarkResult] = []
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Chunk duration: {self.chunk_duration}s")
        
    def generate_test_audio(self, duration: float, sample_rate: int = 16000) -> np.ndarray:
        """Generate synthetic test audio (1kHz sine wave)"""
        logger.info(f"Generating {duration}s test audio...")
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 1000 * t)  # 1kHz sine wave
        return audio.astype(np.float32)
    
    def load_or_create_audio(self, audio_path: Optional[str], duration: float, 
                             sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
        """Load audio from file or generate synthetic audio"""
        if audio_path and os.path.exists(audio_path):
            logger.info(f"Loading audio from {audio_path}...")
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            audio = audio[:int(sr * duration)]  # Trim to duration
            return audio, sr
        else:
            return self.generate_test_audio(duration, sample_rate), sample_rate
    
    def process_chunks(self, audio: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """Split audio into chunks"""
        chunk_size = int(self.chunk_duration * sample_rate)
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunks.append(audio[i:i + chunk_size])
        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks
    
    def benchmark_faster_whisper(self, audio: np.ndarray, sample_rate: int) -> BenchmarkResult:
        """Benchmark Faster Whisper"""
        try:
            from faster_whisper import WhisperModel
            
            logger.info("Loading Faster Whisper model...")
            model = WhisperModel("base", device=self.device, compute_type="float16" if self.use_gpu else "int8")
            
            chunks = self.process_chunks(audio, sample_rate)
            
            # Memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 ** 2)
            
            # Benchmark
            start = time.time()
            for chunk in chunks:
                segments, info = model.transcribe(chunk)
                list(segments)  # Force evaluation
            processing_time = time.time() - start
            
            # Memory after
            mem_after = process.memory_info().rss / (1024 ** 2)
            memory_used = mem_after - mem_before
            
            return BenchmarkResult(
                model_name="faster-whisper",
                audio_duration=len(audio) / sample_rate,
                processing_time=processing_time,
                inference_time=processing_time,
                memory_used_mb=max(0, memory_used),
                device=self.device,
                chunk_count=len(chunks)
            )
        except Exception as e:
            logger.error(f"Error benchmarking Faster Whisper: {e}")
            return None
    
    def benchmark_openai_whisper(self, audio: np.ndarray, sample_rate: int) -> BenchmarkResult:
        """Benchmark OpenAI Whisper"""
        try:
            import whisper
            
            logger.info("Loading OpenAI Whisper model...")
            model = whisper.load_model("base", device=self.device)
            
            chunks = self.process_chunks(audio, sample_rate)
            
            # Memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 ** 2)
            
            # Benchmark
            start = time.time()
            for chunk in chunks:
                result = model.transcribe(chunk)
            processing_time = time.time() - start
            
            # Memory after
            mem_after = process.memory_info().rss / (1024 ** 2)
            memory_used = mem_after - mem_before
            
            return BenchmarkResult(
                model_name="openai-whisper",
                audio_duration=len(audio) / sample_rate,
                processing_time=processing_time,
                inference_time=processing_time,
                memory_used_mb=max(0, memory_used),
                device=self.device,
                chunk_count=len(chunks)
            )
        except Exception as e:
            logger.error(f"Error benchmarking OpenAI Whisper: {e}")
            return None
    
    def benchmark_all_models(self, audio: np.ndarray, sample_rate: int) -> List[BenchmarkResult]:
        """Run benchmarks for all available models"""
        results = []
        
        # Faster Whisper
        logger.info("\n=== Benchmarking Faster Whisper ===")
        result = self.benchmark_faster_whisper(audio, sample_rate)
        if result:
            results.append(result)
            self.print_result(result)
        
        # OpenAI Whisper
        logger.info("\n=== Benchmarking OpenAI Whisper ===")
        result = self.benchmark_openai_whisper(audio, sample_rate)
        if result:
            results.append(result)
            self.print_result(result)
        
        return results
    
    def print_result(self, result: BenchmarkResult):
        """Print formatted benchmark result"""
        print(f"""
╔══════════════════════════════════════╗
║  Model: {result.model_name:<27} ║
╟──────────────────────────────────────╢
║ Audio Duration:      {result.audio_duration:>6.2f}s  ║
║ Processing Time:     {result.processing_time:>6.2f}s  ║
║ Real-Time Factor:    {result.get_throughput():>6.2f}x  ║
║ Memory Used:         {result.memory_used_mb:>6.1f}MB ║
║ Device:              {result.device:>10}  ║
║ Chunks Processed:    {result.chunk_count:>6d}    ║
╚══════════════════════════════════════╝
        """)
    
    def save_results(self, output_file: str):
        """Save results to JSON"""
        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    def run_benchmark(self, duration: float, audio_path: Optional[str] = None, 
                     models: str = "all") -> List[BenchmarkResult]:
        """Run complete benchmark"""
        audio, sample_rate = self.load_or_create_audio(audio_path, duration)
        
        if models == "all":
            results = self.benchmark_all_models(audio, sample_rate)
        else:
            model_list = models.split(",")
            results = []
            for model in model_list:
                model = model.strip()
                if model == "faster-whisper":
                    result = self.benchmark_faster_whisper(audio, sample_rate)
                elif model == "openai-whisper":
                    result = self.benchmark_openai_whisper(audio, sample_rate)
                else:
                    logger.warning(f"Model {model} not supported")
                    continue
                
                if result:
                    results.append(result)
                    self.print_result(result)
        
        self.results = results
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark different Whisper implementations for real-time speech recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all models with GPU for 30 seconds
  python whisper_benchmark_complete.py --duration 30 --use-gpu

  # Benchmark only Faster Whisper with CPU
  python whisper_benchmark_complete.py --model faster-whisper --duration 10

  # Use custom audio file
  python whisper_benchmark_complete.py --audio-path speech.wav --use-gpu

  # Save results to JSON
  python whisper_benchmark_complete.py --output results.json --use-gpu
        """
    )
    
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Audio duration to benchmark in seconds (default: 10.0)')
    parser.add_argument('--model', type=str, default='all',
                        choices=['faster-whisper', 'openai-whisper', 'all'],
                        help='Which model(s) to benchmark (default: all)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for inference if available (default: CPU)')
    parser.add_argument('--chunk-duration', type=float, default=3.0,
                        help='Audio chunk duration for processing in seconds (default: 3.0)')
    parser.add_argument('--audio-path', type=str, default=None,
                        help='Path to audio file to benchmark (optional, generates synthetic audio if not provided)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results in JSON format (optional)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration <= 0:
        parser.error("Duration must be positive")
    if args.chunk_duration <= 0:
        parser.error("Chunk duration must be positive")
    
    # Run benchmark
    benchmark = WhisperBenchmark(use_gpu=args.use_gpu, chunk_duration=args.chunk_duration)
    results = benchmark.run_benchmark(
        duration=args.duration,
        audio_path=args.audio_path,
        models=args.model
    )
    
    # Save results if requested
    if args.output:
        benchmark.results = results
        benchmark.save_results(args.output)
    
    # Print summary
    if results:
        logger.info("\n=== BENCHMARK SUMMARY ===")
        best = max(results, key=lambda r: r.get_throughput())
        logger.info(f"Best performance: {best.model_name} ({best.get_throughput():.2f}x RTF)")


if __name__ == '__main__':
    main()