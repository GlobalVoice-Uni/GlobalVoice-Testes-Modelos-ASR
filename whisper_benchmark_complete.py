import argparse
import time
import json
import logging
import gc
from typing import Dict
import numpy as np
import torch
import sounddevice as sd
from faster_whisper import WhisperModel
import whisper as openai_whisper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhisperBenchmark:
    """Simple Whisper Benchmark - Compare different models with microphone input"""
    
    def __init__(self, model_name: str, use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Selected model: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the selected model"""
        logger.info(f"Loading {self.model_name} model...")
        
        if self.model_name == 'faster-whisper':
            self.model = WhisperModel(
                "base",
                device=self.device,
                compute_type="float16" if self.use_gpu else "int8"
            )
        
        elif self.model_name == 'openai-whisper':
            self.model = openai_whisper.load_model("base", device=self.device)
        
        elif self.model_name == 'insanely-fast-whisper':
            try:
                from insanely_fast_whisper import WhisperModel as IFWModel
                self.model = IFWModel(model_id="openai/whisper-base", device_type=self.device)
            except Exception as e:
                logger.error(f"Error loading insanely-fast-whisper: {e}")
                raise
        
        elif self.model_name == 'whisperx':
            try:
                import whisperx
                self.model = whisperx.load_model("base", device=self.device)
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
            dtype='float32'
        )
        sd.wait()
        
        logger.info("Recording complete!")
        return audio.flatten()
    
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Transcribe audio with the selected model"""
        logger.info("Starting transcription...")
        
        start_time = time.time()
        transcription = ""
        language = "unknown"
        
        if self.model_name == 'faster-whisper':
            segments, info = self.model.transcribe(audio)
            transcription = " ".join([segment.text for segment in segments])
            language = info.language if hasattr(info, 'language') else "unknown"
        
        elif self.model_name == 'openai-whisper':
            result = self.model.transcribe(audio)
            transcription = result.get("text", "")
            language = result.get("language", "unknown")
        
        elif self.model_name == 'insanely-fast-whisper':
            result = self.model.transcribe(audio)
            transcription = result.get("text", "")
            language = result.get("language", "unknown")
        
        elif self.model_name == 'whisperx':
            result = self.model.transcribe(audio)
            if isinstance(result, dict):
                transcription = result.get("text", "")
                language = result.get("language", "unknown")
            else:
                transcription = str(result)
        
        processing_time = time.time() - start_time
        audio_duration = len(audio) / sample_rate
        rtf = audio_duration / processing_time if processing_time > 0 else 0
        
        logger.info(f"Transcription complete! RTF: {rtf:.2f}x")
        
        return {
            "model": self.model_name,
            "device": self.device,
            "audio_duration_s": round(audio_duration, 2),
            "processing_time_s": round(processing_time, 2),
            "real_time_factor": round(rtf, 2),
            "transcription": transcription,
            "language": language
        }
    
    def print_result(self, result: Dict):
        """Print formatted result"""
        print(f"""
╔════════════════════════════════════════╗
║        BENCHMARK RESULT                ║
╟────────────────────────────────────────╢
║ Model:           {result['model']:<19} ║
║ Device:          {result['device']:<19} ║
║ Audio Duration:  {result['audio_duration_s']:<19} s ║
║ Processing Time: {result['processing_time_s']:<19} s ║
║ Real-Time Factor:{result['real_time_factor']:<19} x ║
║ Language:        {result['language']:<19} ║
╟────────────────────────────────────────╢
║ Transcription:                         ║
║ {result['transcription'][:38]:<39} ║
╚════════════════════════════════════════╝
        """)
    
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
            result = self.transcribe_audio(audio)
            self.print_result(result)
            return result
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description='Whisper Benchmark - Compare transcription models with microphone input',
        epilog="""
Examples:
  python whisper_benchmark_complete.py --list-models
  python whisper_benchmark_complete.py --model faster-whisper --duration 10
  python whisper_benchmark_complete.py --model whisperx --duration 10 --use-gpu
  python whisper_benchmark_complete.py --model insanely-fast-whisper --duration 10 --output result.json
        """
    )
    
    parser.add_argument('--list-models', action='store_true', help='Show available models and exit')
    parser.add_argument('--model', type=str, choices=['faster-whisper', 'openai-whisper', 'insanely-fast-whisper', 'whisperx'], help='Model to benchmark')
    parser.add_argument('--duration', type=float, default=10.0, help='Recording duration in seconds')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--output', type=str, default=None, help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\n📋 Available Models:")
        print("  • faster-whisper")
        print("  • openai-whisper")
        print("  • insanely-fast-whisper")
        print("  • whisperx")
        print()
        return
    
    if not args.model:
        parser.error("--model is required (use --list-models to see options)")
    
    if args.duration <= 0:
        parser.error("Duration must be positive")
    
    try:
        benchmark = WhisperBenchmark(model_name=args.model, use_gpu=args.use_gpu)
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