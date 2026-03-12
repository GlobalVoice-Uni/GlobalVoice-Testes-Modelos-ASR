import argparse
import time
import json
import logging
import gc
import warnings
from typing import Dict
import numpy as np
import torch
import sounddevice as sd

# Suppress torchcodec warnings
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhisperXBenchmark:
    """WhisperX Benchmark - Optimized transcription model with speaker diarization"""
    
    def __init__(self, use_gpu: bool = False):
        self.model_name = "whisperx"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Selected model: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load WhisperX model"""
        logger.info(f"Loading {self.model_name} model...")
        
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
        """Transcribe audio with WhisperX"""
        logger.info("Starting transcription...")
        
        start_time = time.time()
        transcription = ""
        language = "unknown"
        
        try:
            result = self.model.transcribe(audio)
            
            # WhisperX returns segments, not direct text
            if isinstance(result, dict) and "segments" in result:
                transcription = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
                language = result.get("language", "unknown")
            else:
                transcription = result.get("text", "")
                language = result.get("language", "unknown")
        
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
        
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
        description='WhisperX Benchmark - Optimized transcription with speaker diarization',
        epilog="""
Examples:
  python whisperx_benchmark.py --duration 10
  python whisperx_benchmark.py --duration 10 --use-gpu
  python whisperx_benchmark.py --duration 10 --output results/whisperx_result.json
        """
    )
    
    parser.add_argument('--duration', type=float, default=10.0, help='Recording duration in seconds')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--output', type=str, default=None, help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    if args.duration <= 0:
        parser.error("Duration must be positive")
    
    try:
        benchmark = WhisperXBenchmark(use_gpu=args.use_gpu)
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