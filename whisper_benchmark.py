import sounddevice as sd
import numpy as np
import time
import csv
import torch
from memory_profiler import memory_usage

# Import models
# Assuming the models are accessible in the current environment
# from faster_whisper import Whisper
# from insanely_fast_whisper import Whisper
# from whisperx import WhisperX

# Parameters
duration = 10  # in seconds
sample_rate = 16000
output_csv = 'benchmark_results.csv'

def record_audio(duration, sample_rate):
    print(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def benchmark_model(model, audio):
    start_time = time.time()
    # Convert audio for model input, placeholder logic
    input_data = np.array(audio)  # Placeholder preprocessing
    output = model.transcribe(input_data)  # Replace with actual model's transcribe
    processing_time = time.time() - start_time
    memory_consumption = memory_usage((model.transcribe, (input_data,)), max_usage=True)[0]
    # Placeholder for confidence score
    confidence_score = np.random.uniform(0, 1)  # Simulated confidence score

    return processing_time, memory_consumption, confidence_score

def write_results_to_csv(results):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Time (s)', 'Memory (MiB)', 'Confidence'])
        for result in results:
            writer.writerow(result)

def main():
    audio = record_audio(duration, sample_rate)

    models = [None, None, None]  # Replace with actual model instances: [faster_whisper_model, insanely_fast_whisper_model, whisperx_model]
    results = []

    for model in models:
        model_name = model.__class__.__name__ if model else "Unknown"
        time_taken, memory_used, confidence = benchmark_model(model, audio)
        results.append((model_name, time_taken, memory_used, confidence))
    
    write_results_to_csv(results)
    print("Benchmarking completed. Results saved to:", output_csv)

if __name__ == "__main__":
    main()