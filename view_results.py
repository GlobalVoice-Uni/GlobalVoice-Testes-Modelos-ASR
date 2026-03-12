import json
import sys
from pathlib import Path

def view_result(filepath):
    """View benchmark result in a readable format"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n" + "╔" + "═"*48 + "╗")
        print(f"║ BENCHMARK RESULT - {data['model']:<27} ║")
        print("╟" + "─"*48 + "╢")
        print(f"║ Device:          {data['device']:<29} ║")
        print(f"║ Audio Duration:  {data['audio_duration_s']:<27} s ║")
        print(f"║ Processing Time: {data['processing_time_s']:<27} s ║")
        print(f"║ Real-Time Factor:{data['real_time_factor']:<27} x ║")
        print(f"║ Language:        {data['language']:<29} ║")
        print("╟" + "─"*48 + "╢")
        print("║ TRANSCRIPTION:                                 ║")
        print("╟" + "─"*48 + "╢")
        
        # Print transcription with word wrapping
        text = data['transcription']
        for i in range(0, len(text), 46):
            line = text[i:i+46]
            print(f"║ {line:<46} ║")
        
        print("╚" + "═"*48 + "╝\n")
        
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {filepath}")
    except json.JSONDecodeError:
        print(f"❌ Erro ao ler JSON: {filepath}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python view_results.py <arquivo.json>")
        print("Exemplo: python view_results.py results/result_fw.json")
    else:
        view_result(sys.argv[1])