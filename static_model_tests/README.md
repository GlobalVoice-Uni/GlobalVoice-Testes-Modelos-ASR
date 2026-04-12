# Static Model Tests

Suite de benchmarks estaticos para comparar Faster-Whisper, OpenAI-Whisper e WhisperX em CPU e GPU.

## Proposito

Esta suite existe para produzir comparacoes reproduziveis entre modelos, tamanhos e hardware, com foco em:

- tempo de processamento e RTF
- qualidade de transcricao (WER)
- consumo de RAM e VRAM
- consolidacao de resultados para analise e decisao

## Estrutura

- audios/: audios de entrada para os cenarios padrao
- textos/: referencias de texto (ground truth)
- run/: filas PowerShell para execucao em lote
- resultados/: JSONs e utilitarios de consolidacao
- setup/: setup de ambiente dessa suite
- whisper_benchmark_complete.py: benchmark para modelos Faster-Whisper e OpenAI-Whisper
- whisperx_benchmark.py: benchmark focado em WhisperX
- requirements.txt: dependencias especificas da suite estatica

## Setup

Windows:

```powershell
.\static_model_tests\setup\setup_environment.bat
```

Linux/macOS:

```bash
chmod +x ./static_model_tests/setup/setup_environment.sh
./static_model_tests/setup/setup_environment.sh
```

## Uso diario (com venv ativada)

Depois do setup, ative a venv e rode comandos diretos.

Windows:

```powershell
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## Comandos

Execucao unica (exemplo):

```powershell
python static_model_tests/whisper_benchmark_complete.py --model faster-whisper --duration 10
```

Fila de testes (exemplo):

```powershell
.\static_model_tests\run\faster_whisper_gpu.ps1
```

## Dashboard

Com a venv ativa:

```powershell
python static_model_tests/resultados/importar_resultados.py --folder static_model_tests/resultados/
streamlit run static_model_tests/resultados/dashboard_resultados.py
```
