# Realtime Speech Tests (Faster-Whisper)

Teste de transcricao em tempo real com captura continua do microfone.

## Objetivo

Validar um fluxo de produto com foco em:

- texto cumulativo (append-only)
- baixa repeticao e baixa sobrescrita
- menor perda de palavras em fronteiras de processamento
- metricas de latencia e RTF por execucao

## Dependencias

Para isolar dependencias do realtime, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r realtime_tests/requirements.txt
```

## Execucao recomendada

Via runner PowerShell:

```powershell
.\realtime_tests\start_realtime.ps1 -model base -device gpu -language pt-br -duration 20
```

Ou chamada direta do script:

```powershell
.\.venv\Scripts\python.exe realtime_tests/realtime_benchmark.py --model base --device gpu --language pt-br --duration 20 --context 5
```

## Parametros

- --model: tiny, base, small, medium, large (default = base)
- --device: cpu, gpu (default = cpu)
- --language: en, pt-br (default = pt-br)
- --duration: segundos de gravacao (default = 20s)
- --context: numero de palavras de contexto para prompt (default = 5)
- --para alterar defaults, editar arquivo start_realtime.ps1

## Estrategia atual de processamento

- Captura continua de audio
- VAD simples por energia
- Processamento por enunciado (nao por parcial agressivo)
- Commit append-only
- Boundary overlap em corte forcado
- Tail guard de palavras em corte forcado

## Saidas geradas

Por execucao, sao gerados:

- JSON de resultado em realtime_tests/resultados/
- WAV de auditoria em realtime_tests/resultados/realtime_debug_audio/

## Observacoes de GPU

No modo GPU, o script tenta os compute types:

- float16
- int8_float16
- float32

Se nenhum for suportado no ambiente, o script encerra com erro explicito.
