# GlobalVoice Testes

Repositorio de validacao ASR com duas frentes complementares:

- static_model_tests: benchmarks estaticos comparativos entre modelos.
- realtime_tests: benchmark de transcricao em tempo real (append-only).

## Visao geral

A ideia e separar os objetivos:

- estatico para comparabilidade e consolidacao de metricas
- realtime para comportamento continuo de produto

Assim cada frente evolui com parametros e dependencias proprias, sem misturar contexto.

## Estrutura

- static_model_tests/
  - benchmarks e filas de testes estaticos
  - setup proprio
  - requisitos proprios
  - resultados e dashboard
- realtime_tests/
  - benchmark realtime
  - runner realtime
  - requisitos e documentacao proprios
  - resultados realtime

## Setup inicial

1. Criar venv na raiz:

```powershell
python -m venv .venv
```

2. Ativar venv:

```powershell
.\.venv\Scripts\activate
```

3. Instalar dependencias de cada frente conforme necessidade:

```powershell
pip install -r static_model_tests/requirements.txt
pip install -r realtime_tests/requirements.txt
```

## Execucao rapida

Com venv ativada.

Estatico (exemplo de fila):

```powershell
.\static_model_tests\run\faster_whisper_gpu.ps1
```

Realtime (exemplo):

```powershell
.\realtime_tests\start_realtime.ps1 -model base -device gpu -language pt-br -duration 20
```
