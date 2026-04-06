# GlobalVoice Testes

## Estrutura do Repositório

- `audios/`: áudios usados nos testes.
- `textos/`: textos de referência (ground truth) por duração.
- `run/`: scripts PowerShell para executar benchmarks por engine/hardware.
- `resultados/`: saídas JSON dos benchmarks e utilitários de consolidação.
- `resultados/resultados_gerais.xlsx`: base consolidada usada para análise.
- `resultados/dashboard_resultados.py`: dashboard Streamlit para comparação dos resultados.
- `setup/`: scripts de preparação de ambiente (Windows e Linux).
- `whisper_benchmark_complete.py`: benchmark principal para OpenAI Whisper e Faster-Whisper.
- `whisperx_benchmark.py`: benchmark específico para WhisperX.
- `requirements.txt`: dependências Python do projeto.

## Propósito

O objetivo do GlobalVoice Testes é padronizar e comparar o desempenho de diferentes stacks de ASR (OpenAI-Whisper, Faster-Whisper e WhisperX), em CPU e GPU, usando cenários reproduzíveis de duração e idioma.

Com isso, o repositório permite:

- medir latência e throughput (Tempo de Processamento e RTF);
- avaliar qualidade de transcrição (WER);
- acompanhar consumo de recursos (RAM e VRAM);
- consolidar os resultados em planilha e dashboard para apoiar decisão de modelo em produção.

# Pra testar no pC

- 1. Clone o repositório
     git clone https://github.com/caiucaindo/GlobalVoice-Testes.git
     cd GlobalVoice-Testes
- 2. Execute o setup (cria venv, instala dependências, cria pasta resultados)
     windows -> ./setup/setup_environment.bat
     linux -> ./setup/setup_environment.sh
     se não tiver permissão, rode antes -> chmod +x ./setup/setup_environment.sh
- 3. Ative o ambiente virtual
     windows -> .\venv\Scripts\activate
     linux -> source .venv/Scripts/activate ou source .venv/bin/activate
- 4. Teste um modelo específico (10 segundos do microfone)
     python whisper_benchmark.py faster-whisper
- 5. Ou teste diversas versões de um mesmo modelo rodando um dos scripts
     com .\run\modelo_device.ps1

## Dashboard de Resultados

Dashboard interativo para visualizar os resultados consolidados em `resultados/resultados_gerais.xlsx`, com foco no idioma EN e sem considerar métricas de chunk.

1. Instale dependências (se ainda não instalou) -> pip install -r requirements.txt
2. Rode o dashboard -> streamlit run resultados/dashboard_resultados.py

Filtros disponíveis no sidebar:

- Hardware
- Tamanho
- Intervalo
- Modelo
