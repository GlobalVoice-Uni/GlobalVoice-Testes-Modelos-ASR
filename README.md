# GlobalVoice Testes

## Project Structure

- `docs/`: Contains documentation files
- `src/`: Contains source code
- `tests/`: Contains test files
- `README.md`: Project overview and instructions

## Purpose

The purpose of the GlobalVoice Testes project is to provide a framework for testing the GlobalVoice application. This project aims to facilitate the development and deployment of testing protocols to ensure the quality and reliability of the application.

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

1. Instale dependências (se ainda não instalou):
   pip install -r requirements.txt
2. Rode o dashboard -> streamlit run dashboard_resultados.py

Filtros disponíveis no sidebar:

- Hardware
- Tamanho
- Intervalo
- Modelo
