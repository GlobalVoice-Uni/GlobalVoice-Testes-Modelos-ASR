# Relatorio Tecnico - Teste Modelos ASR (Static Model Tests)

## 1. Objetivo

Este relatorio documenta a fase inicial de validacao tecnica do projeto, dedicada a comparar variantes da familia Whisper e selecionar a base ASR mais adequada para evolucao do sistema, cobrindo:

- objetivo funcional do produto;
- linha do tempo tecnica dos testes estaticos;
- metodologia de benchmark adotada;
- consolidacao das metricas em planilha/dashboard;
- resultados finais e decisao de modelo para continuidade tecnica da plataforma.

Escopo de dados considerado:

- `static_model_tests/resultados/resultados_gerais.xlsx` (aba `in`);
- JSONs de `static_model_tests/resultados/**`;
- scripts e runners da suite estatica;
- historico de decisoes tecnicas registradas durante a evolucao do projeto.

---

## 2. Contexto do Problema

A proposta do projeto e habilitar traducao simultanea bidirecional em reunioes, com cadeia de processamento em tempo real:

1. captura de audio;
2. transcricao (ASR);
3. traducao;
4. sintese de voz (TTS).

Como base tecnica para essa cadeia, a equipe precisou escolher um modelo ASR confiavel, medindo desempenho e qualidade em condicoes reproduziveis.

---

## 3. Linha do Tempo Tecnica (Pre-Realtime)

## 3.1. Fase Inicial - Benchmark com microfone

Os benchmarks (`whisper_benchmark_complete.py` e `whisperx_benchmark.py`) nasceram com dois modos de entrada:

- microfone (`record_audio`);
- arquivo de audio (`--input-file`).

No inicio, o caminho natural foi gravar no microfone para validar fim-a-fim rapidamente.

Problema observado:

- para comparacao em larga escala, gravar manualmente em cada rodada era inviavel;
- havia alta variacao de fala entre tentativas (ritmo, pronuncia, pausas), prejudicando comparabilidade entre modelos.

## 3.2. Fase de Escalabilidade - Matriz de testes

Com o escopo definido, a bateria estatica ficou grande:

- 3 modelos: Faster-Whisper, OpenAI-Whisper, WhisperX;
- 4 tamanhos: tiny, base, small, medium;
- 2 dispositivos: CPU e GPU;
- 3 duracoes: 10s, 30s, 60s.
- 2 idiomas: EN (ingles) e BR (portugues).

Total planejado e executado: **72 execucoes em EN + 72 em BR = 144 execucoes**.

Conclusao: manter entrada por microfone para tudo seria operacionalmente impraticavel.

## 3.3. Padronizacao metodologica - Audios fixos + ground truth

A suite migrou para execucao padronizada com:

- audios fixos em EN: `static_model_tests/audios/10sEN.mp4`, `30sEN.mp4`, `60sEN.mp4`
- audios fixos em BR: `static_model_tests/audios/10sBR.mp4`, `30sBR.mp4`, `60sBR.mp4`
- textos de referencia em EN: `static_model_tests/textos/10sEN.txt`, `30sEN.txt`, `60sEN.txt`
- textos de referencia em BR: `static_model_tests/textos/10sBR.txt`, `30sBR.txt`, `60sBR.txt`
- comparacao automatica de transcricao via WER (`--reference-file`).

Esse foi o ponto que tornou os resultados realmente comparaveis entre modelos/hardware.

## 3.4. Automacao em lote

Foram criados runners PowerShell para cada engine/dispositivo, com suporte a multiplos idiomas:

**Para ingles (EN):**

- `static_model_tests/run/faster_whisper_cpu.ps1`
- `static_model_tests/run/faster_whisper_gpu.ps1`
- `static_model_tests/run/openai_cpu.ps1`
- `static_model_tests/run/openai_gpu.ps1`
- `static_model_tests/run/whisperx_cpu.ps1`
- `static_model_tests/run/whisperx_gpu.ps1`

**Para portugues (BR):**

- `static_model_tests/run_br/faster_whisper_cpu.ps1`
- `static_model_tests/run_br/faster_whisper_gpu.ps1`
- `static_model_tests/run_br/openai_cpu.ps1`
- `static_model_tests/run_br/openai_gpu.ps1`
- `static_model_tests/run_br/whisperx_cpu.ps1`
- `static_model_tests/run_br/whisperx_gpu.ps1`

Cada runner executa automaticamente as 12 combinacoes internas (4 tamanhos x 3 duracoes) com saida JSON organizada por pasta, identificadas por idioma (en/br).

## 3.5. Tentativa de simulacao por chunks e descarte da metrica

Os scripts suportam medicao de latencia por chunk (`--measure-chunks`), com p50/p95/p99.

Na pratica, esse caminho foi considerado inadequado para decisao final porque:

- adicionava overhead relevante na execucao;
- nao representava bem comportamento de streaming real de produto;
- degradava o ciclo de benchmark sem entregar sinal confiavel para comparacao.

Evidencia na base final estatica (apos testes em EN e BR):

- `chunk_count` ficou **0 em 144/144 JSONs (72 EN + 72 BR)**;
- colunas `Chunk_p50_ms`, `Chunk_p95_ms`, `Chunk_p99_ms` ficaram **vazias na planilha**.

Resultado: as metricas de chunk foram formalmente ignoradas no dashboard e na consolidacao final.

## 3.6. Consolidacao em planilha

Foi criado `static_model_tests/resultados/importar_resultados.py` para importar JSONs para `resultados_gerais.xlsx`, com:

- mapeamento de chave por (modelo, hardware, tamanho, idioma, intervalo);
- suporte a multiplos idiomas (EN, BR) com mapeamento automatico en→EN, pt→BR;
- `--dry-run` para simulacao sem escrita;
- backup automatico da planilha antes de salvar;
- importacao apenas de Tempo, RTF, WER, RAM e VRAM (colunas de chunk removidas da planilha como nao utilizadas).

## 3.7. Camada analitica (dashboard)

O dashboard em Streamlit (`static_model_tests/resultados/dashboard_resultados.py`) consolidou o fechamento da fase estatica com:

- suporte a multiplos idiomas (EN, BR) com filtro dinamico na barra lateral;
- filtros por idioma, modelo, hardware, tamanho e intervalo;
- comparativos por metrica;
- ranking agregado;
- recomendacao orientada a streaming (score ponderado);
- metricas de exposicao atualizadas dinamicamente conforme idioma selecionado.

Nesse ponto foi possivel fechar a fase pre-realtime com uma escolha tecnica unica, baseada no conjunto completo EN+BR ja consolidado.

---

## 4. Metodologia de Medicao (Fase Estatica)

Metricas principais usadas para decisao:

- `Tempo_Processamento_s`
- `RTF = processing_time / audio_duration` (menor e melhor)
- `WER` (menor e melhor)
- `Pico_RAM_MB`
- `Pico_VRAM_MB`

Regras adotadas:

- comparacao principal baseada em EN, com validacao cruzada em BR;
- mesmas entradas de audio para todos os modelos em cada idioma;
- mesma malha de combinacoes (modelo/tamanho/hardware/intervalo) para EN e BR;
- convergencia de ranking entre EN e BR no resultado final;
- chunks excluidos da decisao final (metricas nao utilizadas).

---

## 5. Cobertura dos Dados Estaticos

| Item                            | EN   | BR   | Total |
| ------------------------------- | ---- | ---- | ----: |
| Linhas na planilha              | 72   | 72   |   144 |
| Linhas com metricas preenchidas | 72   | 72   |   144 |
| Cobertura esperada (3x4x2x3)    | 72   | 72   |   144 |
| Cobertura final                 | 100% | 100% |  100% |
| JSONs estaticos encontrados     | 72   | 72   |   144 |

**Nota:** EN e BR possuem cobertura completa (72+72) e sustentam a mesma conclusao de selecao de modelo.

---

## 6. Resultados Consolidados

## 6.1. Medias por modelo e hardware

| Modelo         | Hardware | Tempo (s) |    RTF |    WER |  RAM (MB) | VRAM (MB) |
| -------------- | -------- | --------: | -----: | -----: | --------: | --------: |
| Faster-Whisper | CPU      |    9.4242 | 0.3342 | 0.0809 | 1005.2333 |    0.0000 |
| Faster-Whisper | GPU      |    8.6933 | 0.2817 | 0.0880 | 1322.1550 |  139.9167 |
| OpenAI-Whisper | CPU      |   10.5400 | 0.3767 | 0.0926 | 2143.4175 |    0.0000 |
| OpenAI-Whisper | GPU      |    8.2692 | 0.2817 | 0.0926 | 2132.0408 |   22.1667 |
| WhisperX       | CPU      |   17.7325 | 0.5258 | 0.0856 | 2386.7000 |    0.0000 |
| WhisperX       | GPU      |    7.2625 | 0.2550 | 0.0856 | 2016.7233 |  306.0000 |

Leituras principais:

- melhor RTF medio por par modelo/hardware: **WhisperX GPU (0.2550)**;
- melhor WER medio por par modelo/hardware: **Faster-Whisper CPU (0.0809)**;
- menor RAM media geral: **Faster-Whisper CPU**;
- melhor equilibrio geral entre velocidade/qualidade/custo de memoria: **familia Faster-Whisper**.

## 6.2. Medias globais por modelo (CPU+GPU)

| Modelo         | Tempo (s) |    RTF |    WER |  RAM (MB) | VRAM (MB) |
| -------------- | --------: | -----: | -----: | --------: | --------: |
| Faster-Whisper |    9.0588 | 0.3079 | 0.0844 | 1163.6942 |   69.9583 |
| OpenAI-Whisper |    9.4046 | 0.3292 | 0.0926 | 2137.7292 |   11.0833 |
| WhisperX       |   12.4975 | 0.3904 | 0.0856 | 2201.7117 |  153.0000 |

Leitura: no agregado de toda a matriz estatica, **Faster-Whisper** ficou melhor no compromisso geral.

## 6.3. Tendencias por tamanho

| Tamanho | RTF medio | WER medio | Tempo medio (s) | RAM media (MB) |
| ------- | --------: | --------: | --------------: | -------------: |
| tiny    |    0.0633 |    0.1321 |          1.7639 |       951.0311 |
| base    |    0.0939 |    0.1053 |          2.7094 |      1044.9944 |
| small   |    0.2706 |    0.0593 |          8.5928 |      1538.1939 |
| medium  |    0.9422 |    0.0534 |         28.2150 |      3803.2939 |

Trade-off claro:

- `tiny/base`: mais rapidos, piores em WER;
- `small/medium`: melhor WER, custo alto de latencia e memoria;
- `medium` se aproxima de limite de uso realtime em varios cenarios.

## 6.4. Tendencias por intervalo de audio

| Intervalo (s) | RTF medio | WER medio | Tempo medio (s) |
| ------------: | --------: | --------: | --------------: |
|            10 |    0.4254 |    0.1506 |          4.6125 |
|            30 |    0.2621 |    0.1020 |          7.3262 |
|            60 |    0.3400 |    0.0100 |         19.0221 |

Observacao: o WER muito baixo em 60s indica forte influencia do conteudo especifico do audio de referencia (nao necessariamente generaliza para todos os cenarios).

## 6.5. Ganho medio CPU -> GPU (RTF)

Resumo geral:

- media: **2.303x**
- mediana: **1.960x**
- minimo: **0.651x**
- maximo: **9.400x**

Media por modelo:

- Faster-Whisper: **2.152x**
- OpenAI-Whisper: **1.621x**
- WhisperX: **3.136x**

Leitura: GPU ajuda muito, mas com variabilidade por combinacao; WhisperX e o que mais se beneficia de GPU.

---

## 7. Fechamento da Fase Estatica e Escolha para Avancar

No dashboard foi adotado um score para cenario de streaming (transcricao + traducao + fala), com pesos:

- 45% RTF
- 30% WER
- 15% Tempo de processamento
- 7% RAM
- 3% VRAM

Top combinacoes no fechamento pre-realtime:

| Ranking | Modelo         | Hardware | Tamanho |    RTF |    WER | Score_Streaming |
| ------: | -------------- | -------- | ------- | -----: | -----: | --------------: |
|       1 | Faster-Whisper | GPU      | small   | 0.0900 | 0.0615 |          0.0843 |
|       2 | OpenAI-Whisper | GPU      | small   | 0.1467 | 0.0615 |          0.1174 |
|       3 | WhisperX       | GPU      | small   | 0.1433 | 0.0615 |          0.1223 |
|       4 | Faster-Whisper | CPU      | small   | 0.3233 | 0.0486 |          0.1433 |
|       5 | Faster-Whisper | CPU      | base    | 0.1033 | 0.1010 |          0.2056 |

Decisao tecnica de continuidade (antes do realtime):

- **Modelo de referencia para seguir em testes de produto:** Faster-Whisper
- **Configuracao inicial recomendada para streaming local com GPU:** `small + GPU`

Isso explica a abertura da frente `realtime_tests` com foco em faster-whisper.

---

## 8. Ponto de Arquitetura (Discussao Pre-Realtime)

Nas discussoes pre-realtime, ficou claro que:

- thin client com ASR no servidor melhora previsibilidade, mas aumenta custo e complexidade operacional;
- local-first/hibrido reduz custo inicial, mas depende do hardware do usuario;
- para o momento do projeto, a estrategia pragmatica foi validar primeiro desempenho real com stack local otimizada, antes de assumir custo fixo de servidor.

---

## 9. Limitacoes da Fase Estatica

- corpus limitado aos audios de referencia definidos para EN e BR;
- apenas 3 duracoes de audio (10/30/60) por idioma;
- chunks descartados como metrica de decisao final (nao representam streaming real);
- benchmark estatico nao mede latencia de rede nem concorrencia multiusuario;
- apesar da convergencia EN/BR nesta bateria, resultados podem variar com corpus maior e cenarios acusticos diferentes.

---

## 10. Conclusao Executiva (Pre-Realtime)

A fase estatica atingiu o objetivo com cobertura completa e metodologia reproduzivel, expandindo para multiplos idiomas.

Principais conclusoes:

1. O processo evoluiu corretamente de microfone manual para entradas padronizadas, removendo vies de comparacao.
2. A automacao em lote viabilizou escala experimental: **72 combinacoes EN + 72 em BR = 144 testes** executados com consistencia antes da decisao tecnica.
3. A decisao de ignorar chunks foi tecnicamente correta para esse escopo, dado baixo sinal e alto custo operacional.
4. Faster-Whisper se destacou no equilibrio geral e foi a escolha consistente em EN e BR para iniciar a fase realtime.
5. Dashboard atualizado com suporte dinamico a multiplos idiomas, permitindo comparacao cruzada EN vs BR sem separar decisao por idioma.

Em resumo: a linha pre-realtime fechou com criterio tecnico suficiente para transicao de benchmark estatico para validacao de comportamento continuo de produto, com base consolidada EN+BR e conclusao unica de selecao de modelo.
