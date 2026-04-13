# Relatorio Tecnico de Evolucao - Realtime Speech

Data: 2026-04-11
Projeto: GlobalVoice-Testes

## 1. Objetivo atual

O foco da linha realtime passou a ser claramente este:

- priorizar transcricao correta entre trechos de audio;
- reduzir cortes e repeticoes na fronteira entre partes;
- manter latencia aceitavel, mas sem sacrificar qualidade textual.

Os testes mais recentes mostraram que a principal dor nao e mais medir velocidade de forma isolada, e sim evitar degradacao textual entre blocos.

## 2. O que ficou no script realtime

O benchmark realtime atual ficou centrado em:

- captura continua do microfone;
- formacao de enunciados por fala/silencio;
- transcricao por Faster-Whisper;
- commit append-only do texto confirmado;
- protecoes de fronteira com overlap e tail guard;
- persistencia de JSON e WAV para auditoria.

As metricas de first part latency e tempo de cada parte foram retiradas do script, porque hoje os chunks ja dao a visao principal para comparacao de desempenho.

## 3. Linha do tempo tecnica resumida

## Fase A - Primeira versao realtime

Objetivo:

- sair do benchmark estatico para um fluxo de captura e transcricao ao vivo.

Resultado:

- funcionou como base inicial, mas ainda sem estabilidade de fronteira.

## Fase B - Janela parcial e regressao

Objetivo:

- atualizar texto em tempo real sem perder contexto.

Problema observado:

- repeticoes e sobrescritas crescentes.
- ruido visual no terminal.
- aumento de custo computacional local durante a exibicao parcial continua.
- perda de conteudo em trechos especificos, com reducao anomala da duracao efetiva do audio processado.

Diagnostico importante desta fase:

- foi adicionado salvamento de WAV de debug por execucao;
- a auditoria desses WAVs evidenciou discrepancias de duracao (por exemplo, janelas esperadas de 10s chegando com ~6s em alguns casos);
- a analise apontou que o problema estava no fluxo do script (segmentacao/janelamento e controle de silencio), e nao na captura do microfone.

Conclusao:

- a estrategia de parcial agressivo nao serviu para o caso final.

## Fase C - Commit por enunciado

Mudanca principal:

- abandono da janela deslizante com parcial agressivo;
- troca para commit por enunciado com VAD (Voice Activity Detection) simples.

Efeito:

- reduziu bastante as repeticoes e deixou o texto mais util para uso real.
- reduziu tambem o custo operacional do processamento em tempo real, ao remover a politica de janela parcial continua que elevava a carga local.

## Fase D - Tratamento de fronteira

Problema remanescente:

- palavras cortadas ou omitidas em cortes forcados.

Melhorias aplicadas:

- boundary overlap de audio;
- tail guard textual para segurar palavras instaveis na fronteira.

Efeito:

- melhorou a continuidade, mas ainda existe erro de transicao em alguns cenarios.

## 4. Parâmetros testados e conclusao pratica

Nos testes recentes, os parametros que mais importaram foram:

- `context_window`
- `boundary_overlap_s`
- `tail_guard_words`

Os parametros de fala/silencio e maximo de enunciado ja haviam sido afinados antes e ficaram como base:

- `step_duration_s = 0.2`
- `max_silence_inside_utterance_s = 0.4`
- `max_utterance_s = 3.2`
- `boundary_overlap_s = 0.45`
- `tail_guard_words = 4`

## 5. Comparacao dos resultados mais recentes

### 5.1 Contexto 0

Arquivos:

- `realtime_tests/resultados/realtime-small-gpu-pt-0ctx-20260411-212934.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-0ctx-20260411-213003.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-0ctx-20260411-213036.json`

Leitura:

- first part latency media em torno de 4.8s;
- latencia media de transcricao em torno de 27ms;
- sem loops agressivos como os vistos em alguns testes com contexto maior;
- foi o grupo mais estavel semanticamente no conjunto recente.

### 5.2 Contexto 3

Arquivos:

- `realtime_tests/resultados/realtime-small-gpu-pt-3ctx-test1.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-3ctx-test2.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-3ctx-test3.json`

Leitura:

- first part latency um pouco melhor em media do que 0ctx;
- latencia de transcricao parecida com 0ctx;
- porem apareceu um run claramente degradado, com repeticao forte e drift textual.

Conclusao:

- 3ctx pode parecer bom em um run isolado, mas ainda e instavel para baseline.

### 5.3 Contexto 5

Arquivos mais recentes:

- `realtime_tests/resultados/realtime-small-gpu-pt-5ctx-test1.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-5ctx-20260411-204813.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-5ctx-20260411-204858.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-5ctx-20260411-211450.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-5ctx-20260411-211518.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-5ctx-20260411-211551.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-5ctx-20260411-211629.json`
- `realtime_tests/resultados/realtime-small-gpu-pt-5ctx-20260411-211705.json`

Leitura:

- latencia media ficou mais alta em alguns runs;
- variacao textual ficou grande;
- houve runs bons e ruins, mas sem consistencia suficiente para ser baseline.

Conclusao:

- 5ctx nao trouxe vantagem clara sobre 0ctx no estado atual.

## 6. Comparacao sintetica dos contextos

Resumo prático do conjunto recente:

- `context_window = 0`: melhor estabilidade textual.
- `context_window = 3`: pode dar bom resultado, mas apresenta risco de deriva/repeticao.
- `context_window = 5`: maior variabilidade e sem ganho consistente.

Para o objetivo atual, o contexto 0 virou o melhor baseline temporario.

## 7. Problemas que ainda estamos tentando resolver

Prioridade alta:

- transcricao correta entre chunks, sem cortes e sem repeticao.

Prioridade menor:

- fazer o texto aparecer mais cedo sem piorar a qualidade.

Problemas ainda observados:

- cortes de palavra na troca de partes;
- repeticao em fronteira;
- algumas corrupcoes lexicais do modelo em audio curto.

## 8. Parâmetros finais escolhidos por enquanto

Baseline temporario atual:

- `context_window = 0`
- `step_duration_s = 0.2`
- `max_silence_inside_utterance_s = 0.4`
- `max_utterance_s = 3.2`
- `boundary_overlap_s = 0.45`
- `tail_guard_words = 4`

Por que esse baseline:

- contexto 0 foi o mais estavel nos testes recentes;
- step 0.2 / silencio 0.4 / max 3.2 deram o melhor equilibrio entre aparicao rapida e qualidade;
- overlap 0.45 e guard 4 ainda sao a melhor protecao de fronteira sem explodir latencia.

## 9. O que foi resolvido neste ciclo

- o script realtime ficou mais simples e focado em qualidade;
- os medidores de first part e part timing deixaram de ser usados;
- o contexto 0 passou a ser permitido de verdade;
- a analise agora esta mais voltada ao comportamento textual do que a velocidade isolada.

## 10. Conclusao executiva

No estado atual, a melhor direcao nao é adicionar mais complexidade de timing, e sim consolidar a qualidade textual na fronteira entre partes.

O baseline que faz mais sentido hoje é:

- contexto 0;
- step 0.2;
- silencio 0.4;
- max enunciado 3.2;
- overlap 0.45;
- guard 4.

Os proximos ajustes devem focar primeiro em acuracia entre chunks e, so depois, em antecipar a aparicao do texto, porque acelerar sem estabilizar a borda tende a piorar a transcricao.
