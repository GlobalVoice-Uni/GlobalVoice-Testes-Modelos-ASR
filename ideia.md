### Aplicativo de Tradução Simultânea Bidirecional para Reuniões
#### Tecnologia Principal: Python (Windows Desktop)

1. Problema 
    * Dificuldade de comunicação fluida em reuniões online entre falantes nativos de português e equipes estrangeiras (inglês), gerando barreiras técnicas e atrasos.

2. A Solução (Como funciona)
    * Um aplicativo desktop independente que roda "por cima" de qualquer software de videoconferência (Zoom, Teams, Google Meet, etc.). Ele atua como um intérprete em tempo real, realizando duas tarefas simultâneas:

    * Via de Escuta: Captura o áudio do sistema (o que os estrangeiros falam), transcreve, traduz para o português e exibe em formato de texto na tela do usuário em uma janela de chat, ao mesmo tempo, uma voz interpreta e lê a transcrição.

    * Via de Fala: Captura a voz do usuário no microfone (em português), traduz para o inglês, gera um áudio de voz sintética (TTS) e injeta esse áudio diretamente na reunião através de um microfone virtual com o menor delay possivel, para que a equipe estrangeira ouça a tradução limpa.

3. Arquitetura da Interface (UI)
    * A interface será dividida em duas janelas modulares conectadas (relação Pai/Filho), garantindo que botões importantes do software de reunião não sejam bloqueados:

    * Janela Principal (Chat): Uma janela clássica do Windows (com barra de título e botão de fechar). Aqui aparecerá o histórico da transcrição e tradução.

    * Aba de Opções (Ferramentas): Uma janela flutuante e sem bordas (frameless) conectada ao Chat. Conterá botões de ação rápida (ex: "Falar", "Configurações" e "Encerrar").

    * Ciclo de vida: Se o usuário fechar a janela principal no "X" ou clicar em "Encerrar" na aba de opções, o software inteiro (ambas as janelas e os processos de áudio) é finalizado de forma limpa.

4. Infraestrutura e Instalação
    * O software será agnóstico à plataforma de reunião porque operará na camada de áudio do Windows. Para garantir a injeção do áudio sintético na chamada, o projeto utilizará um driver de Virtual Audio Cable. Para manter a excelente experiência do usuário, a instalação desse driver será embutida no instalador principal do aplicativo, configurando o ambiente automaticamente.

5. Stack Tecnológica e Ferramentas (Ecossistema Python)
* Para desenvolver a aplicação no ambiente Windows e garantir performance em tempo real, o projeto utilizará as seguintes bibliotecas e ferramentas:

    * Interface Gráfica (UI):

        * PyQt6 ou PySide6: Frameworks robustos e padrão da indústria. São essenciais para este projeto pois lidam nativamente com a arquitetura de janelas "Pai e Filho" do Windows, além de permitirem a criação da janela de opções flutuante (frameless) sem conflitos com o sistema operacional.

    * Captura e Roteamento de Áudio:

        * Soundcard: Uma biblioteca Python moderna que interage perfeitamente com a API WASAPI do Windows. É ideal para fazer o "Loopback" (capturar o áudio interno que sai nos fones de ouvido) de forma muito mais simples que as alternativas mais antigas.

        * PyAudio: Como alternativa reserva para o gerenciamento da via de gravação do microfone.

    * Reconhecimento de Voz (Speech-to-Text - STT):

        * Faster-Whisper: Uma implementação altamente otimizada do modelo Whisper da OpenAI. Ele permite que a transcrição do inglês da reunião seja feita localmente no computador do usuário (via CPU ou GPU), oferecendo precisão excepcional sem depender de latência de rede ou custos com APIs externas.

    * Tradução (Motor de Inteligência Artificial):

        * DeepL API (via pacote deepl): Recomendado para a tradução rápida e com alto grau de naturalidade gramatical entre inglês e português.

        * (Alternativa) LLMs via API (OpenAI ou Google GenAI): Caso o projeto exija interpretação de contexto (como jargões técnicos específicos da área de estudo) antes da tradução literal.

    * Sintetização de Voz (Text-to-Speech - TTS):

        * Edge-TTS: Uma biblioteca fantástica que acessa as vozes neurais da Microsoft de forma gratuita. Possui latência baixíssima, não exige chave de API e gera vozes em inglês extremamente realistas para a IA se comunicar com a equipe estrangeira.

    * Empacotamento e Distribuição (Deploy):

        * PyInstaller: Responsável por compilar todo o código Python e suas dependências em um único arquivo .exe autossuficiente.

        * Inno Setup: Ferramenta de terceiros para criar o instalador final para o usuário ("Avançar, Avançar, Concluir"). Será configurado com um script para instalar o aplicativo e executar silenciosamente a instalação do driver de Virtual Audio Cable em segundo plano.