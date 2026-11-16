# Bot de combate do Old School RuneScape

Ataca um monstro, pega do chão os Big bones, as runas e as moedas que ele larga e enterra
os ossos. Só funciona no **cliente oficial** do jogo para Windows (a janela se chama
"Old School RuneScape"). O RuneLite não serve.

## O que ele faz (e o que não faz)

- Ataca o monstro escolhido que estiver mais perto do personagem, um por vez.
- Depois de cada morte, pega do chão os **Big bones**, as **runas**, as **moedas** (Coins) e a
  **Snape grass seed** que caíram perto. Os outros itens (inclusive as outras sementes) ficam
  no chão.
- Enterra os Big bones da mochila (inventário) durante as lutas e enquanto espera o loot
  (fora disso, quando junta 10).
- Com a vida abaixo de 40%, para de atacar e toca o som de erro do Windows. **Ele não come**:
  coma você ou saia da área. Ele volta sozinho quando a vida passa de 60%. Enquanto espera,
  mexe o mouse de leve de vez em quando para o jogo não desconectar.
- Se o jogo desconectar, clica em "Try again", "Play Now" ou "Click here to play". **Nunca
  digita a senha**: se o jogo pedir login, entre você e o bot continua.
- Não pega Bones comuns nem outros itens, não esvazia a mochila e não vai ao banco.

## Instalar (uma vez)

1. Instale o **Python 3.10 ou mais novo** em https://www.python.org/downloads/ e, na
   primeira tela do instalador, marque **"Add python.exe to PATH"**.
2. Dê dois cliques em **`Iniciar bot.bat`**. Na primeira vez ele instala sozinho os pacotes
   do bot (precisa de internet). Se preferir instalar à mão:
   `python -m pip install -r requirements.txt`

## Preparar o jogo

O bot lê a interface do jogo em posições fixas, medidas numa tela só:

- Monitor **1920x1200** com a escala do Windows em **125%** (Configurações > Sistema >
  Vídeo) e a barra de tarefas visível. Em outra resolução ou escala ele avisa e fica parado.
- Janela do jogo **maximizada**, layout **"Resizable - Classic"** e nada por cima dos orbes
  ao lado do minimapa (a vida é lida no orbe).
- Nas configurações do jogo, **NPC 'Attack' options** em **"Left-click where available"**.
  Com o mouse em cima do monstro, o canto de cima da tela tem que mostrar
  "Attack <Nome> (level-N)", e não "Walk here".
- **NPC Highlights** ligado, com o nome dos NPCs em ciano em cima deles. Não é obrigatório,
  mas com ele o bot reconhece e acha o monstro pelo nome, com muito mais precisão. Sem ele,
  procura pela cor do corpo e pode confundir coisas do cenário da mesma cor.
- O nome dos itens no chão tem que aparecer (o texto branco em cima do item, ex.:
  "Big bones (296 gp)" ou "Coins x 8"): é por ele que o bot acha o loot.
- Não mude o zoom da câmera depois de ensinar um monstro. Se mudar, ensine de novo
  (`--reaprender`, abaixo).
- Deixe a mochila aberta no painel (se outra aba abrir, o bot clica na mochila sozinho) e o
  personagem perto dos monstros.

## Ligar

Dê dois cliques em **`Iniciar bot.bat`** (ou, num terminal nesta pasta, `python main.py`).
A janela preta mostra as mensagens; uns 2 s depois o jogo vem para a frente.

- **Monstro ensinado pelo nome à vista** (com NPC Highlights ligado): o bot reconhece,
  toca um som e espera 6 s. Para caçar outro, ponha o mouse em cima dele e aperte **F9**
  nesse tempo; senão, ele começa com o que reconheceu.
- **Senão**, ele toca um som e pede: ponha o mouse em cima do monstro e aperte **F9**. Se já
  conhece esse monstro, começa. Se não, aprende na hora (veja "Ensinar um monstro novo").

Quando começa, a janela preta mostra "Bot iniciado — alvo: <monstro>".

Enquanto ele roda:

- Ele só age com o jogo na frente. Para usar o computador, clique em outra janela (o bot
  pausa) e depois volte ao jogo.
- Mexer no mouse também pausa o bot; ele volta 1 s depois que o mouse para.
- **F8** liga/desliga a coleta de ossos.

## Ensinar um monstro novo (F9)

Com o mouse em cima do monstro, aperte **F9**:

- **som normal** do Windows: aprendeu ("Aprendi pelo nome em ciano..." ou "Aprendi pela cor
  do corpo..."), e o bot começa;
- **som de erro**: não aprendeu. O motivo aparece na janela preta; ajeite e aperte F9 de novo.

O perfil do monstro fica em `perfis/`. Sem `--monstro`, um monstro novo vira
`perfis/monstro_1.json` (depois `monstro_2`, ...). Para o perfil ter o nome do monstro, ligue
com o nome (se ainda não o conhece, o bot pede o F9):

    python main.py --monstro "Nome do monstro"

Para ensinar de novo (ex.: depois de mudar o zoom):

    python main.py --monstro "Moss giant" --reaprender

Monstro de várias cores sem NPC Highlights, ou que não larga Big bones (ex.: Skeleton): use o
`ensinar.py`, que aceita vários exemplos. O `--sem-osso` é para o monstro que não larga Big
bones (o bot não espera nem procura ossos):

    python ensinar.py "Nome do monstro" --sem-osso

Aperte F9 em cima de um monstro de cada cor; F7 desfaz o último e F10 grava. Confira o
resultado em `perfis/<nome>_conferir.png` e depois ligue com
`python main.py --monstro "Nome do monstro"`.

Já vêm prontos: `sapo` (Giant frog) e `hillgiant` (Hill Giant).

## Parar

- **Ctrl+C** na janela preta (clique nela antes). Sempre funciona. Se aparecer
  "Deseja finalizar o arquivo em lotes (S/N)?", responda S.
- Levar o mouse a um canto da tela também para o bot, mas só enquanto ele está mexendo o
  mouse (não quando está pausado).

## Opções

Abra um terminal nesta pasta: no Explorador de Arquivos, clique na barra de endereço, digite
`cmd` e aperte Enter. As opções podem ser combinadas, ex.:
`python main.py --monstro "Moss giant" --sem-moedas`.

    python main.py -h                      lista as opções
    python main.py --monstro "Moss giant"  caça esse monstro; se for novo, aprende com F9
    python main.py --reaprender            ensina de novo o monstro (F9), por cima do que já sabia
    python main.py --sem-runas             não pega as runas que caem no chão
    python main.py --sem-moedas            não pega as moedas (Coins) do chão
    python main.py --sem-sementes          não pega as Snape grass seeds do chão
    python main.py --sem-vida              desliga a pausa por vida baixa e também a conferência
                                           da interface na tela e a volta automática ao jogo
                                           (só se o orbe de vida não aparecer)
    python main.py --amostras              salva quadros em amostras/ para o testar_visao.py

Nome com espaço vai **entre aspas**: `--monstro "Moss giant"`. Com `--monstro` de um monstro
já ensinado, o bot começa direto: clique no jogo para ele agir.

Atalho para um monstro: copie o `Iniciar bot.bat`, abra a cópia com o Bloco de Notas e troque
`python main.py %*` por `python main.py --monstro "Moss giant" %*`.

## Problemas comuns

- **"Janela do RuneScape não encontrada"**: abra o jogo (cliente oficial) antes de ligar o bot.
- **"área do jogo com ... maximize a janela do jogo"** ou **"janela do jogo minimizada"**:
  maximize o jogo. O bot continua sozinho.
- **"o jogo já está maximizado, mas a área dele tem ... px"**: o monitor, a escala do Windows
  ou a barra de tarefas não estão como na calibração (1920x1200, 125%, barra visível).
- **"Não consigo ler o orbe de vida"**: confira se o jogo está logado, a janela maximizada, o
  layout "Resizable - Classic" e se há algo por cima dos orbes. Só use `--sem-vida` se o orbe
  não aparecer de jeito nenhum (aí o bot não pausa com a vida baixa).
- **"Não vejo nomes de NPC em ciano"**: ligue o NPC Highlights no jogo.
- **"o texto do canto não começa com 'Attack'"**: ponha o mouse em cima do monstro e confira
  a opção NPC 'Attack' options ("Left-click where available").
- **"o jogo não está na frente"** (no F9): clique no jogo e aperte F9 de novo.
- **"No module named 'cv2'"** (ou outro nome): rode `python -m pip install -r requirements.txt`
  ou ligue pelo `Iniciar bot.bat`.
- **"unrecognized arguments"**: faltaram as aspas no nome (`--monstro "Moss giant"`) ou a
  opção foi digitada errada (veja `python main.py -h`).
- **"Perfil com problema"**: rode o comando que aparece na linha de baixo ("Ensine de novo: ...").
- **"O jogo pede login com senha"**: entre no jogo; o bot continua sozinho.
- **"O bot parou depois de 3 erros em 10 min"**: ligue de novo. Se repetir, guarde o arquivo
  de log que a mensagem mostra.
- **O bot não faz nada**: o jogo está na frente? Você está mexendo no mouse? Ele está pausado
  por vida baixa ("VIDA BAIXA" na janela preta)?

## Registros

Cada sessão grava `logs\sessao_<data>_<hora>.jsonl`. Para ver o resumo da última:
`python analisar_log.py`. A tela da pausa por vida baixa fica em `logs\vida_baixa_*.png`, e
as telas de fora do jogo, em `logs\telas\`.

## Outros arquivos

- `testar_visao.py`: confere a detecção com as amostras salvas (para quem mexe em `visao.py`).
  Com o bot ligado, use `python testar_visao.py -j 1`.
- `yew.py`: outro bot, que corta Yew em Edgeville. Os requisitos estão no começo do arquivo.
