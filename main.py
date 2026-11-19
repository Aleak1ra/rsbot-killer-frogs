"""Bot de combate para o cliente oficial do Old School RuneScape.

Passo a passo (instalar, preparar o jogo, ensinar um monstro, parar): README.md
Uso:  python main.py            (ou dois cliques em "Iniciar bot.bat")
      python main.py -h         (lista as opções: --monstro, --reaprender, --sem-runas, ...)

Cada sessão grava um log de eventos em logs/sessao_*.jsonl; use
`python analisar_log.py` para ver os tempos medidos.
"""
import argparse
import ctypes
import json
import os
import random
import re
import sys
import threading
import time

import cv2
import keyboard
import numpy as np
import pyautogui

import reconexao
import visao


def _args():
    p = argparse.ArgumentParser(
        allow_abbrev=False,  # '--sem-runa' dá erro em vez de virar --sem-runas calado
        description="Bot de combate do Old School RuneScape (cliente oficial, janela maximizada).",
        epilog="Sem --monstro: reconhece na tela um monstro já aprendido (pelo nome em ciano) ou pede "
               "o mouse em cima dele + F9 e aprende na hora (perfis/monstro_N.json). Com 'NPC "
               "Highlights' ligado no jogo (nome dos NPCs em ciano) o bot acha o monstro pelo nome, "
               "com muito mais precisão. F8 liga/desliga os ossos. Para parar: Ctrl+C no terminal (o "
               "mouse num canto da tela também para, mas só enquanto o bot mexe o mouse).")
    p.add_argument("--monstro", metavar="NOME",
                   help='o monstro a caçar, ex.: --monstro "Moss giant" (embutidos: sapo = Giant frog, '
                        "hillgiant); se for novo, aprende com F9")
    p.add_argument("--reaprender", action="store_true",
                   help="ensina de novo o monstro (F9), por cima do que já sabia")
    p.add_argument("--sem-vida", action="store_true",
                   help="desliga a pausa por vida baixa e também a conferência da interface na tela "
                        "e a volta automática ao jogo (só se o orbe de vida não aparecer)")
    p.add_argument("--sem-runas", action="store_true", help="não pega as runas que caem no chão")
    p.add_argument("--sem-moedas", action="store_true", help="não pega as moedas (Coins) do chão")
    p.add_argument("--sem-sementes", action="store_true", help="não pega as Snape grass seeds nem as Ranarr seeds do chão")
    p.add_argument("--sem-valiosos", action="store_true",
                   help="não pega os outros itens de rótulo azul (valiosos, como a Snapdragon seed)")
    p.add_argument("--amostras", action="store_true",
                   help="salva quadros em amostras/ para o testar_visao.py")
    a = p.parse_args()
    if a.monstro is not None and not re.search(r"[a-z0-9]", a.monstro.lower()):  # = nome_arquivo
        p.error(f"--monstro {a.monstro!r}: o nome precisa de pelo menos uma letra ou número")
    return a


ARGS = _args()  # antes de importar tela.py, que encerra sem o jogo aberto: o -h funciona sem ele

import tela  # noqa: E402
from tela import (  # noqa: E402
    CLIQUE_TIMEOUT,
    HOVER_ATRASO_MIN,
    HOVER_TIMEOUT,
    PONTO_NEUTRO,
    TEMPO_MOUSE_PARADO,
)

EMBUTIDOS = set(visao.PERFIS)  # sapo, hillgiant (carregar_perfil acrescenta os aprendidos em visao.PERFIS)


def nome_do_perfil(texto):
    """'Moss giant' -> 'moss_giant' (arquivo perfis/<nome>.json). Sem '_' também vale (o
    embutido hillgiant para 'hill_giant'; um perfil ensinado com o nome sem '_') e o nome do
    jogo de um embutido ('Giant frog' -> sapo)."""
    nome = visao.nome_arquivo(texto)  # nunca vazio: _args() já recusou o nome sem letra/número
    disponiveis = visao.perfis_disponiveis()
    if nome in disponiveis:
        return nome
    if nome.replace("_", "") in disponiveis:
        return nome.replace("_", "")
    for n in EMBUTIDOS:
        if visao.PERFIS[n]["nome"].lower() == texto.strip().lower():
            return n
    return nome


def recusar_f9(motivo):
    """Bipe de erro e o motivo de um F9 recusado (com a dica se o jogo não estiver na frente)."""
    tela.bip("erro")
    print(f"   ❌ {tela.com_dica_de_foco(motivo)}. Tente de novo (F9).")


def ler_f9(timeout=None):
    """Espera o F9 (até `timeout` s, se dado) e lê a tela e o mouse na hora do aperto:
    (img, pos), ou None (tempo esgotado, ou o jogo fora do lugar: motivo impresso)."""
    leitura = tela.ler_f9(timeout)
    if leitura is None:
        return None
    img, pos, problema = leitura
    if problema:
        recusar_f9(problema)
        return None
    return img, pos


def ler_f9_valido():
    """F9 até o mouse estar sobre um monstro: texto de ação "Attack ..." e ponto dentro do jogo.
    Devolve (img, pos); motivo da recusa impresso (e um bipe)."""
    import ensino

    while True:
        leitura = ler_f9()
        if leitura is None:
            continue
        img, pos = leitura
        if not visao.dentro_do_jogo(pos) or visao.na_interface(pos):
            motivo = "o mouse está fora da área do jogo"
        else:
            motivo = ensino.gerar_molde_texto(visao.recortar(img, visao.FAIXA_TEXTO_ACAO))[1]
        if motivo is None:
            return leitura
        recusar_f9(motivo)


def aprender(nome, nome_jogo, primeiro=None, osso="Big bones", espera_loot=4.5):
    """Ensina o monstro com o mouse em cima dele + F9 (ensino.Ensino: com o nome do NPC em
    ciano na tela, pelo nome; senão, pela cor), grava perfis/<nome>.json e devolve o perfil.
    `primeiro`: (img, pos) de um F9 já lido. Repete o F9 até um exemplo servir; Ctrl+C sai."""
    import ensino

    e = ensino.Ensino(nome, nome_jogo, espera_loot, osso=osso)
    leitura = primeiro
    while True:
        if leitura is None:
            leitura = ler_f9()
            if leitura is None:
                continue
        img, pos = leitura
        leitura = None
        try:
            msg = e.adicionar(img, pos, visao.recortar(img, visao.FAIXA_TEXTO_ACAO))
        except Exception as erro:  # um aperto ruim não pode derrubar o bot
            msg = f"exemplo recusado: erro ao aprender ({erro!r})"
        if e.perfil is not None and not msg.startswith("exemplo recusado") and not e.perfil.get("nome_npc"):
            # pela cor, 1 exemplo: tem que achar o próprio monstro no quadro do F9
            if not e.acha(img, pos):
                e.desfazer()
                msg = ("exemplo recusado: pela cor, não achei esse monstro (ponha o mouse no meio do "
                       "corpo, ou ligue 'NPC Highlights' no jogo)")
        if e.perfil is None or msg.startswith("exemplo recusado"):
            recusar_f9(msg.replace("exemplo recusado: ", ""))
            continue
        break
    caminho = e.salvar()
    e.conferir()
    tela.bip("ok")
    n = len(visao.detectar_sapos(img, e.perfil_visao()))
    if e.perfil.get("nome_npc"):
        print(f"   ✅ Aprendi pelo nome em ciano: {n} na tela agora. Perfil: {caminho}")
    else:
        print(f"   ✅ Aprendi pela cor do corpo ({n} candidato(s) na tela). Perfil: {caminho}")
        print("   ⚠️ Pela cor o bot pode passar o mouse em coisas do cenário da mesma cor; com "
              "'NPC Highlights' ligado no jogo ele acha o monstro pelo nome.")
    visao.PERFIS.pop(nome, None)  # perfis_conhecidos() pode ter guardado o perfil antigo
    return visao.carregar_perfil(nome)


def opcoes_antigas(nome):
    """(osso, espera_loot) de perfis/<nome>.json, para reaprender sem perder o que foi ajustado."""
    try:
        with open(os.path.join(visao.PASTA_PERFIS, f"{nome}.json"), encoding="utf-8") as f:
            d = json.load(f)
        return d.get("osso", "Big bones"), float(d.get("espera_loot", 4.5))
    except (OSError, ValueError):
        return "Big bones", 4.5


def perfis_conhecidos():
    """[(nome, perfil)] de todos os perfis que carregam sem erro (embutidos e aprendidos)."""
    lista = []
    for nome in visao.perfis_disponiveis():
        try:
            lista.append((nome, visao.carregar_perfil(nome)))
        except (KeyError, ValueError):
            pass
    return lista


def monstro_da_tela(img):
    """(nome, perfil) do monstro aprendido pelo nome em ciano com mais exemplares em `img`, ou
    None. Os perfis pela cor não entram: a cor sozinha não diz qual monstro é."""
    melhor = None
    for nome, p in perfis_conhecidos():
        if p.get("nome_npc") is None:
            continue
        n = len(visao.detectar_por_nome(img, p))
        if n and (melhor is None or n > melhor[0]):
            melhor = (n, nome, p)
    return None if melhor is None else melhor[1:]


def monstro_do_mouse(img):
    """(nome, perfil) cujo texto de ação ("Attack <Nome>") bate com o da tela agora, ou None."""
    faixa = visao.recortar(img, visao.FAIXA_TEXTO_ACAO)
    batem = [(nome, p) for nome, p in perfis_conhecidos()
             if visao.score_texto(faixa, p["texto"]) >= visao.limiar_texto(p["texto"])]
    if not batem:
        return None

    # mais de um perfil do mesmo monstro (ex.: um antigo pela cor): o pelo nome que acha o
    # monstro agora (sem "NPC Highlights" ele não acharia nada), depois o embutido, o mais novo
    def chave(item):
        nome, p = item
        arq = os.path.join(visao.PASTA_PERFIS, f"{nome}.json")
        pelo_nome = p.get("nome_npc") is not None and bool(visao.detectar_por_nome(img, p))
        return (pelo_nome, nome in EMBUTIDOS, os.path.getmtime(arq) if os.path.isfile(arq) else 0.0)
    return max(batem, key=chave)


def nome_livre():
    """monstro_1, monstro_2, ...: o primeiro sem perfil (o bot não lê o nome do monstro)."""
    n = 1
    while f"monstro_{n}" in visao.perfis_disponiveis():
        n += 1
    return f"monstro_{n}"


TROCAR_SEG = 6  # s para apertar F9 em outro monstro depois de reconhecer um na tela


def escolher_monstro(reaprender=False):
    """Sem --monstro: reconhece na tela um monstro já aprendido (pelo nome em ciano) e dá
    TROCAR_SEG s para um F9 em outro; se não reconhecer, pede o mouse em cima do monstro + F9 e
    usa o perfil que já bate com ele ou aprende um novo. `reaprender`: sempre pelo F9, e o
    monstro do mouse é ensinado de novo."""
    print("\n👉 Vou olhar a tela do jogo. Se eu pedir, ponha o mouse em cima do monstro que quer "
          "caçar e aperte F9 (vai ouvir um bipe). Ctrl+C sai.")
    print("   Dica: ligue 'NPC Highlights' no jogo (nome dos NPCs em ciano): o bot reconhece e acha "
          "o monstro pelo nome, com muito mais precisão.")
    time.sleep(2)
    tela.trazer_para_frente()
    time.sleep(0.5)
    leitura = None
    if not reaprender:
        achado = monstro_da_tela(tela.capturar())
        if achado:
            print(f"🎯 Reconheci {achado[1]['nome']} na tela (perfil {achado[0]}). Para caçar OUTRO, "
                  f"ponha o mouse nele e aperte F9 nos próximos {TROCAR_SEG} s.")
            tela.bip("pergunta")
            leitura = ler_f9(TROCAR_SEG)
            if leitura is None:
                return achado
    if leitura is None:
        tela.bip("pergunta")
        print("👉 Ponha o mouse em cima do monstro que quer caçar e aperte F9.")
        leitura = ler_f9_valido()
    conhecido = monstro_do_mouse(leitura[0])
    if conhecido and not (reaprender and conhecido[0] not in EMBUTIDOS):
        tela.bip("ok")
        print(f"🎯 Esse eu já conheço: {conhecido[1]['nome']} (perfil {conhecido[0]}).")
        return conhecido
    if conhecido:
        nome, nome_jogo = conhecido[0], conhecido[1]["nome"]
        osso, espera = opcoes_antigas(nome)
    else:
        nome = nome_livre()
        nome_jogo = nome.replace("_", " ").capitalize()
        osso, espera = "Big bones", 4.5
    print(f"🆕 {'Reaprendendo' if conhecido else 'Monstro novo'}: perfil {nome}.")
    return nome, aprender(nome, nome_jogo, primeiro=leitura, osso=osso, espera_loot=espera)


# Perfil do monstro: python main.py  (reconhece na tela ou pergunta com F9)
#                    python main.py --monstro "Moss giant"  (esse; se novo, aprende na hora)
REAPRENDER = ARGS.reaprender
ARG_MONSTRO = ARGS.monstro  # None: escolher_monstro
try:
    if ARG_MONSTRO is None:
        NOME_PERFIL, PERFIL = escolher_monstro(REAPRENDER)
    else:
        NOME_PERFIL = nome_do_perfil(ARG_MONSTRO)
        if NOME_PERFIL in EMBUTIDOS or (not REAPRENDER and NOME_PERFIL in visao.perfis_disponiveis()):
            PERFIL = visao.carregar_perfil(NOME_PERFIL)
        else:
            print(f"\n🆕 Ainda não conheço o monstro '{ARG_MONSTRO}'. Ponha o mouse em cima de um "
                  f"(o canto de cima mostra 'Attack ...') e aperte F9 (vai ouvir um bipe).")
            time.sleep(2)  # dá tempo de ler antes de o jogo vir para a frente
            tela.trazer_para_frente()
            tela.bip("pergunta")
            osso, espera = opcoes_antigas(NOME_PERFIL)
            PERFIL = aprender(NOME_PERFIL, ARG_MONSTRO if ARG_MONSTRO != NOME_PERFIL
                              else NOME_PERFIL.replace("_", " ").capitalize(), osso=osso, espera_loot=espera)
except KeyboardInterrupt:
    sys.exit("\nSaindo.")
except ValueError as erro:  # perfis/<nome>.json existe mas está incompleto ou estragado
    monstro = f'--monstro "{ARG_MONSTRO}" ' if ARG_MONSTRO else ""
    sys.exit(f"Perfil com problema: {erro}\nEnsine de novo: python main.py {monstro}--reaprender")

# ========== Configurações ==========
# Em tela.py (as mesmas do yew.py): TEMPO_MOUSE_PARADO, os tempos do texto de ação ao passar o
# mouse (HOVER_ATRASO_MIN, HOVER_TIMEOUT), o do X de clique (CLIQUE_TIMEOUT) e PONTO_NEUTRO.
DELAY_CICLO = 0.05
MAX_TENTATIVAS_SAPO = 4
# Mira: antes de clicar, relocaliza o sapo num recorte em volta do mouse
RAIO_MIRA = PERFIL["raio_mira"]
MIRA_DESLOC_MIN = 4  # px; abaixo disso não vale mexer o mouse
# px; acima disso o "mais perto" da mira é outro monstro (o recorte do perfil pelo nome é largo):
# não troca o alvo que o texto de ação confirmou
MIRA_DESLOC_MAX = 60
# Sapo com barra de vida sobre a cabeça já está lutando com outro jogador.
# Medido: 8 de 9 ataques que falharam tinham >= 36 px de barra; os 47 que deram certo, 0.
BARRA_SAPO_OCUPADO = 25
MAX_COLETAS_SEGUIDAS = 2  # coletas do chão entre dois ataques, havendo monstro livre à vista
# Sementes valiosas (Ranarr ~22K, Snape grass ~8K): pegas antes de tudo, sem o limite acima e até
# no meio de uma luta (procura a cada VALIOSO_INTERVALO s enquanto luta)
VALIOSOS = ("semente_ranarr", "semente", "valioso")
VALIOSO_INTERVALO = 0.5

# Menu de botão direito (pilhas de itens): meia largura/altura da área lida em volta do clique
MENU_RAIO = (280, 330)
MENU_TIMEOUT = 0.3  # medido em 479 menus certos: abrem em 0,066 s (p50), no máximo 0,174 s

# Segurança: abaixo disso o bot para de atacar e avisa (ele não come nem foge)
VIDA_MINIMA = 0.40
VIDA_RETOMAR = 0.60
VIDA_CONFIRMACOES = 3  # leituras seguidas antes de pausar/retomar (uma isolada não basta)
VIDA_TENTATIVAS_CONFIRMAR = 8  # quadros novos (~0,5 s) para confirmar uma leitura assim
ESPERA_TELA_ASSENTAR = 0.5  # s depois do jogo voltar ao foco/à janela certa
VIDA_AVISO_SEM_LEITURA = 60  # s jogando sem conseguir ler o orbe -> avisa no terminal
VIDA_AVISO_PAUSADO = 20  # idem, com o bot pausado esperando a vida voltar
SEM_VIDA = ARGS.sem_vida

# Combate
ESPERA_INICIO_COMBATE = 5.0  # medido em 120 ataques: 1,5 s típico, 4,25 s no pior caso
# Alvo perto começa a luta mais cedo: espera 3,0 s + dist/150 (medido em 4905 ataques: a até
# 150 px o pior caso foi 3,28 s; de 150 a 300 px, 4,94 s). Um ataque que não pega custa menos.
ESPERA_COMBATE_PERTO = 3.0
# Recorte do canto com o painel do oponente: a espera do combate lê só ele (~10x mais rápido que
# a janela inteira); começa em (0, 0), então as posições de visao.PAINEL_OPONENTE valem nele
PAINEL = (0, 0, visao.PAINEL_OPONENTE[2], visao.PAINEL_OPONENTE[3])
ESPERA_LOOT = PERFIL["espera_loot"]  # tempo para o osso aparecer depois da barra zerar
RAIO_LOOT = 320  # px em volta do personagem
LOOT_ROTULO_MIN = 1.0  # s: perfil sem osso só aceita o loot depois disso (medido: cai em 1,8-3,1 s)
# Perfil com osso, monstro que não larga Big bones: o loot inteiro cai de uma vez; rótulo novo
# sem Big bones por isso -> não vai ter osso, não espera o resto do ESPERA_LOOT
LOOT_SEM_OSSO_SEG = 0.4
# ...mas nunca antes disso da morte: um rótulo que não é do loot (item de outro jogador)
# encerraria cedo e o osso, que cai em ~2,95 s (p90 3,03 s, 946 mortes de Hill Giant), ficaria
LOOT_SEM_OSSO_MIN = 3.2
ABA_INTERVALO = 5.0  # s entre cliques na aba da mochila (garantir_mochila)

# Itens do chão pegos antes dos ossos: runas e moedas (--sem-runas / --sem-moedas desligam)
PEGAR_RUNAS = not ARGS.sem_runas
PEGAR_MOEDAS = not ARGS.sem_moedas
PEGAR_SEMENTES = not ARGS.sem_sementes  # só Snape grass seed e Ranarr seed (as outras ficam no chão)
PEGAR_VALIOSOS = not ARGS.sem_valiosos  # qualquer outro item de rótulo azul (o jogo marca os caros)
ESPERA_PEGAR_ITEM = 2.5  # + dist/150, como os ossos
# Distância do loot (runas, moedas e ossos): o das próprias lutas cai perto (ossos: 95% a
# <= 271 px; runas e moedas <= 122 px). O que está longe costuma ser de outros jogadores e
# não compensa a caminhada: com monstro na tela, mata o mais perto e pega o loot quando passar.
COLETA_RAIO_PERTO = 280  # px: com monstro na tela, só pega loot até aqui
COLETA_RAIO_MAX = 450  # px: sem monstro à vista, vai buscar até aqui
# Recorte do quadro onde procurar o loot (recorte_loot): além do raio, a largura dos
# rótulos para os lados, a pilha de rótulos acima e a pilha abaixo do rótulo
LOOT_MARGEM_LADO, LOOT_MARGEM_ACIMA, LOOT_MARGEM_ABAIXO = 380, 230, 60
COLETA_IGNORAR_SEG = 60.0  # pilha que falhou fica ignorada por esse tempo (as outras não)
COLETA_IGNORAR_VALIOSO_SEG = 3.0  # ...a não ser de semente valiosa: outra tentativa logo
COLETA_IGNORAR_RAIO = 25  # px em volta da pilha que falhou
COLETA_FALHAS_PAUSA = 3  # falhas seguidas -> deixa aquele item de lado por COLETA_PAUSA s
COLETA_PAUSA = 30.0
# Conferência da coleta pelo inventário (só a área dos itens, visao.INVENTARIO_ITENS): px
# que mudaram de tom (> INVENTARIO_DIF_TOM em algum canal). Medido em 429 pares de quadros:
# sem mudança, 0 px; um dígito trocando ("6" -> "8" numa pilha de runas), 17 px.
INVENTARIO_DIF_TOM = 40
INVENTARIO_DIF_PX = 6
INVENTARIO_ESPERA_MIN = 0.25  # s após o clique antes de aceitar a mudança (filtra as passageiras)
# Depois de pegar um item, o rótulo da pilha ainda aparece por um instante, entre onde ela estava
# na tela e o personagem (a câmera ainda alcançando). Visto em 15/11: 3 mira_perdida e 2 menus
# sem o item (contados como falha, um com 3,2 s de procura) logo após pegar runa/moedas. Por esse
# tempo (s), rótulos do mesmo item a até esse raio (px) do segmento pilha-personagem ficam de fora.
COLETA_RECEM_PEGO = (1.2, 40)
COLETA_SOB_PERSONAGEM = 90  # px do centro: depois de andar até a pilha, ela fica aqui
# Onde procurar o item (texto de ação "Take") quando o clique caiu fora dele: primeiro uma
# linha de rótulo abaixo, depois em volta
PROCURA_TAKE = [(0, 22), (12, 0), (-12, 0), (0, 10), (12, 22), (-12, 22), (24, 0), (-24, 0), (0, -10)]
# Mira com a tela parada (mirar_parado): região relida em volta do alvo (os rótulos ficam
# acima da pilha, até ~8 linhas de 22 px) e tolerância entre duas leituras seguidas
MIRA_RAIO_X, MIRA_ACIMA, MIRA_ABAIXO = 260, 230, 90
MIRA_PARADO_PX = 3
MIRA_LONGA = (1.5, 80)  # s, px: até isso depois de uma coleta mais longe que isso, mira com 2 leituras
# Visto com Skeleton (15/11): o painel do oponente sumiu ~1,3 s no meio da luta e voltou; o bot
# dava a luta por encerrada e saía passando o mouse em outros. Sem painel ("livre") só encerra
# a luta depois disso (o fim normal é "morto", que não espera).
COMBATE_LIVRE_SEG = 1.5
MAX_FOTOS_DIAG = 10  # por tipo e por sessão: logs/combate_livre/, logs/texto_quase/, *_osso_menu
TEXTO_QUASE = 0.15  # hover recusado com nota até isso abaixo do limiar: guarda a faixa
MIRA_TIMEOUT = 1.0
# s: quadro do ciclo mais velho que isso na hora de atacar (coleta que falhou pelo menu, ~3,9 s)
# é renovado. Visto em 14-15/11: 1º hover aceito em 96% com quadro de até 0,5 s; 33% com >= 1 s
QUADRO_VELHO_SEG = 0.5
COLETA_MAX_FOTOS = 40  # fotos de falha por sessão (logs/coleta_falhas/)

# Ossos
ESPERA_PEGAR_OSSO = 2.5  # medido em 48 coletas: 1,2 s típico, 1,6 s no pior caso
INTERVALO_ENTERRAR = 1.3  # o jogo ignora cliques de enterrar mais rápidos que ~2 ticks
MIN_OSSOS_ENTERRAR = 10  # fora de combate só enterra se acumular isso
# O jogo desconecta depois de 5 min sem mouse/teclado (visto às 03:32 de 15/11, numa pausa
# por vida baixa). Parado (pausa por vida), o bot mexe o mouse na coluna de pedra e toca o
# Shift a cada intervalo sorteado entre esses valores (s), sem clicar em nada.
ANTI_AFK_SEG = (60.0, 120.0)

SALVAR_AMOSTRAS = ARGS.amostras
MAX_AMOSTRAS_POR_CLASSE = 8
MAX_FAIXAS = 120

ossos_ativos = threading.Event()
ossos_ativos.set()
mouse_parado_desde = time.time()
pos_anterior = pyautogui.position()
ultima_entrada = time.time()  # último movimento/clique/tecla (do bot ou do usuário)
anti_afk_intervalo = random.uniform(*ANTI_AFK_SEG)

# Janela do jogo: a captura acompanha a área do jogo onde quer que a janela esteja
# (tela.atualizar_janela, chamado a cada ciclo). É o mesmo dicionário de tela.REGIAO.
CAPTURE_REGION = tela.REGIAO
CENTRO = (tela.LARGURA // 2, tela.ALTURA // 2)


# ========== Registro de eventos ==========
LOG = tela.Registro()  # logs/sessao_*.jsonl
_amostras = {}


def salvar_amostra(classe, img, pos, **extra):
    """Guarda um frame com o que o próprio jogo confirmou (texto de ação) naquela posição."""
    if not SALVAR_AMOSTRAS or _amostras.get(classe, 0) >= MAX_AMOSTRAS_POR_CLASSE:
        return
    pasta = os.path.join(visao.BASE_DIR, "amostras")
    os.makedirs(pasta, exist_ok=True)
    n = _amostras[classe] = _amostras.get(classe, 0) + 1
    nome = f"{time.strftime('%H%M%S')}_{classe}_{n}.png"
    cv2.imwrite(os.path.join(pasta, nome), img)
    with open(os.path.join(pasta, "rotulos.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"arquivo": nome, "classe": classe, "pos": list(pos), **extra}) + "\n")


def salvar_faixa(faixa, rotulo, scores):
    """Guarda a faixa do texto de ação vista num hover (para calibrar os templates)."""
    if not SALVAR_AMOSTRAS or faixa is None or _amostras.get("faixa", 0) >= MAX_FAIXAS:
        return
    pasta = os.path.join(visao.BASE_DIR, "amostras", "faixas")
    os.makedirs(pasta, exist_ok=True)
    n = _amostras["faixa"] = _amostras.get("faixa", 0) + 1
    partes = "_".join(f"{c}{int(100 * s)}" for c, s in scores.items())
    cv2.imwrite(os.path.join(pasta, f"{n:03d}_{rotulo}_{partes}.png"), faixa)


def foto_diagnostico(tipo, img, sufixo=""):
    """logs/<tipo>/HHMMSS_mmm<sufixo>.png, no máximo MAX_FOTOS_DIAG por tipo na sessão (sempre,
    sem --amostras)."""
    if img is None or _amostras.get(f"diag_{tipo}", 0) >= MAX_FOTOS_DIAG:
        return None
    _amostras[f"diag_{tipo}"] = _amostras.get(f"diag_{tipo}", 0) + 1
    pasta = os.path.join(visao.BASE_DIR, "logs", tipo)
    os.makedirs(pasta, exist_ok=True)
    nome = f"{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}{sufixo}.png"
    cv2.imwrite(os.path.join(pasta, nome), img)
    return nome


def salvar_clique(antes, depois, tipo):
    """Guarda a região do clique antes/depois (lado a lado) para testar_visao.py."""
    if not SALVAR_AMOSTRAS or _amostras.get("clique", 0) >= MAX_FAIXAS:
        return
    pasta = os.path.join(visao.BASE_DIR, "amostras", "cliques")
    os.makedirs(pasta, exist_ok=True)
    n = _amostras["clique"] = _amostras.get("clique", 0) + 1
    nome = f"{time.strftime('%H%M%S')}_{n:03d}_{tipo}.png"
    cv2.imwrite(os.path.join(pasta, nome), np.hstack([antes, depois]))


# ========== Tela e mouse ==========
capturar = tela.capturar  # janela inteira ou uma região (x1, y1, x2, y2) dela
jogo_em_foco = tela.jogo_em_foco  # segue a janela ativa do jogo (ver tela.py)
AVISO_INATIVO_SEG = 240  # fora de foco e sem entrada há isso: o jogo desconecta em ~1 min


def titulo_janela_ativa():
    """Título da janela em primeiro plano (para o log dizer quem tirou o foco do jogo)."""
    buf = ctypes.create_unicode_buffer(120)
    ctypes.windll.user32.GetWindowTextW(ctypes.windll.user32.GetForegroundWindow(), buf, 120)
    return buf.value


def tempo_mouse_parado():
    global mouse_parado_desde, pos_anterior, ultima_entrada
    atual = pyautogui.position()
    if np.linalg.norm(np.subtract(atual, pos_anterior)) > 2:
        mouse_parado_desde = ultima_entrada = time.time()
        pos_anterior = atual
    return time.time() - mouse_parado_desde >= TEMPO_MOUSE_PARADO


def mover(pos):
    """Move o mouse para uma posição da captura, sem contar como movimento do usuário.

    Nunca sai da área do jogo: um ponto fora dela (na moldura ou na barra de tarefas do
    Windows) é trazido para a borda de dentro e registrado no log.
    """
    global pos_anterior, ultima_entrada
    x, y = int(pos[0]), int(pos[1])
    if not visao.dentro_do_jogo((x, y)):
        cx, cy, cw, ch = visao.CLIENTE_CALIBRADO
        m = visao.DENTRO_MARGEM
        x2 = min(max(x, cx + m), cx + cw - m - 1)
        y2 = min(max(y, cy + m), cy + ch - m - 1)
        LOG.ev("mover_fora", pos=[x, y], ajustado=[x2, y2])
        x, y = x2, y2
    pyautogui.moveTo(CAPTURE_REGION["left"] + x, CAPTURE_REGION["top"] + y)
    pos_anterior = pyautogui.position()
    ultima_entrada = time.time()


def clicar(botao="left"):
    """Clica só se o jogo seguir em foco e o usuário não tiver pegado o mouse: o laço confere
    isso no começo do ciclo, mas o clique sai até ~2,8 s depois (menu). Um clique direito
    caído no terminal cola o que estiver copiado. Devolve se clicou."""
    global pos_anterior, ultima_entrada
    mexeu = bool(np.linalg.norm(np.subtract(pyautogui.position(), pos_anterior)) > 2)
    if mexeu or not jogo_em_foco():
        LOG.ev("clique_cancelado", botao=botao, mouse_mexeu=mexeu)
        return False
    pyautogui.click(button=botao)
    pos_anterior = pyautogui.position()
    ultima_entrada = time.time()
    return True


def clicar_em(pos):
    """Move até `pos` (coordenadas da captura) e clica (botões das telas de login)."""
    mover(pos)
    time.sleep(0.15)
    clicar()



_aba = {"t": 0.0}


def garantir_mochila(img):
    """Abre a aba da mochila se outra estiver aberta (as teclas F9/F7/F10 do ensino trocam a aba
    no jogo; sem a mochila o bot não vê os ossos nem confere as coletas). No máximo a cada
    ABA_INTERVALO s. True se clicou (o quadro do ciclo ficou velho)."""
    if visao.mochila_aberta(img) or time.time() - _aba["t"] < ABA_INTERVALO:
        return False
    _aba["t"] = time.time()
    x1, y1, x2, y2 = visao.ABA_MOCHILA
    LOG.ev("aba_mochila")
    clicar_em(((x1 + x2) // 2, (y1 + y2) // 2))
    mover(PONTO_NEUTRO)  # a dica do jogo sobre a aba cobriria o topo da mochila
    time.sleep(0.3)
    return True


SEM_NOMES_AVISO_SEG = 30
_sem_nomes = {"desde": None, "avisou": False}


def avisar_sem_nomes(img):
    """Perfil pelo nome e nenhum nome de NPC em ciano na tela há SEM_NOMES_AVISO_SEG: avisa uma
    vez ("NPC Highlights" desligado; o bot segue pela cor de reserva, se o perfil tiver)."""
    if visao.ha_nomes_npc(img):
        _sem_nomes["desde"] = None
        return
    agora = time.time()
    if _sem_nomes["desde"] is None:
        _sem_nomes["desde"] = agora
    elif not _sem_nomes["avisou"] and agora - _sem_nomes["desde"] > SEM_NOMES_AVISO_SEG:
        _sem_nomes["avisou"] = True
        LOG.ev("sem_nomes_npc", seg=SEM_NOMES_AVISO_SEG)
        print(f"\n⚠️ Não vejo nomes de NPC em ciano há {SEM_NOMES_AVISO_SEG} s: o 'NPC Highlights' "
              "está ligado no jogo? Sem ele o bot procura pela cor do corpo, bem menos precisa.")


RECONEXAO = reconexao.Reconexao(clicar_em, LOG.ev)
VIGIA_VIDA = visao.VigiaVida(VIDA_MINIMA, VIDA_RETOMAR, VIDA_CONFIRMACOES)

def manter_conectado():
    """Parado (pausa por vida baixa), evita a desconexão por inatividade: depois de um
    intervalo sem nenhuma entrada, mexe o mouse na coluna de pedra do inventário e toca o
    Shift. Não clica em nada. Só deve ser chamada com o jogo em foco e na tela de jogo."""
    global anti_afk_intervalo, ultima_entrada
    if time.time() - ultima_entrada < anti_afk_intervalo:
        return False
    x, y = PONTO_NEUTRO
    mover((x + random.randint(-3, 3), y + random.randint(-20, 20)))
    time.sleep(random.uniform(0.08, 0.2))
    mover(PONTO_NEUTRO)
    pyautogui.press("shift")
    ultima_entrada = time.time()
    anti_afk_intervalo = random.uniform(*ANTI_AFK_SEG)
    LOG.ev("anti_afk")
    return True


def clicar_conferindo(pos, classe):
    """Clica e lê o X que o jogo desenha: 'alvo' (vermelho), 'chao' (amarelo) ou None.

    Um X amarelo significa que o alvo saiu de baixo do mouse e o clique virou
    "Walk here": dá para desistir na hora em vez de esperar o timeout.
    """
    regiao = tela.caixa(pos, visao.CLIQUE_RAIO)
    antes = capturar(regiao)
    if not clicar():
        return "cancelado"
    inicio = time.perf_counter()
    tipo, depois = None, antes
    while tipo is None and time.perf_counter() - inicio < CLIQUE_TIMEOUT:
        time.sleep(0.01)
        depois = capturar(regiao)
        tipo = visao.tipo_clique(antes, depois)
    lat = time.perf_counter() - inicio
    LOG.ev("clique", classe=classe, tipo=tipo, lat=round(lat, 3))
    salvar_clique(antes, depois, tipo)
    return tipo


def refinar_mira(pos):
    """Posição atual do sapo que estava em `pos` e quanto ele andou (px)."""
    reg = tela.caixa(pos, RAIO_MIRA)
    x1, y1 = reg[:2]
    achado, _ = visao.sapo_mais_proximo(capturar(reg), (pos[0] - x1, pos[1] - y1), PERFIL)
    if achado is None:
        return pos, None
    novo = (x1 + achado[0], y1 + achado[1])
    return novo, distancia(pos, novo)


distancia = tela.distancia


def dist_segmento(p, a, b):
    """Distância (px) do ponto p ao segmento a-b."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return distancia(p, (ax + t * dx, ay + t * dy))


# ========== Leituras com espera ==========
# esperar_texto devolve (chave ou None, latência, {chave: maior score}, última faixa);
# esperar devolve (ok, tempo decorrido). Ver tela.py.
esperar_texto = tela.esperar_texto
esperar = tela.esperar


def contar_ossos_inventario():
    return visao.detectar_ossos_inventario(capturar(visao.INVENTARIO))


# ========== Ações ==========
class Enterro:
    """Enterra um osso por vez, respeitando o intervalo do jogo, e confere depois.

    Enterrar não interrompe a luta, então o bot usa os tempos mortos (combate e
    espera do loot). A conferência é feita num ciclo posterior, com o mouse fora
    do inventário, porque o tooltip do jogo esconde ícones vizinhos.
    """

    def __init__(self):
        self.ultimo = 0.0
        self.pendente = None  # (instante do clique, ossos antes, motivo)
        self.pegos_desde = 0
        self.total = 0

    def pode(self):
        return time.time() - self.ultimo >= INTERVALO_ENTERRAR

    def conferir(self, n_inventario):
        if self.pendente is None or time.time() - self.pendente[0] < INTERVALO_ENTERRAR:
            return
        _, antes, motivo = self.pendente
        ok = n_inventario < antes + self.pegos_desde
        LOG.ev("enterrar", ok=ok, motivo=motivo, antes=antes, depois=n_inventario)
        if ok:
            self.total += 1
            print(f"🦴 Total de ossos enterrados: {self.total}")
        self.pendente = None

    def um(self, inventario, motivo):
        mover(inventario[0])
        time.sleep(0.05)
        clicar()
        time.sleep(0.03)
        mover(PONTO_NEUTRO)
        self.ultimo = time.time()
        self.pendente = (self.ultimo, len(inventario), motivo)
        self.pegos_desde = 0


ENTERRO = Enterro()


def salvar_menu(rec, ok, item="osso"):
    """Grava HHMMSS_NNN_ok_<item>.png (ou _sem_): testar_menus e testar_item_menu usam o item do
    nome (os 45 menus antigos, sem sufixo, são de osso)."""
    if not SALVAR_AMOSTRAS or rec is None or _amostras.get("menu", 0) >= MAX_FAIXAS:
        return
    pasta = os.path.join(visao.BASE_DIR, "amostras", "menus")
    os.makedirs(pasta, exist_ok=True)
    n = _amostras["menu"] = _amostras.get("menu", 0) + 1
    sufixo = f"_{item}"
    nome = f"{time.strftime('%H%M%S')}_{n:03d}_{'ok' if ok else 'sem'}{sufixo}.png"
    cv2.imwrite(os.path.join(pasta, nome), rec)


_ultimo_menu = {"rec": None, "regiao": None}  # o último menu lido (para a foto de falha)
# a tela não parou a tempo (mirar_parado): o quadro do ciclo ficou velho (~1 s) para o ataque
_mira = {"perdida": False}
_ultima_coleta = {"t": 0.0, "dist": 0.0}  # a última coleta confirmada (mirar_parado)


def pegar_pelo_menu(pos, achar=visao.achar_take_no_menu, item="osso"):
    """Pilha de itens: abre o menu e clica na linha que `achar` encontrar
    (por padrão "Take Big bones"; para runas, "Take ... rune")."""
    if not clicar("right"):
        _ultimo_menu["rec"] = None
        return None  # cancelado (sem foco ou o usuário no mouse): ninguém tenta de novo agora
    x, y = int(pos[0]), int(pos[1])
    regiao = (max(x - MENU_RAIO[0], 0), max(y - MENU_RAIO[1], 0),
              min(x + MENU_RAIO[0], CAPTURE_REGION["width"]),
              min(y + MENU_RAIO[1], CAPTURE_REGION["height"]))
    inicio = time.perf_counter()
    ponto, score, rec = None, 0.0, None
    while ponto is None and time.perf_counter() - inicio < MENU_TIMEOUT:
        time.sleep(0.03)
        rec = capturar(regiao)
        ponto, score = achar(rec)
    LOG.ev("menu", item=item, ok=ponto is not None, score=round(score, 3),
           lat=round(time.perf_counter() - inicio, 3))
    _ultimo_menu["rec"], _ultimo_menu["regiao"] = rec, regiao
    salvar_menu(rec, ponto is not None, item)
    if ponto is None:  # fecha o menu tirando o mouse de cima dele
        mover(PONTO_NEUTRO)
        return False
    mover((regiao[0] + ponto[0], regiao[1] + ponto[1]))
    time.sleep(0.03)
    return clicar()


def procurar_take(pos):
    """Passa o mouse em volta de `pos` até o texto de ação mostrar "Take" (há um item sob o
    cursor). Devolve o ponto achado ou None. O item só é clicável em cima do desenho dele,
    que nem sempre fica no centro do rótulo (visto em jogo: 18 px ao lado numa pilha com
    "Big bones x 2"), e a descida pela pilha às vezes para uma linha antes (22 px acima)."""
    for dx, dy in PROCURA_TAKE:
        ponto = (pos[0] + dx, pos[1] + dy)
        if not visao.dentro_do_jogo(ponto):
            continue
        mover(ponto)
        time.sleep(HOVER_ATRASO_MIN)
        seguidas = [0]  # duas leituras seguidas: o texto de antes pode ainda estar na tela

        def tem_take():
            seguidas[0] = seguidas[0] + 1 if visao.tem_take_acao(capturar(visao.FAIXA_TEXTO_ACAO)) else 0
            return seguidas[0] >= 2

        ok, _ = esperar(tem_take, HOVER_TIMEOUT, 0.03)
        if ok:
            return ponto
    return None


def pegar_pelo_menu_pilha(pos, achar=visao.achar_take_no_menu, item="osso"):
    """pegar_pelo_menu na pilha; se o menu abriu sem nenhum "Take" (o clique caiu fora do
    item), procura com o mouse um ponto em volta onde o jogo mostre "Take" e tenta ali."""
    r = pegar_pelo_menu(pos, achar, item)
    if r is None:
        return None  # clique cancelado (sem foco ou o usuário no mouse): não insiste
    if r:
        return True
    if visao.menu_tem_take(_ultimo_menu["rec"]):
        return False  # a pilha estava certa, só não tinha o item
    ponto = procurar_take(pos)
    LOG.ev("menu_procura", item=item, achou=ponto is not None,
           desloc=None if ponto is None else [int(ponto[0] - pos[0]), int(ponto[1] - pos[1])])
    if ponto is None:
        return False
    return pegar_pelo_menu(ponto, achar, item)


def mirar_parado(pos, detectar_local):
    """Onde o alvo está AGORA, com a tela parada; None se sumiu ou a tela não parou.

    Visto em jogo: logo depois de pegar um osso o personagem ainda anda, a câmera o segue
    e a tela desliza; o clique feito pelo quadro do início do ciclo caía ~65 px fora da
    pilha. Relê a região do alvo até duas leituras seguidas concordarem (±3 px). A 1ª leitura
    é comparada com `pos` (do quadro do ciclo, ~0,2 s antes): com a tela parada já confere e
    poupa uma leitura (~55 ms por coleta; 80% das coletas); se não conferir, segue como antes.
    `detectar_local(recorte, origem)` devolve as posições candidatas (coordenadas da janela).
    """
    x, y = int(pos[0]), int(pos[1])
    regiao = (max(x - MIRA_RAIO_X, 0), max(y - MIRA_ACIMA, 0),
              min(x + MIRA_RAIO_X, CAPTURE_REGION["width"]), min(y + MIRA_ABAIXO, CAPTURE_REGION["height"]))
    # logo depois de uma coleta longa a câmera ainda desliza devagar e a comparação com o quadro
    # do ciclo aceitava cedo demais (10,7% de menus falhos, contra 2,6% exigindo 2 leituras)
    longa = time.time() - _ultima_coleta["t"] < MIRA_LONGA[0] and _ultima_coleta["dist"] > MIRA_LONGA[1]
    atual, anterior = pos, (None if longa else pos)
    inicio = time.perf_counter()
    while time.perf_counter() - inicio < MIRA_TIMEOUT:
        candidatos = detectar_local(capturar(regiao), regiao[:2])
        if not candidatos:
            return None
        atual = min(candidatos, key=lambda p: distancia(p, atual))
        if anterior is not None and distancia(atual, anterior) <= MIRA_PARADO_PX:
            return atual
        anterior = atual
        time.sleep(0.04)
    return None


def pegar_osso(img, ossos):
    for osso in sorted(ossos, key=lambda o: distancia(CENTRO, o["pos"]))[:2]:
        pos = osso["pos"]
        if PERFIL["loot_menu"]:
            pos = mirar_parado(pos, lambda rec, o: [b["pos"] for b in visao.detectar_ossos_chao(rec, origem=o)])
            if pos is None:
                LOG.ev("mira_perdida", item="osso")
                _mira["perdida"] = True
                return False
        dist = distancia(CENTRO, pos)
        mover(pos)
        if PERFIL["loot_menu"]:
            chave = None
            time.sleep(0.05)
        else:
            chave, lat, scores, faixa = esperar_texto(["take"])
            salvar_faixa(faixa, "take-ok" if chave else "take-rej", scores)
            LOG.ev("hover_osso", ok=chave is not None, lat=round(lat, 3), dist=round(dist),
                   score_rotulo=osso["score"], score_texto=round(scores["take"], 3))
        antes = len(contar_ossos_inventario())
        if chave is not None:
            via = "direto"
            clique = clicar_conferindo(pos, "osso")
            if clique in ("chao", "cancelado"):
                LOG.ev("pegar_osso", ok=False, lat=0.0, dist=round(dist), clique=clique, via=via)
                return False
        else:
            # Outro item por cima do osso (ou um monstro em cima da pilha)
            via, clique = "menu", None
            r = pegar_pelo_menu_pilha(pos)
            if r is None:
                return False  # clique cancelado: não passa para o próximo osso
            if not r:
                # diagnóstico: o menu de osso que falhou (visto em 15/11: 8 de 544 recusados, ~0,901)
                foto_diagnostico("coleta_falhas", _ultimo_menu["rec"], "_osso_menu")
                salvar_amostra("osso_rej", img, pos, score=osso["score"])
                continue
        salvar_amostra("osso_ok", img, pos, score=osso["score"])
        ok, dt = esperar(lambda: len(contar_ossos_inventario()) > antes,
                         ESPERA_PEGAR_OSSO + dist / 150)
        LOG.ev("pegar_osso", ok=ok, lat=round(dt, 3), dist=round(dist), clique=clique, via=via,
               pos=[int(pos[0]), int(pos[1])])
        if ok:
            ENTERRO.pegos_desde += 1
            _ultima_coleta.update(t=time.time(), dist=dist)
        return True
    return False


class ColetaChao:
    """Pega um tipo de item do chão (runas, moedas) pelo menu da pilha e confere pelo
    rótulo sumindo da tela.

    Rótulo e menu são conferidos separadamente (visao.detectar_itens_chao e
    visao.achar_take_item_no_menu). Uma pilha que falhou fica ignorada por um tempo (as
    outras continuam sendo pegas); muitas falhas seguidas deixam o item de lado por um
    tempo, para não travar o bot. Cada falha grava uma foto em logs/coleta_falhas/.
    """

    PEGAS = "pegas"  # "Runas pegas: N"

    def __init__(self, item, plural, emoji, ligado):
        self.item, self.plural, self.emoji, self.ligado = item, plural, emoji, ligado
        self.total = 0
        self.falhas = 0
        self.pausa_ate = 0.0
        self.ignoradas = []  # [(pos, até quando)] pilhas que falharam
        self.recem_ate = 0.0  # até quando ignorar o rótulo "fantasma" da pilha recém-pega
        self.recem_pos = CENTRO  # onde ela estava na tela quando foi pega

    def ativas(self):
        return self.ligado and time.time() >= self.pausa_ate

    def _detectar(self, img, origem=(0, 0), ossos=None, mascara=None, azul=None):
        """Os rótulos do item na imagem (sem filtro de distância)."""
        return visao.detectar_itens_chao(img, self.item, origem=origem, ossos=ossos, mascara=mascara)

    def _achar_menu(self, alvo):
        """Função que acha a linha "Take" do item no recorte do menu aberto."""
        return lambda rec: visao.achar_take_item_no_menu(rec, self.item)

    def detectar(self, img, ossos=None, mascara=None, raio=COLETA_RAIO_MAX, origem=(0, 0), azul=None):
        """Itens a até `raio` px do personagem, sem as pilhas que falharam há pouco.

        `img` pode ser um recorte da janela com canto em `origem` (ver recorte_loot)."""
        agora = time.time()
        self.ignoradas = [(p, t) for p, t in self.ignoradas if t > agora]
        itens = self._detectar(img, origem=origem, ossos=ossos, mascara=mascara, azul=azul)
        if agora < self.recem_ate:  # o rótulo da pilha recém-pega ainda aparece um instante
            itens = [i for i in itens
                     if dist_segmento(i["pos"], self.recem_pos, CENTRO) > COLETA_RECEM_PEGO[1]]
        return [i for i in itens if distancia(CENTRO, i["pos"]) <= raio
                and not any(distancia(i["pos"], p) <= COLETA_IGNORAR_RAIO for p, _ in self.ignoradas)]

    def _foto_falha(self, img, alvo, motivo, depois=None):
        """Quadro com o alvo marcado, o menu lido e o quadro do fim da espera (se houver),
        para descobrir depois por que falhou."""
        global _fotos_falha
        if _fotos_falha >= COLETA_MAX_FOTOS:
            return
        _fotos_falha += 1
        pasta = os.path.join(visao.BASE_DIR, "logs", "coleta_falhas")
        os.makedirs(pasta, exist_ok=True)
        foto = img.copy()
        x, y = alvo["pos"]
        cv2.circle(foto, (int(x), int(y)), 9, (0, 0, 255), 2)
        rx, ry = alvo["rotulo"]
        cv2.rectangle(foto, (rx, ry), (rx + 40, ry + 20), (0, 255, 0), 1)
        nome = time.strftime("%H%M%S") + f"_{self.item}_{motivo}"
        cv2.imwrite(os.path.join(pasta, nome + ".png"), foto)
        if _ultimo_menu["rec"] is not None and motivo == "menu":
            cv2.imwrite(os.path.join(pasta, nome + "_menu.png"), _ultimo_menu["rec"])
        if depois is not None:
            cv2.imwrite(os.path.join(pasta, nome + "_depois.png"), depois)

    def _falhou(self, img, alvo, motivo, depois=None, ignorar="alvo"):
        """`ignorar`: ponto da tela onde a pilha que falhou está agora ("alvo" = onde estava
        no início; None = não se sabe, não ignora nada)."""
        self._foto_falha(img, alvo, motivo, depois)
        if ignorar == "alvo":
            ignorar = alvo["pos"]
        valioso = self.item in VALIOSOS  # semente: tenta de novo logo, antes que outro jogador pegue
        if ignorar is not None:
            seg = COLETA_IGNORAR_VALIOSO_SEG if valioso else COLETA_IGNORAR_SEG
            self.ignoradas.append((ignorar, time.time() + seg))
        if valioso:
            return  # e nunca deixa as sementes de lado por falhas seguidas
        self.falhas += 1
        if self.falhas >= COLETA_FALHAS_PAUSA:
            self.falhas = 0
            self.pausa_ate = time.time() + COLETA_PAUSA
            LOG.ev(f"{self.plural}_pausa", seg=COLETA_PAUSA)
            print(f"{self.emoji} Não consegui pegar {self.plural} {COLETA_FALHAS_PAUSA}x seguidas: "
                  f"deixando de lado por {COLETA_PAUSA:.0f} s.")

    def pegar(self, img, itens):
        alvo = min(itens, key=lambda r: distancia(CENTRO, r["pos"]))
        # Os valiosos são procurados também no meio de uma luta, inclusive logo depois do ataque a um
        # monstro longe, com o personagem andando e a tela deslizando (visto em 17/11: a mira
        # conferiu num instante parado da câmera, o menu abriu ~90 px fora da pilha e a Snape grass
        # seed ficou para trás). Só tenta com a tela parada; senão, o próximo ciclo tenta de novo.
        if self.item in VALIOSOS:
            parada = esperar_tela_parar()
            if not parada["ok"]:
                LOG.ev("valioso_andando", item=self.item, seg=parada["seg"])
                return False
        pos = mirar_parado(alvo["pos"], lambda rec, o: [i["pos"] for i in self._detectar(rec, origem=o)])
        if pos is None:
            LOG.ev("mira_perdida", item=self.item)
            _mira["perdida"] = True
            return False
        if distancia(pos, alvo["pos"]) > MIRA_PARADO_PX:
            alvo = dict(alvo, pos=pos)
        dist = distancia(CENTRO, pos)
        evento = f"pegar_{self.item}"
        # Inventário de referência: um osso enterrado há pouco ainda pode sumir dele
        while time.time() - ENTERRO.ultimo < INTERVALO_ENTERRAR:
            time.sleep(0.05)
        inv_antes = capturar(visao.INVENTARIO_ITENS)
        mover(pos)
        time.sleep(0.05)
        r = pegar_pelo_menu_pilha(pos, self._achar_menu(alvo), self.item)
        if r is None:
            return False  # clique cancelado: não é falha da pilha
        if not r:
            LOG.ev(evento, ok=False, motivo="menu", dist=round(dist), score=alvo["score"], pos=list(pos))
            salvar_amostra(f"{self.item}_rej", img, pos, score=alvo["score"])
            self._falhou(img, alvo, "menu")
            return False
        salvar_amostra(f"{self.item}_ok", img, pos, score=alvo["score"])

        # Pegou quando o inventário muda: runa e moedas empilham, então aparece a pilha nova
        # ou muda o número dela. Contar rótulos na tela dava falha falsa em caminhadas
        # longas (a tela desliza e outra pilha do item entra no caminho). Duas leituras
        # seguidas, para um redesenho isolado não contar.
        seguidas = [0]
        leitura = {"px": 0}
        clique = time.perf_counter()

        def mudou():
            dif = cv2.absdiff(capturar(visao.INVENTARIO_ITENS), inv_antes).max(axis=2) > INVENTARIO_DIF_TOM
            leitura["px"] = int(dif.sum())
            if time.perf_counter() - clique < INVENTARIO_ESPERA_MIN:
                return False  # mudança antes disso não pode ser a coleta (passageira)
            seguidas[0] = seguidas[0] + 1 if leitura["px"] >= INVENTARIO_DIF_PX else 0
            return seguidas[0] >= 2

        ok, dt = esperar(mudou, ESPERA_PEGAR_ITEM + dist / 150, 0.1)
        LOG.ev(evento, ok=ok, lat=round(dt, 3), dist=round(dist), score=alvo["score"], pos=list(pos),
               inv_px=leitura["px"])
        if ok:
            self.falhas = 0
            self.total += 1
            self.recem_ate = time.time() + COLETA_RECEM_PEGO[0]
            self.recem_pos = pos
            _ultima_coleta.update(t=time.time(), dist=dist)
            print(f"{self.emoji} {self.plural.capitalize()} {self.PEGAS}: {self.total}")
            if self.item in VALIOSOS:  # raros: guarda o quadro para conferir o que foi pego
                foto_diagnostico("valioso_ok" if self.item == "valioso" else "semente_ok", img)
        else:
            # o personagem andou até a pilha: ela (se ficou) está agora perto dele
            depois = capturar()
            perto = [i["pos"] for i in self._detectar(depois)
                     if distancia(i["pos"], CENTRO) <= COLETA_SOB_PERSONAGEM]
            onde = min(perto, key=lambda p: distancia(p, CENTRO)) if perto else None
            self._falhou(img, alvo, "inventario_igual", depois=depois, ignorar=onde)
        return True


class ColetaValiosa(ColetaChao):
    """Qualquer item de rótulo azul (o jogo marca assim os que passam do limite de valor), sem
    molde do nome: a linha do menu sai do desenho do nome lido no próprio rótulo
    (visao.detectar_valiosos_chao e visao.achar_take_valioso_no_menu)."""

    PEGAS = "pegos"

    def _detectar(self, img, origem=(0, 0), ossos=None, mascara=None, azul=None):
        return visao.detectar_valiosos_chao(img, origem=origem, azul=azul, mascara=mascara)

    def _achar_menu(self, alvo):
        return lambda rec: visao.achar_take_valioso_no_menu(rec, alvo["nome"])


_fotos_falha = 0
# Ordem de prioridade: Ranarr seed e Snape grass seed, os outros itens de rótulo azul (raros e
# caros), runas, moedas (os ossos vêm depois, no laço)
RUNAS = ColetaChao("runa", "runas", "🔮", PEGAR_RUNAS)
MOEDAS = ColetaChao("moedas", "moedas", "🪙", PEGAR_MOEDAS)
SEMENTES = ColetaChao("semente", "Snape grass seeds", "🌱", PEGAR_SEMENTES)
RANARR = ColetaChao("semente_ranarr", "Ranarr seeds", "🌿", PEGAR_SEMENTES)
VALIOSO = ColetaValiosa("valioso", "itens valiosos (rótulo azul)", "💎", PEGAR_VALIOSOS)
COLETAS = (RANARR, SEMENTES, VALIOSO, RUNAS, MOEDAS)


def atacar_sapo(img, sapos, t_quadro=None):
    """Passa o mouse nos candidatos (mais perto primeiro) e ataca o 1º que o jogo confirmar.
    `t_quadro`: perf_counter da captura de `img` (só para o log: idade do quadro no hover)."""
    candidatos = sorted(sapos, key=lambda s: distancia(CENTRO, s["pos"]))
    tentativas = 0
    for sapo in candidatos:
        if tentativas >= MAX_TENTATIVAS_SAPO:
            break
        pos = sapo["pos"]
        dist = distancia(CENTRO, pos)
        barra = visao.barra_vida_perto(img, pos, *PERFIL["barra"])
        if barra >= BARRA_SAPO_OCUPADO:
            LOG.ev("sapo_ocupado", dist=round(dist), barra=barra, pos=[int(pos[0]), int(pos[1])])
            continue
        tentativas += 1
        idade = None if t_quadro is None else round(time.perf_counter() - t_quadro, 3)
        mover(pos)
        chave, lat, scores, faixa = esperar_texto([PERFIL["texto"]])
        salvar_faixa(faixa, f"sapo-{chave}", scores)
        LOG.ev("hover_sapo", texto=chave, lat=round(lat, 3), dist=round(dist), area=sapo["area"],
               caixa=sapo["caixa"], barra=barra, score=round(scores[PERFIL["texto"]], 3),
               pos=[int(pos[0]), int(pos[1])], idade=idade)
        if chave != PERFIL["texto"]:
            salvar_amostra(f"{NOME_PERFIL}_rej", img, pos, area=sapo["area"], texto=chave)
            limiar = visao.limiar_texto(PERFIL["texto"])
            if limiar - TEXTO_QUASE <= scores[PERFIL["texto"]] < limiar:  # quase: por que não passou?
                foto_diagnostico("texto_quase", faixa)
            continue
        salvar_amostra(f"{NOME_PERFIL}_ok", img, pos, area=sapo["area"])
        novo, desloc = refinar_mira(pos)
        if desloc is not None and MIRA_DESLOC_MIN <= desloc <= MIRA_DESLOC_MAX:
            mover(novo)
            pos = novo
        desloc = None if desloc is None else round(desloc, 1)
        clique = clicar_conferindo(pos, "sapo")
        if clique in ("chao", "cancelado"):  # o sapo pulou (ou clique cancelado): próximo ciclo
            LOG.ev("atacar", ok=False, lat=0.0, dist=round(dist), barra=barra, clique=clique,
                   desloc=desloc)
            return False
        ok, dt = esperar(lambda: visao.estado_combate(capturar(PAINEL)) == "lutando",
                         min(ESPERA_INICIO_COMBATE, ESPERA_COMBATE_PERTO + dist / 150), 0.1)
        LOG.ev("atacar", ok=ok, lat=round(dt, 3), dist=round(dist), barra=barra, clique=clique,
               desloc=desloc)
        if not ok:  # por que a luta não começou? (o chat costuma dizer: "Someone else is fighting that")
            foto_diagnostico("ataque_falhou", capturar())
        if not ok and SALVAR_AMOSTRAS:  # sem a flag, a captura do 'depois' seria à toa
            salvar_amostra("ataque_falhou_antes", img, pos, barra=barra)
            salvar_amostra("ataque_falhou_depois", capturar(), pos, barra=barra)
        return ok
    return False


def area_loot():
    """Região em volta do personagem onde o loot da luta cai (esperar_loot)."""
    return (CENTRO[0] - RAIO_LOOT - 70, CENTRO[1] - RAIO_LOOT - 30,
            CENTRO[0] + RAIO_LOOT + 10, CENTRO[1] + RAIO_LOOT + 10)


def recorte_loot(raio):
    """(x1, y1, x2, y2) da janela onde cabe todo rótulo de loot a até `raio` px do
    personagem: os rótulos ficam acima da pilha (até 8 linhas de 22 px) e se estendem
    para os lados. Conferido em 249 quadros, raios 280 e 450: mesmos achados que a janela
    inteira (671 ossos, runas e moedas), em ~65% do tempo."""
    return (max(CENTRO[0] - raio - LOOT_MARGEM_LADO, 0), max(CENTRO[1] - raio - LOOT_MARGEM_ACIMA, 0),
            min(CENTRO[0] + raio + LOOT_MARGEM_LADO, tela.LARGURA),
            min(CENTRO[1] + raio + LOOT_MARGEM_ABAIXO, tela.ALTURA))


def esperar_loot(inicio, osso=True):
    """O osso só aparece ~2 s depois da barra de vida zerar: espera por ele. Monstro que não
    larga Big bones (aprendido sem saber): o loot cai com rótulo de outro item; sem osso
    LOOT_SEM_OSSO_SEG depois dele, para de esperar (o resto do ESPERA_LOOT seria à toa).
    `osso=False` (perfil sem osso): aceita qualquer rótulo novo depois de LOOT_ROTULO_MIN."""
    # Só olha em volta do personagem: ~10x mais rápido que varrer a janela toda
    area = area_loot()
    base = visao.mascara_rotulos(capturar(area))[0]
    st = {"seguidas": 0, "outro_desde": None, "via": None if osso else "rotulo"}

    def pronto():
        rec = capturar(area)
        m = visao.mascara_rotulos(rec)
        if osso and any(distancia(CENTRO, o["pos"]) < RAIO_LOOT
                        for o in visao.detectar_ossos_chao(rec, origem=area[:2], mascara=m)):
            st["via"] = "osso"
            return True
        agora = time.perf_counter()
        st["seguidas"] = st["seguidas"] + 1 if visao.rotulo_novo(base, m[0]) else 0
        if st["seguidas"] == 0:
            st["outro_desde"] = None  # o rótulo novo sumiu (dica do mouse, item passageiro)
        # duas leituras seguidas, e nunca antes de LOOT_ROTULO_MIN da morte (o loot leva ~2-3 s)
        if st["seguidas"] >= 2 and agora - inicio >= LOOT_ROTULO_MIN:
            if not osso:
                return True
            if st["outro_desde"] is None:
                st["outro_desde"] = agora
            elif agora - st["outro_desde"] >= LOOT_SEM_OSSO_SEG and agora - inicio >= LOOT_SEM_OSSO_MIN:
                st["via"] = "rotulo_sem_osso"
                return True
        return False

    restante = max(ESPERA_LOOT - (time.perf_counter() - inicio), 0.1)
    ok, _ = esperar(pronto, restante, 0.05)
    LOG.ev("loot", ok=ok, lat=round(time.perf_counter() - inicio, 3), via=st["via"])
    return ok


# ========== Loops ==========
TELA_PARADA_PX = 1.0  # deslocamento entre dois quadros abaixo disso = câmera parada
TELA_PARADA_TIMEOUT = 1.5
TELA_PARADA_AREA = (560, 250, 1380, 900)  # miolo da área do jogo, sem a interface


def esperar_tela_parar():
    """Espera a câmera parar de deslizar (o personagem parar de andar): dois quadros seguidos do
    miolo do jogo com deslocamento < TELA_PARADA_PX (correlação de fase, reduzido 4x; monstros
    se mexendo não deslocam o quadro inteiro). Devolve {'ok', 'seg'} para o log."""
    def quadro():
        g = cv2.cvtColor(capturar(TELA_PARADA_AREA), cv2.COLOR_BGR2GRAY)
        return np.float32(cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))

    inicio = time.perf_counter()
    anterior = quadro()
    while time.perf_counter() - inicio < TELA_PARADA_TIMEOUT:
        time.sleep(0.06)
        atual = quadro()
        (dx, dy), _ = cv2.phaseCorrelate(anterior, atual)
        if 4 * float(np.hypot(dx, dy)) < TELA_PARADA_PX:
            return {"ok": True, "seg": round(time.perf_counter() - inicio, 2)}
        anterior = atual
    return {"ok": False, "seg": round(time.perf_counter() - inicio, 2)}


def coletar_chao(img, raio, coletar, so_valiosos=False):
    """Uma coleta do chão em volta do personagem: os valiosos (sementes e rótulos azuis) antes de
    tudo (até COLETA_RAIO_MAX, haja monstro à vista ou não), depois runas e moedas (valem mais e
    caem na mesma pilha) e ossos (se `coletar`) até `raio`. `so_valiosos`: só os valiosos (no
    limite de coletas entre ataques e no meio de uma luta). Devolve (agiu, t): agiu = "osso", True (outro
    item) ou False; t = perf_counter do fim da detecção (o tempo 'rotulos' do log)."""
    valiosas = [c for c in COLETAS if c.ativas() and c.item in VALIOSOS]
    if valiosas:
        x1, y1, x2, y2 = recorte_loot(COLETA_RAIO_MAX)
        sub = img[y1:y2, x1:x2]
        azul = visao.mascara_azul(sub)
        mv = visao.mascara_rotulos_valor(sub, azul)  # branco + azul-claro, uma vez para todos
        for coleta in valiosas:  # (ossos=[]: sem o ajuste pela pilha do osso; poupa ~27 ms por ciclo)
            itens = coleta.detectar(sub, ossos=[], mascara=mv, raio=COLETA_RAIO_MAX, origem=(x1, y1),
                                    azul=azul)
            if itens and coleta.pegar(img, itens):
                return True, time.perf_counter()
    if so_valiosos:
        return False, time.perf_counter()
    coletas = [c for c in COLETAS if c.ativas() and c.item not in VALIOSOS]
    if not (coletar or coletas):
        return False, time.perf_counter()
    # só o recorte que pode ter loot dentro do raio (metade do tempo com raio 280)
    x1, y1, x2, y2 = recorte_loot(raio)
    sub = img[y1:y2, x1:x2]
    mascara = visao.mascara_rotulos(sub)
    ossos = [o for o in visao.detectar_ossos_chao(sub, origem=(x1, y1), mascara=mascara)
             if distancia(CENTRO, o["pos"]) <= raio]
    for coleta in coletas:
        itens = coleta.detectar(sub, ossos=ossos, mascara=mascara, raio=raio, origem=(x1, y1))
        if itens and coleta.pegar(img, itens):
            return True, time.perf_counter()
    t = time.perf_counter()
    if coletar and ossos:
        if mochila_cheia(img):
            # sem espaço o osso do chão não entra: enterra um da mochila para abrir um espaço
            inventario = visao.detectar_ossos_inventario(visao.recortar(img, visao.INVENTARIO))
            if inventario and ENTERRO.pode():
                ENTERRO.um(inventario, "cheia")
                return "enterrou", t
            if inventario:
                time.sleep(0.1)  # o enterro anterior ainda não terminou
                return "enterrou", t
            avisar_mochila_cheia()
        elif pegar_osso(img, ossos):
            return "osso", t
    return False, t


MOCHILA_AVISO_SEG = 300  # s entre avisos de mochila cheia sem osso para enterrar
_cheia = {"aviso": 0.0}


def mochila_cheia(img):
    """Mochila com os 28 espaços ocupados? Com ela cheia o osso do chão não entra ("You don't
    have enough inventory space") e o bot ficava tentando pegar, travado."""
    return visao.inventario_cheio(visao.recortar(img, visao.INVENTARIO))


def avisar_mochila_cheia():
    """Cheia e sem osso para enterrar: os ossos ficam no chão. Avisa (com bipe) no máximo a cada
    MOCHILA_AVISO_SEG; com osso na mochila o bot resolve sozinho e não incomoda."""
    if time.time() - _cheia["aviso"] < MOCHILA_AVISO_SEG:
        return
    _cheia["aviso"] = time.time()
    LOG.ev("mochila_cheia")
    tela.bip("erro")
    print("\n🎒 Mochila cheia e sem osso para enterrar: deixo os ossos no chão até abrir espaço "
          "(guarde ou largue itens). Sigo atacando e pegando o que empilha.")


def aproveitar_pausa(img):
    """Pausa por vida baixa: não ataca, mas pega o loot perto (COLETA_RAIO_PERTO) e enterra os
    ossos da mochila, que não começam luta e contam como atividade (sem desconexão por
    inatividade). Em luta (alguém atacou), só enterra. Devolve "osso", True ou False (nada a
    fazer: o laço mantém a conexão)."""
    if garantir_mochila(img):
        return True
    coletar = ossos_ativos.is_set() and PERFIL.get("osso", True)
    inventario = (visao.detectar_ossos_inventario(visao.recortar(img, visao.INVENTARIO))
                  if coletar else [])
    ENTERRO.conferir(len(inventario))
    if visao.estado_combate(img) != "lutando":
        agiu, _ = coletar_chao(img, COLETA_RAIO_PERTO, coletar)
        if agiu:
            LOG.ev("pausa_coleta", item="osso" if agiu == "osso" else "item")
            return agiu
    if inventario and ENTERRO.pode():
        ENTERRO.um(inventario, "pausa")
        return True
    return False


def loop_principal():
    sapos_atacados = 0
    ossos_pegos = 0
    inicio_sessao = time.time()
    em_luta = False
    inicio_luta = 0.0
    livre_desde = None  # em luta, o painel do oponente sumiu desde então (COMBATE_LIVRE_SEG)
    coletas_seguidas = 0  # coletas do chão desde a última tentativa de ataque
    ultimo_valioso = 0.0  # última procura de semente valiosa no meio de uma luta
    em_foco = True
    avisou_inativo = False
    vigia_vida = VIGIA_VIDA  # do módulo: um recomeço do laço (erro) não esquece a pausa
    ultima_vida = 0.0
    sem_leitura_desde = None  # início da sequência atual de leituras vazias do orbe
    avisou_sem_leitura = False
    problema_janela = None
    LOG.ev("inicio", regiao=dict(CAPTURE_REGION), cliente=tela.area_cliente(),
           perfil=NOME_PERFIL, sem_vida=SEM_VIDA)
    while True:
        if not jogo_em_foco():
            if em_foco:
                LOG.ev("foco", ok=False, janela=titulo_janela_ativa())
                em_foco = False
            elif not avisou_inativo and time.time() - ultima_entrada > AVISO_INATIVO_SEG:
                avisou_inativo = True
                LOG.ev("inativo_fora_de_foco", seg=round(time.time() - ultima_entrada))
                if tela.area_cliente() is None:  # sem janela do jogo: jogo_em_foco acha a nova
                    print("\n⚠️ A janela do jogo sumiu (fechada?). Abra e entre no jogo: o bot "
                          "continua sozinho.")
                else:
                    print("\n⚠️ Jogo fora de foco e sem entrada há 4 min: ele desconecta em ~1 min. "
                          "Volte o foco para o jogo (se desconectar, o bot volta sozinho).")
            sem_leitura_desde = None  # tempo fora do jogo não conta como falha de leitura
            time.sleep(0.5)
            continue
        if not em_foco:
            LOG.ev("foco", ok=True)
            em_foco = True
            avisou_inativo = False
            time.sleep(ESPERA_TELA_ASSENTAR)  # logo após o Alt+Tab a tela ainda é a transição
            continue

        # A janela pode ter sido movida: alinha a captura. Se o tamanho não bater com
        # a calibração, as posições fixas da interface estariam erradas -> não age.
        problema = tela.atualizar_janela()
        if problema != problema_janela:
            problema_janela = problema
            if problema:
                LOG.ev("janela", ok=False, problema=problema, cliente=tela.area_cliente())
                print(f"\n⚠️ {problema}. Bot pausado até a janela voltar ao normal.")
            else:
                LOG.ev("janela", ok=True, regiao=dict(CAPTURE_REGION))
                print("✅ Janela do jogo ok" + (" (continua pausado: vida baixa)."
                                                if vigia_vida.pausado else ": bot retomado."))
                time.sleep(ESPERA_TELA_ASSENTAR)
                continue
        if problema:
            sem_leitura_desde = None
            time.sleep(1.0)
            continue

        if not tempo_mouse_parado():
            time.sleep(DELAY_CICLO)
            continue

        t0 = time.perf_counter()
        img = capturar()
        t1 = time.perf_counter()

        # Âncora: a moldura da caixa do orbe de vida tem que estar no lugar exato. Se não
        # estiver (tela em transição, algo por cima), nada neste quadro está onde o bot
        # espera; ele não age nele (exceto com --sem-vida, que desliga essa checagem).
        caixa = visao.diferenca_caixa_vida(img)
        alinhado = SEM_VIDA or caixa <= visao.CAIXA_VIDA_LIMIAR
        # Sem a interface por muito tempo (desconectado?): grava a tela e, se reconhecer o
        # botão, volta ao jogo (reconexao.py; nunca digita nada)
        if alinhado:
            RECONEXAO.no_jogo()
        else:
            RECONEXAO.fora_do_jogo(img)

        # Segurança: vida do jogador pelo orbe (None = não teve certeza de ler o orbe)
        vida = None if SEM_VIDA else visao.vida_pct(img)
        if not SEM_VIDA and time.time() - ultima_vida >= 10:
            ultima_vida = time.time()
            LOG.ev("vida", pct=None if vida is None else round(vida, 3), caixa=round(caixa, 2))
        if vida is not None:
            sem_leitura_desde = None
            avisou_sem_leitura = False
            mudanca = vigia_vida.ler(vida)
            rajada = mudanca is None and vigia_vida.seguidas > 0
            if rajada:
                # Leitura pedindo pausa (ou retomada): confirma já, com quadros novos, antes
                # de qualquer clique. Esperar as próximas voltas do laço deixava o bot abrir
                # outra luta entre uma leitura baixa e a seguinte.
                lidas = [round(vida, 3)]
                for _ in range(VIDA_TENTATIVAS_CONFIRMAR):
                    time.sleep(0.05)
                    img = capturar()
                    nova = visao.vida_pct(img)
                    lidas.append(None if nova is None else round(nova, 3))
                    mudanca = vigia_vida.ler(nova)
                    vida = vida if nova is None else nova
                    if mudanca or vigia_vida.seguidas == 0:
                        break
                if mudanca is None:
                    vigia_vida.descartar()  # não se repetiu: leitura isolada, ignora
                    LOG.ev("vida_isolada", lidas=lidas)
            if mudanca == "pausar":
                foto = os.path.join(visao.BASE_DIR, "logs",
                                    time.strftime("vida_baixa_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(foto, img)  # para conferir depois o que foi lido
                LOG.ev("vida_baixa", pct=round(vida, 3), foto=os.path.basename(foto))
                print(f"\n🚨 VIDA BAIXA (~{100 * vida:.0f}%): bot pausado. Coma ou saia da área; ele "
                      f"volta sozinho acima de {VIDA_RETOMAR:.0%}.")
                tela.bip("erro")  # o terminal fica atrás do jogo
            elif mudanca == "retomar":
                LOG.ev("vida_ok", pct=round(vida, 3))
                print(f"💚 Vida recuperada (~{100 * vida:.0f}%): bot retomado.")
            if rajada:
                continue  # o quadro do início do ciclo ficou velho: recomeça
        elif not SEM_VIDA:
            # Sem leitura a pausa por vida baixa não funciona: avisa em vez de seguir calado
            agora = time.time()
            if sem_leitura_desde is None:
                sem_leitura_desde = agora
            limite = VIDA_AVISO_PAUSADO if vigia_vida.pausado else VIDA_AVISO_SEM_LEITURA
            if not avisou_sem_leitura and agora - sem_leitura_desde > limite:
                avisou_sem_leitura = True
                LOG.ev("vida_sem_leitura", seg=round(agora - sem_leitura_desde),
                       pausado=vigia_vida.pausado, caixa=round(caixa, 2))
                if vigia_vida.pausado:
                    consequencia = "O bot segue pausado."
                elif not alinhado:
                    consequencia = ("Não acho a interface do jogo no lugar de sempre: o bot fica parado "
                                    "até ela voltar. Confira: jogo logado, janela maximizada, layout "
                                    "'Resizable - Classic' e nada por cima dos orbes ao lado do minimapa.")
                else:
                    consequencia = "A pausa por vida baixa NÃO está funcionando."
                if "fora" not in RECONEXAO.avisou:  # desconectado: o aviso da reconexão já basta
                    print(f"\n⚠️ Não consigo ler o orbe de vida há {agora - sem_leitura_desde:.0f} s "
                          f"(ele está visível, sem nada por cima?). {consequencia}")
        if vigia_vida.pausado:
            # tela de jogo (não a de login): enquanto a vida sobe, pega o loot perto e enterra
            # ossos (não ataca); sem nada para fazer, só mantém a conexão
            agiu = aproveitar_pausa(img) if alinhado else False
            if agiu == "osso":
                ossos_pegos += 1
                print(f"🦴 Ossos coletados do chão (na pausa): {ossos_pegos}")
            if not agiu:
                if alinhado:
                    manter_conectado()
                time.sleep(0.5)
            continue
        if not alinhado:
            time.sleep(0.2)
            continue
        if garantir_mochila(img):
            continue
        estado = visao.estado_combate(img)
        coletar = ossos_ativos.is_set() and PERFIL.get("osso", True)
        inventario = (
            visao.detectar_ossos_inventario(visao.recortar(img, visao.INVENTARIO))
            if coletar
            else []
        )
        ENTERRO.conferir(len(inventario))

        # Painel do oponente sumido no meio da luta: espera um pouco antes de dar a luta por
        # encerrada (ele pode voltar; visto com Skeleton)
        if em_luta and estado == "livre":
            agora = time.time()
            if livre_desde is None:
                livre_desde = agora
                foto_diagnostico("combate_livre", img)
            if agora - livre_desde < COMBATE_LIVRE_SEG:
                time.sleep(0.1)
                continue
        elif livre_desde is not None:
            if em_luta and estado == "lutando":
                LOG.ev("painel_voltou", seg=round(time.time() - livre_desde, 2))
            livre_desde = None

        # Fim de luta: enterra um osso enquanto espera o loot cair
        if em_luta and estado != "lutando":
            livre_desde = None
            em_luta = False
            LOG.ev("fim_combate", dur=round(time.time() - inicio_luta, 2), estado=estado)
            if estado == "morto":
                # só medição: há monstro livre perto na hora da morte? (decide se vale atacar
                # antes de o loot cair, ~2,5 s por monstro; ver analisar-logs-madrugada)
                viz = sorted((round(distancia(CENTRO, s["pos"])),
                              visao.barra_vida_perto(img, s["pos"], *PERFIL["barra"]))
                             for s in visao.detectar_sapos(img, PERFIL))
                LOG.ev("vizinhos_fim", viz=viz[:5])
            if coletar and estado == "morto":
                if inventario and ENTERRO.pode():
                    ENTERRO.um(inventario, "loot")
                esperar_loot(t0)
            elif estado == "morto" and not PERFIL.get("osso", True) and any(c.ativas() for c in COLETAS):
                esperar_loot(t0, False)  # perfil sem osso: qualquer rótulo novo
            continue

        # Em combate: não clica em outro alvo; aproveita para enterrar
        if estado == "lutando":
            if not em_luta:  # luta que não começou por um clique nosso
                em_luta = True
                inicio_luta = time.time()
                LOG.ev("combate_externo")
            # semente valiosa no chão: larga a luta um instante e pega (outro jogador a pegaria; o
            # monstro continua no personagem e a luta segue depois). Visto em 16/11: uma Ranarr
            # seed (22K) ficou no chão enquanto o bot lutava.
            if time.time() - ultimo_valioso >= VALIOSO_INTERVALO:
                ultimo_valioso = time.time()
                if coletar_chao(img, COLETA_RAIO_MAX, False, so_valiosos=True)[0]:
                    LOG.ev("valioso_em_luta")
                    continue
            if inventario and ENTERRO.pode():
                ENTERRO.um(inventario, "combate")
            time.sleep(0.1)
            continue

        # Fora de combate só enterra se acumulou demais
        if len(inventario) >= MIN_OSSOS_ENTERRAR:
            if ENTERRO.pode():
                ENTERRO.um(inventario, "limite")
            else:
                time.sleep(0.1)
            continue

        # Monstros na tela: com algum à vista, o loot longe fica para depois
        t2 = time.perf_counter()
        sapos = visao.detectar_sapos(img, PERFIL)
        if not sapos and PERFIL.get("nome_npc") is not None:
            avisar_sem_nomes(img)
        t3 = t4 = time.perf_counter()
        raio = COLETA_RAIO_PERTO if sapos else COLETA_RAIO_MAX

        # Loot do chão: runas e moedas primeiro (valem mais e caem na mesma pilha), depois ossos.
        # Com monstro livre à vista, no máximo MAX_COLETAS_SEGUIDAS coletas entre um ataque e
        # outro: com muito loot no chão o bot ficava só catando (visto: 1 ataque/min em vez de 3;
        # 152 de 302 s entre lutas eram coleta). O que sobra é pego depois da próxima luta.
        livre = coletas_seguidas >= MAX_COLETAS_SEGUIDAS and any(
            visao.barra_vida_perto(img, s["pos"], *PERFIL["barra"]) < BARRA_SAPO_OCUPADO for s in sapos)
        # (as sementes valiosas não entram no limite: são pegas antes que outro jogador as pegue)
        agiu, t4 = coletar_chao(img, raio, coletar, so_valiosos=livre)
        if agiu == "osso":
            ossos_pegos += 1
            print(f"🦴 Ossos coletados do chão: {ossos_pegos}")
        if agiu:
            if agiu != "enterrou":  # enterrar para abrir espaço não conta como coleta
                coletas_seguidas += 1
            continue

        # A espera da mira (~1 s, tela andando) ou uma coleta que falhou pelo menu (~3,9 s) deixou
        # o quadro velho: ataca por um quadro novo, conferido como o do começo do ciclo (visto: 2
        # hovers falhos com quadro de 1,0-1,3 s). A mira perdida conta mesmo com o quadro novo.
        # Logo depois de uma coleta longe o personagem ainda anda e a câmera desliza: o clique de
        # ataque caía no chão ou em outro monstro (visto: 2 falhas em 5 ataques assim que o limite
        # de coletas pôs o ataque logo depois da coleta). Espera a tela parar.
        andando = (time.time() - _ultima_coleta["t"] < MIRA_LONGA[0]
                   and _ultima_coleta["dist"] > MIRA_LONGA[1])
        if andando:
            LOG.ev("tela_parou", **esperar_tela_parar())
        if andando or _mira["perdida"] or time.perf_counter() - t0 > QUADRO_VELHO_SEG:
            _mira["perdida"] = False
            t0 = time.perf_counter()
            img = capturar()
            t1 = time.perf_counter()
            desalinhado = visao.diferenca_caixa_vida(img) > visao.CAIXA_VIDA_LIMIAR
            if desalinhado or visao.estado_combate(img) == "lutando":
                continue
            sapos = visao.detectar_sapos(img, PERFIL)
            LOG.ev("quadro_renovado", n_sapos=len(sapos))

        # Atacar o sapo mais próximo
        LOG.ev("ciclo", captura=round(t1 - t0, 4), rotulos=round(t4 - t3, 4),
               sapos=round(t3 - t2, 4), n_sapos=len(sapos))
        coletas_seguidas = 0  # vai tentar atacar (ou não há monstro): o limite recomeça
        if sapos and atacar_sapo(img, sapos, t0):
            for coleta in COLETAS:
                coleta.ignoradas.clear()  # o personagem vai andar: as posições de tela mudam
            sapos_atacados += 1
            em_luta = True
            inicio_luta = time.time()
            por_hora = 3600 * sapos_atacados / max(time.time() - inicio_sessao, 1)
            print(f"⚔️ {PERFIL['nome']} atacados: {sapos_atacados} (~{por_hora:.0f}/h)")
        time.sleep(DELAY_CICLO)


def loop_atalhos():
    while True:
        if keyboard.is_pressed("F8"):
            if not PERFIL.get("osso", True):
                print(f"🩴 {PERFIL['nome']} não larga 'Big bones' (perfil sem osso): F8 não faz nada")
            elif ossos_ativos.is_set():
                ossos_ativos.clear()
                print("🩴 Coleta de ossos DESATIVADA!")
            else:
                ossos_ativos.set()
                print("🩴 Coleta de ossos ATIVADA!")
            time.sleep(0.5)
        time.sleep(0.05)


def executar_principal():
    """Roda o laço; um erro inesperado (ex.: falha passageira da captura de tela) vai para o
    log e o laço recomeça em 5 s, mantendo a pausa por vida (VIGIA_VIDA). 3 erros em 10 min
    encerram o bot, como antes."""
    import traceback

    erros = []
    while True:
        try:
            loop_principal()
        except pyautogui.FailSafeException:
            LOG.ev("encerrado", motivo="failsafe", mouse=list(pyautogui.position()))
            print("\n🛑 Mouse no canto da tela: bot encerrado (failsafe).")
            break
        except Exception as erro:
            traceback.print_exc()
            tela.reiniciar_captura()  # a próxima captura abre do zero
            agora = time.time()
            erros = [t for t in erros if agora - t < 600] + [agora]
            LOG.ev("erro", tipo=type(erro).__name__, msg=str(erro)[:200],
                   onde=traceback.format_exc()[-1500:], seguidos=len(erros))
            if len(erros) >= 3:
                print(f"\n🛑 O bot parou depois de 3 erros em 10 min. Ligue de novo; se repetir, "
                      f"guarde o arquivo {LOG.caminho}.")
                tela.bip("erro")
                time.sleep(1)  # o bipe é assíncrono: o os._exit logo abaixo cortaria o som
                break
            print("\n⚠️ Erro inesperado (evento 'erro' no log): recomeço em 5 s.")
            time.sleep(5)
    os._exit(1)


# ========== Execução ==========
if __name__ == "__main__":
    print(f"🔁 Bot iniciado — alvo: {PERFIL['nome']}. Pressione CTRL+C para parar.")
    if PERFIL.get("osso", True):
        print("   (só age com o jogo em primeiro plano; F8 liga/desliga os ossos)")
    else:
        print("   (só age com o jogo em primeiro plano; perfil sem osso: não pega nem enterra ossos)")
    if SEM_VIDA:
        print("   pausa por vida baixa DESLIGADA (--sem-vida)")
    sem_itens = "nada além dos ossos" if PERFIL.get("osso", True) else "nada"
    print("   pega do chão: " + (", ".join(c.plural for c in COLETAS if c.ligado) or sem_itens))
    print(f"   log: {LOG.caminho}")
    try:
        threading.Thread(target=executar_principal, daemon=True).start()
        threading.Thread(target=loop_atalhos, daemon=True).start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Execução finalizada pelo usuário.")
