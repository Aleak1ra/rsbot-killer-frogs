"""Bot de Giant frog para o cliente oficial do Old School RuneScape.

Uso:  python main.py [--monstro sapo|hillgiant] [--sem-vida] [--sem-runas] [--sem-moedas] [--amostras]
  --monstro   qual monstro caçar (padrão: sapo = Giant frog)
  --sem-vida  desliga a pausa por vida baixa
  --sem-runas não pega as runas que caem no chão
  --sem-moedas não pega as moedas (Coins) que caem no chão
  --amostras  salva frames rotulados em amostras/ (para testar_visao.py)

Cada sessão grava um log de eventos em logs/sessao_*.jsonl; use
`python analisar_log.py` para ver os tempos medidos.
"""
import ctypes
import json
import os
import random
import sys
import threading
import time

import cv2
import keyboard
import numpy as np
import pyautogui
from mss import mss

import reconexao
import tela
import visao

# Perfil do monstro: python main.py --monstro hillgiant
NOME_PERFIL = sys.argv[sys.argv.index("--monstro") + 1] if "--monstro" in sys.argv else "sapo"
try:  # embutido (visao.PERFIS) ou aprendido com ensinar.py (perfis/<nome>.json)
    PERFIL = visao.carregar_perfil(NOME_PERFIL)
except KeyError:
    sys.exit(f"Monstro desconhecido: {NOME_PERFIL}. Opções: {', '.join(visao.perfis_disponiveis())}")
except ValueError as erro:  # perfis/<nome>.json existe mas está incompleto ou estragado
    sys.exit(f"Perfil {NOME_PERFIL} com problema: {erro}")

# ========== Configurações ==========
TITULO_JANELA = "Old School RuneScape"

DELAY_CICLO = 0.05
TEMPO_MOUSE_PARADO = 1.0  # pausa quando VOCÊ mexe o mouse

# Texto de ação ao passar o mouse
HOVER_ATRASO_MIN = 0.06  # ignora leituras antes disso (texto antigo ainda na tela)
HOVER_TIMEOUT = 0.25  # medido: o texto certo aparece em ~0,09 s
MAX_TENTATIVAS_SAPO = 4
# Mira: antes de clicar, relocaliza o sapo num recorte em volta do mouse
RAIO_MIRA = PERFIL["raio_mira"]
MIRA_DESLOC_MIN = 4  # px; abaixo disso não vale mexer o mouse
# Sapo com barra de vida sobre a cabeça já está lutando com outro jogador.
# Medido: 8 de 9 ataques que falharam tinham >= 36 px de barra; os 47 que deram certo, 0.
BARRA_SAPO_OCUPADO = 25

# X de clique: aparece ~40 ms depois do clique e dura ~300 ms
CLIQUE_TIMEOUT = 0.25

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
SEM_VIDA = "--sem-vida" in sys.argv

# Combate
ESPERA_INICIO_COMBATE = 5.0  # medido em 120 ataques: 1,5 s típico, 4,25 s no pior caso
ESPERA_LOOT = PERFIL["espera_loot"]  # tempo para o osso aparecer depois da barra zerar
RAIO_LOOT = 320  # px em volta do personagem
LOOT_ROTULO_MIN = 1.0  # s: perfil sem osso só aceita o loot depois disso (medido: cai em 1,8-3,1 s)

# Itens do chão pegos antes dos ossos: runas e moedas (--sem-runas / --sem-moedas desligam)
PEGAR_RUNAS = "--sem-runas" not in sys.argv
PEGAR_MOEDAS = "--sem-moedas" not in sys.argv
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
COLETA_IGNORAR_RAIO = 25  # px em volta da pilha que falhou
COLETA_FALHAS_PAUSA = 3  # falhas seguidas -> deixa aquele item de lado por COLETA_PAUSA s
COLETA_PAUSA = 30.0
# Conferência da coleta pelo inventário (só a área dos itens, visao.INVENTARIO_ITENS): px
# que mudaram de tom (> INVENTARIO_DIF_TOM em algum canal). Medido em 429 pares de quadros:
# sem mudança, 0 px; um dígito trocando ("6" -> "8" numa pilha de runas), 17 px.
INVENTARIO_DIF_TOM = 40
INVENTARIO_DIF_PX = 6
INVENTARIO_ESPERA_MIN = 0.25  # s após o clique antes de aceitar a mudança (filtra as passageiras)
INVENTARIO_RAPIDA = 0.4  # coleta confirmada antes disso grava o inventário antes|depois (diagnóstico)
COLETA_SOB_PERSONAGEM = 90  # px do centro: depois de andar até a pilha, ela fica aqui
# Onde procurar o item (texto de ação "Take") quando o clique caiu fora dele: primeiro uma
# linha de rótulo abaixo, depois em volta
PROCURA_TAKE = [(0, 22), (12, 0), (-12, 0), (0, 10), (12, 22), (-12, 22), (24, 0), (-24, 0), (0, -10)]
# Mira com a tela parada (mirar_parado): região relida em volta do alvo (os rótulos ficam
# acima da pilha, até ~8 linhas de 22 px) e tolerância entre duas leituras seguidas
MIRA_RAIO_X, MIRA_ACIMA, MIRA_ABAIXO = 260, 230, 90
MIRA_PARADO_PX = 3
MIRA_TIMEOUT = 1.0
COLETA_MAX_FOTOS = 40  # fotos de falha por sessão (logs/coleta_falhas/)

# Ossos
ESPERA_PEGAR_OSSO = 2.5  # medido em 48 coletas: 1,2 s típico, 1,6 s no pior caso
INTERVALO_ENTERRAR = 1.3  # o jogo ignora cliques de enterrar mais rápidos que ~2 ticks
MIN_OSSOS_ENTERRAR = 10  # fora de combate só enterra se acumular isso
# Onde o mouse descansa depois de clicar no inventário (coluna de pedra, sem tooltip)
PONTO_NEUTRO = (1593, 960)
# O jogo desconecta depois de 5 min sem mouse/teclado (visto às 03:32 de 15/11, numa pausa
# por vida baixa). Parado (pausa por vida), o bot mexe o mouse na coluna de pedra e toca o
# Shift a cada intervalo sorteado entre esses valores (s), sem clicar em nada.
ANTI_AFK_SEG = (60.0, 120.0)

pyautogui.PAUSE = 0.03

SALVAR_AMOSTRAS = "--amostras" in sys.argv
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
class Registro:
    def __init__(self):
        pasta = os.path.join(visao.BASE_DIR, "logs")
        os.makedirs(pasta, exist_ok=True)
        self.caminho = os.path.join(pasta, time.strftime("sessao_%Y%m%d_%H%M%S.jsonl"))
        self.arquivo = open(self.caminho, "a", encoding="utf-8")
        self.t0 = time.time()
        self.trava = threading.Lock()

    def ev(self, nome, **dados):
        linha = {"t": round(time.time() - self.t0, 3), "ev": nome, **dados}
        with self.trava:
            self.arquivo.write(
                json.dumps(linha, default=lambda o: o.item() if hasattr(o, "item") else str(o))
                + "\n"
            )
            self.arquivo.flush()


LOG = Registro()
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
_local = threading.local()


def capturar(regiao=None):
    """Captura a janela inteira ou só uma região (x1, y1, x2, y2) dela."""
    if not hasattr(_local, "sct"):
        _local.sct = mss()
    if regiao is None:
        area = CAPTURE_REGION
    else:
        x1, y1, x2, y2 = regiao
        area = {
            "left": CAPTURE_REGION["left"] + x1,
            "top": CAPTURE_REGION["top"] + y1,
            "width": x2 - x1,
            "height": y2 - y1,
        }
    return cv2.cvtColor(np.array(_local.sct.grab(area)), cv2.COLOR_BGRA2BGR)


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
    r = visao.CLIQUE_RAIO
    x = min(max(int(pos[0]), r), CAPTURE_REGION["width"] - r)
    y = min(max(int(pos[1]), r), CAPTURE_REGION["height"] - r)
    regiao = (x - r, y - r, x + r, y + r)
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
    x = min(max(int(pos[0]), RAIO_MIRA), CAPTURE_REGION["width"] - RAIO_MIRA)
    y = min(max(int(pos[1]), RAIO_MIRA), CAPTURE_REGION["height"] - RAIO_MIRA)
    local = capturar((x - RAIO_MIRA, y - RAIO_MIRA, x + RAIO_MIRA, y + RAIO_MIRA))
    achado, _ = visao.sapo_mais_proximo(
        local, (pos[0] - x + RAIO_MIRA, pos[1] - y + RAIO_MIRA), PERFIL
    )
    if achado is None:
        return pos, None
    novo = (x - RAIO_MIRA + achado[0], y - RAIO_MIRA + achado[1])
    return novo, distancia(pos, novo)


def distancia(p1, p2):
    return float(np.linalg.norm(np.subtract(p1, p2)))


# ========== Leituras com espera ==========
def esperar_texto(chaves, timeout=HOVER_TIMEOUT):
    """Depois de mover o mouse, espera o texto de ação virar um dos `chaves`.

    Devolve (chave ou None, latência em s, {chave: maior score visto}, última faixa).
    Exige duas leituras seguidas iguais para não clicar com texto de passagem.
    """
    inicio = time.perf_counter()
    melhores = {c: 0.0 for c in chaves}
    anterior = None
    faixa = None
    while True:
        dt = time.perf_counter() - inicio
        if dt > timeout:
            return None, dt, melhores, faixa
        if dt < HOVER_ATRASO_MIN:
            time.sleep(0.01)
            continue
        faixa = capturar(visao.FAIXA_TEXTO_ACAO)
        achou = None
        for c in chaves:
            s = visao.score_texto(faixa, c)
            melhores[c] = max(melhores[c], s)
            if achou is None and s >= visao.limiar_texto(c):
                achou = c
        if achou is not None and achou == anterior:
            return achou, dt, melhores, faixa
        anterior = achou
        time.sleep(0.01)


def contar_ossos_inventario():
    return visao.detectar_ossos_inventario(capturar(visao.INVENTARIO))


def esperar(condicao, timeout, intervalo=0.05):
    """Espera `condicao()` ficar verdadeira. Devolve (ok, tempo decorrido)."""
    inicio = time.perf_counter()
    while True:
        if condicao():
            return True, time.perf_counter() - inicio
        if time.perf_counter() - inicio > timeout:
            return False, time.perf_counter() - inicio
        time.sleep(intervalo)


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
    pilha. Relê a região do alvo até duas leituras seguidas concordarem (±3 px).
    `detectar_local(recorte, origem)` devolve as posições candidatas (coordenadas da janela).
    """
    x, y = int(pos[0]), int(pos[1])
    regiao = (max(x - MIRA_RAIO_X, 0), max(y - MIRA_ACIMA, 0),
              min(x + MIRA_RAIO_X, CAPTURE_REGION["width"]), min(y + MIRA_ABAIXO, CAPTURE_REGION["height"]))
    atual, anterior = pos, None
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
                salvar_amostra("osso_rej", img, pos, score=osso["score"])
                continue
        salvar_amostra("osso_ok", img, pos, score=osso["score"])
        ok, dt = esperar(lambda: len(contar_ossos_inventario()) > antes,
                         ESPERA_PEGAR_OSSO + dist / 150)
        LOG.ev("pegar_osso", ok=ok, lat=round(dt, 3), dist=round(dist), clique=clique, via=via)
        if ok:
            ENTERRO.pegos_desde += 1
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

    def __init__(self, item, plural, emoji, ligado):
        self.item, self.plural, self.emoji, self.ligado = item, plural, emoji, ligado
        self.total = 0
        self.falhas = 0
        self.pausa_ate = 0.0
        self.ignoradas = []  # [(pos, até quando)] pilhas que falharam

    def ativas(self):
        return self.ligado and time.time() >= self.pausa_ate

    def detectar(self, img, ossos=None, mascara=None, raio=COLETA_RAIO_MAX, origem=(0, 0)):
        """Itens a até `raio` px do personagem, sem as pilhas que falharam há pouco.

        `img` pode ser um recorte da janela com canto em `origem` (ver recorte_loot)."""
        agora = time.time()
        self.ignoradas = [(p, t) for p, t in self.ignoradas if t > agora]
        itens = visao.detectar_itens_chao(img, self.item, origem=origem, ossos=ossos, mascara=mascara)
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

    def _foto_inventario(self, antes, depois):
        """Inventário antes|depois lado a lado (coleta confirmada rápido demais)."""
        global _fotos_falha
        if _fotos_falha >= COLETA_MAX_FOTOS:
            return
        _fotos_falha += 1
        pasta = os.path.join(visao.BASE_DIR, "logs", "coleta_falhas")
        os.makedirs(pasta, exist_ok=True)
        cv2.imwrite(os.path.join(pasta, time.strftime("%H%M%S") + f"_{self.item}_inventario_rapido.png"),
                    np.hstack([antes, depois]))

    def _falhou(self, img, alvo, motivo, depois=None, ignorar="alvo"):
        """`ignorar`: ponto da tela onde a pilha que falhou está agora ("alvo" = onde estava
        no início; None = não se sabe, não ignora nada)."""
        self._foto_falha(img, alvo, motivo, depois)
        if ignorar == "alvo":
            ignorar = alvo["pos"]
        if ignorar is not None:
            self.ignoradas.append((ignorar, time.time() + COLETA_IGNORAR_SEG))
        self.falhas += 1
        if self.falhas >= COLETA_FALHAS_PAUSA:
            self.falhas = 0
            self.pausa_ate = time.time() + COLETA_PAUSA
            LOG.ev(f"{self.plural}_pausa", seg=COLETA_PAUSA)
            print(f"{self.emoji} Não consegui pegar {self.plural} {COLETA_FALHAS_PAUSA}x seguidas: "
                  f"deixando de lado por {COLETA_PAUSA:.0f} s.")

    def pegar(self, img, itens):
        alvo = min(itens, key=lambda r: distancia(CENTRO, r["pos"]))
        pos = mirar_parado(alvo["pos"], lambda rec, o: [
            i["pos"] for i in visao.detectar_itens_chao(rec, self.item, origem=o)])
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
        achar = lambda rec: visao.achar_take_item_no_menu(rec, self.item)  # noqa: E731
        r = pegar_pelo_menu_pilha(pos, achar, self.item)
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
        leitura = {"inv": inv_antes, "px": 0}
        clique = time.perf_counter()

        def mudou():
            leitura["inv"] = capturar(visao.INVENTARIO_ITENS)
            dif = cv2.absdiff(leitura["inv"], inv_antes).max(axis=2) > INVENTARIO_DIF_TOM
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
            print(f"{self.emoji} {self.plural.capitalize()} pegas: {self.total}")
            if dt < INVENTARIO_RAPIDA:  # diagnóstico: o que mudou no inventário tão rápido?
                self._foto_inventario(inv_antes, leitura["inv"])
        else:
            # o personagem andou até a pilha: ela (se ficou) está agora perto dele
            depois = capturar()
            perto = [i["pos"] for i in visao.detectar_itens_chao(depois, self.item)
                     if distancia(i["pos"], CENTRO) <= COLETA_SOB_PERSONAGEM]
            onde = min(perto, key=lambda p: distancia(p, CENTRO)) if perto else None
            self._falhou(img, alvo, "inventario_igual", depois=depois, ignorar=onde)
        return True


_fotos_falha = 0
# Ordem de prioridade: runas, depois moedas (os ossos vêm depois, no laço)
RUNAS = ColetaChao("runa", "runas", "🔮", PEGAR_RUNAS)
MOEDAS = ColetaChao("moedas", "moedas", "🪙", PEGAR_MOEDAS)
COLETAS = (RUNAS, MOEDAS)


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
            continue
        salvar_amostra(f"{NOME_PERFIL}_ok", img, pos, area=sapo["area"])
        novo, desloc = refinar_mira(pos)
        if desloc is not None and desloc >= MIRA_DESLOC_MIN:
            mover(novo)
            pos = novo
        desloc = None if desloc is None else round(desloc, 1)
        clique = clicar_conferindo(pos, "sapo")
        if clique in ("chao", "cancelado"):  # o sapo pulou (ou clique cancelado): próximo ciclo
            LOG.ev("atacar", ok=False, lat=0.0, dist=round(dist), barra=barra, clique=clique,
                   desloc=desloc)
            return False
        ok, dt = esperar(lambda: visao.estado_combate(capturar()) == "lutando",
                         ESPERA_INICIO_COMBATE, 0.1)
        LOG.ev("atacar", ok=ok, lat=round(dt, 3), dist=round(dist), barra=barra, clique=clique,
               desloc=desloc)
        if not ok:
            salvar_amostra("ataque_falhou_antes", img, pos, barra=barra)
            salvar_amostra("ataque_falhou_depois", capturar(), pos, barra=barra)
        return ok
    return False


def area_loot():
    """Região em volta do personagem onde o loot da luta cai (esperar_loot)."""
    return (CENTRO[0] - RAIO_LOOT - 70, CENTRO[1] - RAIO_LOOT - 30,
            CENTRO[0] + RAIO_LOOT + 10, CENTRO[1] + RAIO_LOOT + 10)


def esperar_loot_rotulo(inicio):
    """Monstro sem "Big bones" (perfil aprendido com osso nulo): espera aparecer texto
    branco novo de rótulo de item em volta do personagem, ou o tempo do perfil acabar."""
    area = area_loot()
    base = visao.mascara_rotulos(capturar(area))[0]
    seguidas = [0]

    def caiu():
        # duas leituras seguidas, e nunca antes de LOOT_ROTULO_MIN da morte (o loot leva ~2-3 s)
        novo = visao.rotulo_novo(base, visao.mascara_rotulos(capturar(area))[0])
        seguidas[0] = seguidas[0] + 1 if novo else 0
        return seguidas[0] >= 2 and time.perf_counter() - inicio >= LOOT_ROTULO_MIN

    restante = max(ESPERA_LOOT - (time.perf_counter() - inicio), 0.1)
    ok, _ = esperar(caiu, restante, 0.05)
    LOG.ev("loot", ok=ok, lat=round(time.perf_counter() - inicio, 3), via="rotulo")
    return ok


def recorte_loot(raio):
    """(x1, y1, x2, y2) da janela onde cabe todo rótulo de loot a até `raio` px do
    personagem: os rótulos ficam acima da pilha (até 8 linhas de 22 px) e se estendem
    para os lados. Conferido em 249 quadros, raios 280 e 450: mesmos achados que a janela
    inteira (671 ossos, runas e moedas), em ~65% do tempo."""
    return (max(CENTRO[0] - raio - LOOT_MARGEM_LADO, 0), max(CENTRO[1] - raio - LOOT_MARGEM_ACIMA, 0),
            min(CENTRO[0] + raio + LOOT_MARGEM_LADO, tela.LARGURA),
            min(CENTRO[1] + raio + LOOT_MARGEM_ABAIXO, tela.ALTURA))


def esperar_loot(inicio):
    """O osso só aparece ~2 s depois da barra de vida zerar: espera por ele."""
    # Só olha em volta do personagem: ~10x mais rápido que varrer a janela toda
    area = area_loot()

    def osso_perto():
        ossos = visao.detectar_ossos_chao(capturar(area), origem=area[:2])
        return any(distancia(CENTRO, o["pos"]) < RAIO_LOOT for o in ossos)

    restante = max(ESPERA_LOOT - (time.perf_counter() - inicio), 0.1)
    ok, _ = esperar(osso_perto, restante, 0.05)
    LOG.ev("loot", ok=ok, lat=round(time.perf_counter() - inicio, 3))
    return ok


# ========== Loops ==========
def loop_principal():
    sapos_atacados = 0
    ossos_pegos = 0
    inicio_sessao = time.time()
    em_luta = False
    inicio_luta = 0.0
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
                print(f"\n🚨 VIDA BAIXA (~{100 * vida:.0f}%): bot pausado. Coma ou saia da área!")
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
                    consequencia = ("O bot fica parado até ela aparecer "
                                    "(para ignorar: --sem-vida).")
                else:
                    consequencia = "A pausa por vida baixa NÃO está funcionando."
                print(f"\n⚠️ Não consigo ler o orbe de vida há {agora - sem_leitura_desde:.0f} s "
                      f"(ele está visível, sem nada por cima?). {consequencia}")
        if vigia_vida.pausado:
            if alinhado:  # tela de jogo (não a de login): mantém a conexão enquanto espera
                manter_conectado()
            time.sleep(0.5)
            continue
        if not alinhado:
            time.sleep(0.2)
            continue
        estado = visao.estado_combate(img)
        coletar = ossos_ativos.is_set() and PERFIL.get("osso", True)
        inventario = (
            visao.detectar_ossos_inventario(visao.recortar(img, visao.INVENTARIO))
            if coletar
            else []
        )
        ENTERRO.conferir(len(inventario))

        # Fim de luta: enterra um osso enquanto espera o loot cair
        if em_luta and estado != "lutando":
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
                esperar_loot_rotulo(t0)
            continue

        # Em combate: não clica em outro alvo; aproveita para enterrar
        if estado == "lutando":
            if not em_luta:  # luta que não começou por um clique nosso
                em_luta = True
                inicio_luta = time.time()
                LOG.ev("combate_externo")
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
        t3 = t4 = time.perf_counter()
        raio = COLETA_RAIO_PERTO if sapos else COLETA_RAIO_MAX

        # Loot do chão: runas e moedas primeiro (valem mais e caem na mesma pilha), depois ossos
        coletas = [c for c in COLETAS if c.ativas()]
        if coletar or coletas:
            # só o recorte que pode ter loot dentro do raio (metade do tempo com raio 280)
            x1, y1, x2, y2 = recorte_loot(raio)
            sub = img[y1:y2, x1:x2]
            mascara = visao.mascara_rotulos(sub)
            ossos = [o for o in visao.detectar_ossos_chao(sub, origem=(x1, y1), mascara=mascara)
                     if distancia(CENTRO, o["pos"]) <= raio]
            agiu = False
            for coleta in coletas:
                itens = coleta.detectar(sub, ossos=ossos, mascara=mascara, raio=raio, origem=(x1, y1))
                if itens and coleta.pegar(img, itens):
                    agiu = True
                    break
            t4 = time.perf_counter()
            if agiu:
                continue
            if coletar and ossos and pegar_osso(img, ossos):
                ossos_pegos += 1
                print(f"🦴 Ossos coletados do chão: {ossos_pegos}")
                continue

        # A espera da mira deixou o quadro velho (~1 s, tela andando): ataca por um quadro novo,
        # conferido como o do começo do ciclo (visto: 2 hovers falhos com quadro de 1,0-1,3 s)
        if _mira["perdida"]:
            _mira["perdida"] = False
            img = capturar()
            t0 = time.perf_counter()
            desalinhado = visao.diferenca_caixa_vida(img) > visao.CAIXA_VIDA_LIMIAR
            if desalinhado or visao.estado_combate(img) == "lutando":
                continue
            sapos = visao.detectar_sapos(img, PERFIL)
            LOG.ev("quadro_renovado", n_sapos=len(sapos))

        # Atacar o sapo mais próximo
        LOG.ev("ciclo", captura=round(t1 - t0, 4), rotulos=round(t4 - t3, 4),
               sapos=round(t3 - t2, 4), n_sapos=len(sapos))
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
            sct = _local.__dict__.pop("sct", None)  # recria a captura do zero
            if sct is not None:
                try:
                    sct.close()
                except Exception:
                    pass
            agora = time.time()
            erros = [t for t in erros if agora - t < 600] + [agora]
            LOG.ev("erro", tipo=type(erro).__name__, msg=str(erro)[:200],
                   onde=traceback.format_exc()[-1500:], seguidos=len(erros))
            if len(erros) >= 3:
                print("\n🛑 3 erros em 10 min: bot encerrado (evento 'erro' no log).")
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
