"""Entrada e saída compartilhadas pelos bots: janela do jogo, captura, mouse e log.

Todas as posições são em coordenadas da captura calibrada (janela maximizada; ver
visao.CLIENTE_CALIBRADO). A captura acompanha a área do jogo onde quer que a janela
esteja: chame atualizar_janela() a cada ciclo.
"""
import ctypes
import json
import os
import threading
import time
from ctypes import wintypes

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from mss import mss

import visao

TITULO_JANELA = "Old School RuneScape"
TEMPO_MOUSE_PARADO = 1.0  # pausa quando VOCÊ mexe o mouse
HOVER_ATRASO_MIN = 0.06  # ignora leituras antes disso (texto antigo ainda na tela)
HOVER_TIMEOUT = 0.25
CLIQUE_TIMEOUT = 0.25
PONTO_NEUTRO = (1593, 960)  # coluna de pedra do inventário: sem tooltip

pyautogui.PAUSE = 0.03  # o import do pyautogui também deixa o processo "DPI aware"

_janelas = [w for w in gw.getWindowsWithTitle(TITULO_JANELA) if w.title == TITULO_JANELA]
if not _janelas:
    raise SystemExit("Janela do RuneScape não encontrada. Certifique-se que o jogo está aberto.")
JANELA = _janelas[0]
LARGURA, ALTURA = visao.CAPTURA_CALIBRADA
# Área capturada (em coordenadas de tela). O dicionário é atualizado no lugar por
# atualizar_janela(), então quem guardou uma referência a ele vê sempre o valor atual.
REGIAO = {"left": JANELA.left, "top": JANELA.top, "width": LARGURA, "height": ALTURA}
_user32 = ctypes.windll.user32


def area_cliente():
    """(x, y, largura, altura) da área do jogo na tela, sem a moldura da janela."""
    global JANELA
    if not _user32.IsWindow(JANELA._hWnd):  # cliente fechado e aberto de novo
        janelas = [w for w in gw.getWindowsWithTitle(TITULO_JANELA) if w.title == TITULO_JANELA]
        if not janelas:
            return None
        JANELA = janelas[0]
    r = wintypes.RECT()
    p = wintypes.POINT(0, 0)
    if not _user32.GetClientRect(JANELA._hWnd, ctypes.byref(r)):
        return None
    if not _user32.ClientToScreen(JANELA._hWnd, ctypes.byref(p)):
        return None
    return p.x, p.y, r.right - r.left, r.bottom - r.top


def atualizar_janela():
    """Alinha a captura à área do jogo (a janela pode ter sido movida).

    Devolve None se estiver tudo certo, ou um texto explicando o problema (janela
    minimizada ou com tamanho diferente do calibrado). Custa microssegundos.
    """
    cliente = area_cliente()
    if cliente is None:
        return "não consegui ler a posição da janela do jogo"
    left, top, problema = visao.alinhar_captura(cliente)
    REGIAO["left"], REGIAO["top"] = left, top
    return problema


atualizar_janela()

_local = threading.local()
_mouse_parado_desde = time.time()
_pos_anterior = pyautogui.position()


# ---------- Log ----------
class Registro:
    def __init__(self, prefixo="sessao"):
        pasta = os.path.join(visao.BASE_DIR, "logs")
        os.makedirs(pasta, exist_ok=True)
        self.caminho = os.path.join(pasta, time.strftime(f"{prefixo}_%Y%m%d_%H%M%S.jsonl"))
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


# ---------- Tela ----------
def capturar(regiao=None):
    """Captura a janela inteira ou só uma região (x1, y1, x2, y2) dela."""
    if not hasattr(_local, "sct"):
        _local.sct = mss()
    if regiao is None:
        area = REGIAO
    else:
        x1, y1, x2, y2 = regiao
        area = {"left": REGIAO["left"] + x1, "top": REGIAO["top"] + y1,
                "width": x2 - x1, "height": y2 - y1}
    return cv2.cvtColor(np.array(_local.sct.grab(area)), cv2.COLOR_BGRA2BGR)


def caixa(pos, raio_x, raio_y=None):
    """Região (x1, y1, x2, y2) centrada em pos, sem sair da janela."""
    raio_y = raio_x if raio_y is None else raio_y
    x = min(max(int(pos[0]), raio_x), LARGURA - raio_x)
    y = min(max(int(pos[1]), raio_y), ALTURA - raio_y)
    return (x - raio_x, y - raio_y, x + raio_x, y + raio_y)


def jogo_em_foco():
    """Os bots leem a tela: só agem se o jogo for a janela ativa.

    Se a janela ativa for outro cliente do jogo (reaberto, ou um segundo cliente), o bot
    passa a seguir essa: captura e cliques valem sempre para a janela que está na frente.
    """
    global JANELA
    try:
        ativa = gw.getActiveWindow()
    except Exception:
        return False
    if ativa is None or ativa.title != TITULO_JANELA:
        return False
    if ativa._hWnd != JANELA._hWnd:
        JANELA = ativa
        atualizar_janela()
    return True


def trazer_para_frente():
    for tentativa in range(3):
        if jogo_em_foco():
            return True
        try:
            if tentativa == 1:
                pyautogui.press("alt")
            if tentativa == 2:
                JANELA.minimize()
                time.sleep(0.3)
                JANELA.restore()
                JANELA.maximize()
            else:
                JANELA.activate()
        except Exception:
            pass
        time.sleep(0.8)
    return jogo_em_foco()


# ---------- Mouse ----------
def usuario_mexeu_mouse():
    """True enquanto o usuário estiver mexendo o mouse (e por TEMPO_MOUSE_PARADO depois)."""
    global _mouse_parado_desde, _pos_anterior
    atual = pyautogui.position()
    if np.hypot(atual[0] - _pos_anterior[0], atual[1] - _pos_anterior[1]) > 2:
        _mouse_parado_desde = time.time()
        _pos_anterior = atual
    return time.time() - _mouse_parado_desde < TEMPO_MOUSE_PARADO


def mover(pos):
    global _pos_anterior
    pyautogui.moveTo(REGIAO["left"] + int(pos[0]), REGIAO["top"] + int(pos[1]))
    _pos_anterior = pyautogui.position()


def clicar(botao="left"):
    global _pos_anterior
    pyautogui.click(button=botao)
    _pos_anterior = pyautogui.position()


def clicar_minimapa(ponto, recalcular=None):
    """Clique no minimapa. Medido: o clique instantâneo do pyautogui é ignorado ali;
    precisa parar o mouse em cima e segurar o botão por um instante.

    `recalcular`: função que devolve o ponto atualizado. Com o personagem andando o
    minimapa desliza, então o ponto é refeito logo antes de apertar o botão (sem
    isso o clique caía ~1 tile fora do alvo).
    """
    global _pos_anterior
    mover(ponto)
    time.sleep(0.15)
    if recalcular is not None:
        novo = recalcular()
        if novo is not None and tuple(novo) != tuple(ponto):
            mover(novo)
    pyautogui.mouseDown()
    time.sleep(0.08)
    pyautogui.mouseUp()
    _pos_anterior = pyautogui.position()


def tecla(nome):
    pyautogui.press(nome)


def distancia(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


# ---------- Leituras com espera ----------
def esperar(condicao, timeout, intervalo=0.05):
    """Espera `condicao()` ficar verdadeira. Devolve (ok, tempo decorrido)."""
    inicio = time.perf_counter()
    while True:
        if condicao():
            return True, time.perf_counter() - inicio
        if time.perf_counter() - inicio > timeout:
            return False, time.perf_counter() - inicio
        time.sleep(intervalo)


def texto_acao(chave):
    """Score (0-1) do texto de ação atual contra o template `chave`."""
    return visao.score_texto(capturar(visao.FAIXA_TEXTO_ACAO), chave)


def esperar_texto(chaves, timeout=HOVER_TIMEOUT):
    """Depois de mover o mouse, espera o texto de ação virar um dos `chaves`.

    Devolve (chave ou None, latência, {chave: maior score}). Exige duas leituras
    seguidas iguais para não clicar com texto de passagem.
    """
    inicio = time.perf_counter()
    melhores = {c: 0.0 for c in chaves}
    anterior = None
    while True:
        dt = time.perf_counter() - inicio
        if dt > timeout:
            return None, dt, melhores
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
            return achou, dt, melhores
        anterior = achou
        time.sleep(0.01)


def clicar_conferindo(pos):
    """Clica e lê o X que o jogo desenha: 'alvo' (vermelho), 'chao' (amarelo) ou None."""
    regiao = caixa(pos, visao.CLIQUE_RAIO)
    antes = capturar(regiao)
    clicar()
    inicio = time.perf_counter()
    tipo = None
    while tipo is None and time.perf_counter() - inicio < CLIQUE_TIMEOUT:
        time.sleep(0.01)
        tipo = visao.tipo_clique(antes, capturar(regiao))
    return tipo
