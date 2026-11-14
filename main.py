"""Bot de Giant frog para o cliente oficial do Old School RuneScape.

Uso:  python main.py [--monstro sapo|hillgiant] [--amostras]
  --monstro   qual monstro caçar (padrão: sapo = Giant frog)
  --amostras  salva frames rotulados em amostras/ (para testar_visao.py)

Cada sessão grava um log de eventos em logs/sessao_*.jsonl; use
`python analisar_log.py` para ver os tempos medidos.
"""
import json
import os
import sys
import threading
import time

import cv2
import keyboard
import numpy as np
import pyautogui
import pygetwindow as gw
from mss import mss

import visao

# Perfil do monstro: python main.py --monstro hillgiant
NOME_PERFIL = sys.argv[sys.argv.index("--monstro") + 1] if "--monstro" in sys.argv else "sapo"
if NOME_PERFIL not in visao.PERFIS:
    sys.exit(f"Monstro desconhecido: {NOME_PERFIL}. Opções: {', '.join(visao.PERFIS)}")
PERFIL = visao.PERFIS[NOME_PERFIL]

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
MENU_TIMEOUT = 0.5

# Segurança: abaixo disso o bot para de atacar e avisa (ele não come nem foge)
VIDA_MINIMA = 0.40
VIDA_RETOMAR = 0.60

# Combate
ESPERA_INICIO_COMBATE = 5.0  # medido em 120 ataques: 1,5 s típico, 4,25 s no pior caso
ESPERA_LOOT = PERFIL["espera_loot"]  # tempo para o osso aparecer depois da barra zerar
RAIO_LOOT = 320  # px em volta do personagem

# Ossos
ESPERA_PEGAR_OSSO = 2.5  # medido em 48 coletas: 1,2 s típico, 1,6 s no pior caso
INTERVALO_ENTERRAR = 1.3  # o jogo ignora cliques de enterrar mais rápidos que ~2 ticks
MIN_OSSOS_ENTERRAR = 10  # fora de combate só enterra se acumular isso
# Onde o mouse descansa depois de clicar no inventário (coluna de pedra, sem tooltip)
PONTO_NEUTRO = (1593, 960)

pyautogui.PAUSE = 0.03

SALVAR_AMOSTRAS = "--amostras" in sys.argv
MAX_AMOSTRAS_POR_CLASSE = 8
MAX_FAIXAS = 120

ossos_ativos = threading.Event()
ossos_ativos.set()
mouse_parado_desde = time.time()
pos_anterior = pyautogui.position()

# Detectar janela do RuneScape
rs_window = [w for w in gw.getWindowsWithTitle(TITULO_JANELA) if w.title == TITULO_JANELA]
if not rs_window:
    raise Exception(
        "Janela do RuneScape não encontrada. Certifique-se que o jogo está aberto."
    )
rs_win = rs_window[0]

CAPTURE_REGION = {
    "left": rs_win.left,
    "top": rs_win.top,
    "width": rs_win.width,
    "height": rs_win.height,
}
CENTRO = (CAPTURE_REGION["width"] // 2, CAPTURE_REGION["height"] // 2)


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


def jogo_em_foco():
    """O bot lê a tela: só age se o jogo for a janela ativa."""
    try:
        ativa = gw.getActiveWindow()
    except Exception:
        return False
    return ativa is not None and ativa.title == TITULO_JANELA


def tempo_mouse_parado():
    global mouse_parado_desde, pos_anterior
    atual = pyautogui.position()
    if np.linalg.norm(np.subtract(atual, pos_anterior)) > 2:
        mouse_parado_desde = time.time()
        pos_anterior = atual
    return time.time() - mouse_parado_desde >= TEMPO_MOUSE_PARADO


def mover(pos):
    """Move o mouse para uma posição da captura, sem contar como movimento do usuário."""
    global pos_anterior
    pyautogui.moveTo(CAPTURE_REGION["left"] + int(pos[0]), CAPTURE_REGION["top"] + int(pos[1]))
    pos_anterior = pyautogui.position()


def clicar():
    global pos_anterior
    pyautogui.click()
    pos_anterior = pyautogui.position()


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
    clicar()
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
            if achou is None and s >= visao.TEXTO_THRESHOLD:
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


def salvar_menu(rec, ok):
    if not SALVAR_AMOSTRAS or _amostras.get("menu", 0) >= MAX_FAIXAS:
        return
    pasta = os.path.join(visao.BASE_DIR, "amostras", "menus")
    os.makedirs(pasta, exist_ok=True)
    n = _amostras["menu"] = _amostras.get("menu", 0) + 1
    nome = f"{time.strftime('%H%M%S')}_{n:03d}_{'ok' if ok else 'sem'}.png"
    cv2.imwrite(os.path.join(pasta, nome), rec)


def pegar_pelo_menu(pos):
    """Pilha com outro item por cima: abre o menu e clica na linha "Take Big bones"."""
    global pos_anterior
    pyautogui.click(button="right")
    pos_anterior = pyautogui.position()
    x, y = int(pos[0]), int(pos[1])
    regiao = (max(x - MENU_RAIO[0], 0), max(y - MENU_RAIO[1], 0),
              min(x + MENU_RAIO[0], CAPTURE_REGION["width"]),
              min(y + MENU_RAIO[1], CAPTURE_REGION["height"]))
    inicio = time.perf_counter()
    ponto, score, rec = None, 0.0, None
    while ponto is None and time.perf_counter() - inicio < MENU_TIMEOUT:
        time.sleep(0.03)
        rec = capturar(regiao)
        ponto, score = visao.achar_take_no_menu(rec)
    LOG.ev("menu", ok=ponto is not None, score=round(score, 3),
           lat=round(time.perf_counter() - inicio, 3))
    salvar_menu(rec, ponto is not None)
    if ponto is None:  # fecha o menu tirando o mouse de cima dele
        mover(PONTO_NEUTRO)
        return False
    mover((regiao[0] + ponto[0], regiao[1] + ponto[1]))
    time.sleep(0.03)
    clicar()
    return True


def pegar_osso(img, ossos):
    for osso in sorted(ossos, key=lambda o: distancia(CENTRO, o["pos"]))[:2]:
        pos = osso["pos"]
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
            if clique == "chao":
                LOG.ev("pegar_osso", ok=False, lat=0.0, dist=round(dist), clique=clique, via=via)
                return False
        else:
            # Outro item por cima do osso (ou um monstro em cima da pilha)
            via, clique = "menu", None
            if not pegar_pelo_menu(pos):
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


def atacar_sapo(img, sapos):
    candidatos = sorted(sapos, key=lambda s: distancia(CENTRO, s["pos"]))
    tentativas = 0
    for sapo in candidatos:
        if tentativas >= MAX_TENTATIVAS_SAPO:
            break
        pos = sapo["pos"]
        dist = distancia(CENTRO, pos)
        barra = visao.barra_vida_perto(img, pos, *PERFIL["barra"])
        if barra >= BARRA_SAPO_OCUPADO:
            LOG.ev("sapo_ocupado", dist=round(dist), barra=barra)
            continue
        tentativas += 1
        mover(pos)
        chave, lat, scores, faixa = esperar_texto([PERFIL["texto"]])
        salvar_faixa(faixa, f"sapo-{chave}", scores)
        LOG.ev("hover_sapo", texto=chave, lat=round(lat, 3), dist=round(dist), area=sapo["area"],
               caixa=sapo["caixa"], barra=barra, score=round(scores[PERFIL["texto"]], 3))
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
        if clique == "chao":  # o sapo pulou: tenta de novo no próximo ciclo, com posições novas
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


def esperar_loot(inicio):
    """O osso só aparece ~2 s depois da barra de vida zerar: espera por ele."""
    # Só olha em volta do personagem: ~10x mais rápido que varrer a janela toda
    area = (CENTRO[0] - RAIO_LOOT - 70, CENTRO[1] - RAIO_LOOT - 30,
            CENTRO[0] + RAIO_LOOT + 10, CENTRO[1] + RAIO_LOOT + 10)

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
    vida_baixa = False
    ultima_vida = 0.0
    LOG.ev("inicio", regiao=CAPTURE_REGION, perfil=NOME_PERFIL)
    while True:
        if not jogo_em_foco():
            if em_foco:
                LOG.ev("foco", ok=False)
                em_foco = False
            time.sleep(0.5)
            continue
        if not em_foco:
            LOG.ev("foco", ok=True)
            em_foco = True
        if not tempo_mouse_parado():
            time.sleep(DELAY_CICLO)
            continue

        t0 = time.perf_counter()
        img = capturar()
        t1 = time.perf_counter()
        estado = visao.estado_combate(img)

        # Segurança: vida do jogador pelo orbe
        vida = visao.vida_pct(img)
        if time.time() - ultima_vida >= 10:
            ultima_vida = time.time()
            LOG.ev("vida", pct=None if vida is None else round(vida, 3))
        if vida is not None:
            if not vida_baixa and vida < VIDA_MINIMA:
                vida_baixa = True
                LOG.ev("vida_baixa", pct=round(vida, 3))
                print(f"\n🚨 VIDA BAIXA (~{100 * vida:.0f}%): bot pausado. Coma ou saia da área!")
                try:
                    import winsound

                    for _ in range(3):
                        winsound.Beep(1200, 400)
                except Exception:
                    pass
            elif vida_baixa and vida >= VIDA_RETOMAR:
                vida_baixa = False
                LOG.ev("vida_ok", pct=round(vida, 3))
                print(f"💚 Vida recuperada (~{100 * vida:.0f}%): bot retomado.")
        if vida_baixa:
            time.sleep(0.5)
            continue
        coletar = ossos_ativos.is_set()
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
            if coletar and estado == "morto":
                if inventario and ENTERRO.pode():
                    ENTERRO.um(inventario, "loot")
                esperar_loot(t0)
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

        # Pegar ossos do chão
        t2 = t3 = time.perf_counter()
        if coletar:
            ossos = visao.detectar_ossos_chao(img)
            t3 = time.perf_counter()
            if ossos and pegar_osso(img, ossos):
                ossos_pegos += 1
                print(f"🦴 Ossos coletados do chão: {ossos_pegos}")
                continue

        # Atacar o sapo mais próximo
        sapos = visao.detectar_sapos(img, PERFIL)
        t4 = time.perf_counter()
        LOG.ev("ciclo", captura=round(t1 - t0, 4), rotulos=round(t3 - t2, 4),
               sapos=round(t4 - t3, 4), n_sapos=len(sapos))
        if sapos and atacar_sapo(img, sapos):
            sapos_atacados += 1
            em_luta = True
            inicio_luta = time.time()
            por_hora = 3600 * sapos_atacados / max(time.time() - inicio_sessao, 1)
            print(f"⚔️ {PERFIL['nome']} atacados: {sapos_atacados} (~{por_hora:.0f}/h)")
        time.sleep(DELAY_CICLO)


def loop_atalhos():
    while True:
        if keyboard.is_pressed("F8"):
            if ossos_ativos.is_set():
                ossos_ativos.clear()
                print("🩴 Coleta de ossos DESATIVADA!")
            else:
                ossos_ativos.set()
                print("🩴 Coleta de ossos ATIVADA!")
            time.sleep(0.5)
        time.sleep(0.05)


def executar_principal():
    try:
        loop_principal()
    except pyautogui.FailSafeException:
        print("\n🛑 Mouse no canto da tela: bot encerrado (failsafe).")
    except Exception:
        import traceback

        traceback.print_exc()
    os._exit(1)


# ========== Execução ==========
if __name__ == "__main__":
    print(f"🔁 Bot iniciado — alvo: {PERFIL['nome']}. Pressione CTRL+C para parar.")
    print("   (só age com o jogo em primeiro plano; F8 liga/desliga os ossos)")
    print(f"   log: {LOG.caminho}")
    try:
        threading.Thread(target=executar_principal, daemon=True).start()
        threading.Thread(target=loop_atalhos, daemon=True).start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Execução finalizada pelo usuário.")
