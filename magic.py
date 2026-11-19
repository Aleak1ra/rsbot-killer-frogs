"""Corta as Magic trees ao lado do banco do Tree Gnome Stronghold e deposita os logs.

Uso:  python magic.py   (ou dois cliques em "Iniciar magic.bat")
Requisitos: janela maximizada, bússola no norte, zoom no mínimo, machado EQUIPADO
(o depósito usa "Deposit inventory", que guarda tudo o que estiver no inventário).
Pode começar em qualquer ponto entre as árvores e o banco, no térreo ou no andar do banco.

O banco fica no andar de cima: o bot sobe pela escada ("Climb-up Staircase"), usa a cabine
e desce ("Climb-down Staircase"). No térreo o jogador é localizado pelo minimapa (ver
visao.localizar); as posições abaixo são em tiles do minimapa, relativas ao tile onde o
personagem chega ao descer a escada: x cresce para leste, y cresce para sul.
"""
import time
from collections import deque

import cv2
import numpy as np

import corte
import tela
import visao

# ========== Mapa ==========
ESCADA = (0, 0)  # tile do térreo onde o personagem chega ao descer (e de onde sobe)
# Onde o personagem fica para cortar cada árvore, e onde a copa dela aparece na tela a partir
# dali (px a partir do personagem).
ARVORES = {
    "oeste": {"ponto": (-6.0, 2.0), "copa": (-58, 7)},
    "leste": {"ponto": (21.4, 0.6), "copa": (67, -35)},
}
# Fotos do minimapa em pontos conhecidos (as árvores ficam a ~27 tiles uma da outra, mais do
# que uma foto alcança): pé da escada, meio do caminho e árvore leste.
REFERENCIAS = visao.carregar_referencias({
    "escada": ("minimap_gnome_escada.png", (0, 0)),
    "meio": ("minimap_gnome_meio.png", (13.4, 2.0)),
    "leste": ("minimap_gnome_leste.png", (21.4, 0.6)),
})
SCORE_MIN = 0.6  # os pontos amarelos dos gnomos passam por cima do minimapa e baixam a nota
# Onde passar o mouse (px a partir do personagem) para achar cada coisa; o texto de ação é
# conferido antes de qualquer clique.
# (o personagem para a até 1 tile do ponto: a varredura cobre os dois lados)
ESCADA_SUBIR = [(-55, 20), (-70, 10), (-45, 35), (-60, 0), (-85, 20), (-100, 20), (-130, 20), (-130, 0),
                (-160, 20), (-25, 20), (5, 20), (40, 20), (-55, 50), (-100, 50), (-55, -30), (-100, -30)]
CABINE = [(-33, -215), (-33, -245), (77, -245), (77, -215), (-33, -30), (77, -30), (-33, -90), (77, -90)]
ESCADA_DESCER = [(0, 372), (-15, 372), (20, 372), (0, 345), (0, 400), (0, 150), (0, 200), (0, 250),
                 (0, 300), (0, 450), (-15, 20), (0, 60), (0, 100)]

# ========== Parâmetros ==========
INVENTARIO_CHEIO = 28
# Copa da Magic tree sobre a grama do Stronghold: a grama é mais saturada (S > 200) e fica de
# fora. Medido numa caixa de 60 px na copa: em pé 0,85; toco 0,03.
COPA_HSV = ((28, 90, 20), (40, 195, 140))
COPA_VIVA = 0.35  # (de um tile vizinho ao ponto a copa em pé lê 0,50)
COPA_TOCO = 0.25
PERTO_DA_ARVORE = 1.5  # tiles do ponto da árvore: só dali o estado dela é lido
# Chegar "perto" basta (o clique na árvore ou na escada leva ao tile certo). Visto ao vivo: exigir
# o tile exato custava 26 s de cliques no minimapa, com a posição lida oscilando 1 tile.
CHEGADA = 1.0
SEM_LOG_RECLICAR = 150.0  # s sem ganhar log com a árvore em pé -> clica nela de novo
MOVIMENTO_PARADO = 1.0  # animação do personagem (como no yew.py): cortando ~2; parado ~0,3
RESPAWN_MINIMO = 100.0  # s: árvore que caiu há menos que isso ainda é toco, sem ir olhar
MINIMAPA_ESCURO = 0.5  # fração de pixels pretos no minimapa: 0,95 no andar do banco; 0,02 no térreo
ESPERA_ANDAR = 8.0  # s para o andar mudar depois do clique na escada
VARREDURA = [(0, 0), (0, -30), (-30, 0), (30, 0), (0, 30), (-30, -30), (30, -30), (0, -60)]

LOG = tela.Registro("magic")
NAV = corte.Navegador(REFERENCIAS, LOG, SCORE_MIN)
posicao, ir_para, esperar_parar = NAV.posicao, NAV.ir_para, NAV.esperar_parar


# ========== Leituras ==========
def no_andar_de_cima(img=None):
    """O personagem está no andar do banco? Lá o minimapa é quase todo preto."""
    mm = tela.capturar(visao.MINIMAPA) if img is None else visao.recortar(img, visao.MINIMAPA)
    cx, cy = visao.MM_CENTRO
    yy, xx = np.ogrid[:mm.shape[0], :mm.shape[1]]
    circulo = (xx - cx) ** 2 + (yy - cy) ** 2 <= 70 ** 2
    return float((mm.max(axis=2)[circulo] < 40).mean()) >= MINIMAPA_ESCURO


def ponto_copa(nome, pos):
    """Onde a copa da árvore aparece na tela com o personagem em `pos` (perto do ponto dela)."""
    arvore = ARVORES[nome]
    dx, dy = arvore["ponto"][0] - pos[0], arvore["ponto"][1] - pos[1]
    return (int(visao.JOGADOR[0] + arvore["copa"][0] + dx * visao.PX_TILE),
            int(visao.JOGADOR[1] + arvore["copa"][1] + dy * visao.PX_TILE))


def estado_arvore(nome, img, pos):
    """'viva', 'toco' ou None (longe demais para ver / indefinido), e a fração de copa."""
    if tela.distancia(pos, ARVORES[nome]["ponto"]) > PERTO_DA_ARVORE:
        return None, None
    f = visao.fracao_copa(img, ponto_copa(nome, pos), raio=30, faixa=COPA_HSV)
    if f >= COPA_VIVA:
        return "viva", f
    if f <= COPA_TOCO:
        return "toco", f
    return None, f


# ========== Ações ==========
def passar_e_clicar(chave, pontos, evento):
    """Passa o mouse pelos `pontos` (px a partir do personagem) até o texto de ação ser `chave`
    e clica ali. Devolve o tipo do clique ('alvo'/'chao'/None) ou False se o texto não apareceu."""
    for dx, dy in pontos:
        p = (visao.JOGADOR[0] + dx, visao.JOGADOR[1] + dy)
        if not (20 <= p[0] < 1560 and 60 <= p[1] < 1150):
            continue  # fora da área do jogo (ou em cima do inventário)
        tela.mover(p)
        achou, *_ = tela.esperar_texto([chave], 0.35)
        if achou is None:
            continue
        clique = tela.clicar_conferindo(p)
        tela.mover(tela.PONTO_NEUTRO)
        LOG.ev(evento, ok=clique != "chao", clique=clique, desvio=(dx, dy))
        return clique
    tela.mover(tela.PONTO_NEUTRO)
    LOG.ev(evento, ok=False, clique=None)
    return False


def cortar(nome):
    """Clica na árvore (conferindo o texto "Chop down Magic tree") e espera chegar nela."""
    pos = posicao()
    if pos is None:
        return False
    base = ponto_copa(nome, pos)
    pontos = [(base[0] + dx - visao.JOGADOR[0], base[1] + dy - visao.JOGADOR[1]) for dx, dy in VARREDURA]
    clique = passar_e_clicar("magic", pontos, "cortar")
    if clique is False or clique == "chao":
        return False
    esperar_parar()
    LOG.ev("chegou", arvore=nome, pos=posicao())
    return True


def subir():
    """Do térreo: vai até a escada e sobe. True se chegou ao andar do banco."""
    if not ir_para(ESCADA, tolerancia=CHEGADA):
        return False
    time.sleep(0.6)
    if passar_e_clicar("escada_subir", ESCADA_SUBIR, "subir") in (False, "chao"):
        return False
    ok, dt = tela.esperar(no_andar_de_cima, ESPERA_ANDAR, 0.2)
    LOG.ev("andar", cima=True, ok=ok, dur=round(dt, 2))
    time.sleep(0.8)  # a tela ainda está sendo redesenhada
    return ok


def descer():
    """Do andar do banco: clica na escada e desce. True se chegou ao térreo."""
    if passar_e_clicar("escada_descer", ESCADA_DESCER, "descer") in (False, "chao"):
        return False
    ok, dt = tela.esperar(lambda: not no_andar_de_cima(), ESPERA_ANDAR + 6.0, 0.2)
    LOG.ev("andar", cima=False, ok=ok, dur=round(dt, 2))
    time.sleep(0.8)
    return ok


def depositar_em_cima():
    """No andar do banco: abre a cabine, guarda o inventário e fecha a interface."""
    inicio = time.time()
    antes = corte.logs_no_inventario()
    if not corte.banco_aberto():
        if passar_e_clicar("banco", CABINE, "cabine") in (False, "chao"):
            LOG.ev("depositar", ok=False, motivo="cabine")
            return False
        aberto, _ = tela.esperar(corte.banco_aberto, 8.0, 0.1)
        if not aberto:
            LOG.ev("depositar", ok=False, motivo="banco_nao_abriu")
            return False
    vazio, fechou, com_esc = corte.depositar_tudo(esvaziar_recipientes=True)
    LOG.ev("depositar", ok=vazio and fechou, logs=antes, dur=round(time.time() - inicio, 2),
           esc=com_esc, depois=corte.logs_no_inventario())
    return vazio and fechou


# ========== Loop ==========
def principal():
    alvo = None  # árvore que está sendo cortada
    ultimo_log = time.time()
    n_anterior = None
    leitura_anterior = None
    total_logs = 0
    depositados = 0
    inicio = time.time()
    ultimo_status = 0.0
    em_foco = True
    problema_janela = None
    caiu_em = {}  # árvore -> instante em que virou toco
    movimentos = deque(maxlen=14)  # ~6 s de amostras da animação do personagem
    rec_anterior = None
    tocos_seguidos = 0
    falhas_cima = 0
    ja_depositou = False  # nesta subida ao banco
    if tela.jogo_em_foco() and tela.atualizar_janela() is None and corte.banco_aberto():
        corte.fechar_banco()
    LOG.ev("inicio", pos=posicao(), cima=no_andar_de_cima())
    print("🌳 Magic tree iniciado. Pressione CTRL+C para parar. Log:", LOG.caminho)
    while True:
        if not tela.jogo_em_foco():
            if em_foco:
                LOG.ev("foco", ok=False)
                em_foco = False
            time.sleep(0.5)
            continue
        if not em_foco:
            LOG.ev("foco", ok=True)
            em_foco = True
            time.sleep(0.5)  # logo após o Alt+Tab a tela ainda é a transição
            continue
        problema = tela.atualizar_janela()
        if problema != problema_janela:
            LOG.ev("janela", ok=problema is None, problema=problema)
            print(f"\n⚠️ {problema}. Bot pausado até a janela voltar ao normal." if problema
                  else "✅ Janela do jogo ok: bot retomado.")
            problema_janela = problema
            if not problema:
                time.sleep(0.5)
                continue
        if problema:
            time.sleep(1.0)
            continue
        if tela.usuario_mexeu_mouse():
            time.sleep(0.1)
            continue

        img = tela.capturar()
        if not corte.banco_aberto() and corte.garantir_mochila(img):
            LOG.ev("aba_mochila")
            n_anterior = leitura_anterior = None  # a contagem lida com outra aba não vale
            continue
        # Inventário: só aceita a contagem quando duas leituras seguidas concordam
        n_lido = visao.slots_ocupados(visao.recortar(img, visao.INVENTARIO))
        if n_anterior is None:
            if n_lido == leitura_anterior:
                n_anterior = n_lido
            leitura_anterior = n_lido
            if n_anterior is None:
                time.sleep(0.2)
                continue
        n = n_lido if n_lido == leitura_anterior else n_anterior
        leitura_anterior = n_lido
        if n > n_anterior:
            total_logs += n - n_anterior
            ultimo_log = time.time()
            LOG.ev("log", n=n, arvore=alvo)
        n_anterior = n
        agora = time.time()

        # A) No andar do banco: deposita (se houver o que guardar) e desce
        if no_andar_de_cima(img):
            alvo = None
            if not ja_depositou and depositar_em_cima():
                depositados += n
                n_anterior = leitura_anterior = 0
                ja_depositou = True
                print(f"🏦 Cesta esvaziada e {n} logs do inventário depositados (total do inventário: {depositados})")
            if ja_depositou or falhas_cima >= 3:
                if descer():
                    falhas_cima = 0
                    ja_depositou = False
                    continue
            falhas_cima += 1
            LOG.ev("cima_falhou", n=n, falhas=falhas_cima)
            if falhas_cima == 5:
                print("\n⚠️ Não consegui usar o banco/escada no andar de cima. Confira a câmera "
                      "(bússola no norte, zoom no mínimo); continuo tentando.")
                tela.bip("erro")
            time.sleep(1.5)
            continue
        falhas_cima = 0

        pos, score, _ = visao.localizar(visao.recortar(img, visao.MINIMAPA), REFERENCIAS, SCORE_MIN)
        if pos is None:
            LOG.ev("perdido", score=round(score, 3))
            time.sleep(1.0)
            continue

        estados, copas = {}, {}
        for nome in ARVORES:
            estados[nome], copas[nome] = estado_arvore(nome, img, pos)
            if estados[nome] == "viva":
                caiu_em.pop(nome, None)
            elif estados[nome] is None and nome in caiu_em and agora - caiu_em[nome] < RESPAWN_MINIMO:
                estados[nome] = "toco"  # longe, mas caiu há pouco: ainda não voltou

        if agora - ultimo_status >= 30:
            ultimo_status = agora
            por_hora = 3600 * total_logs / max(agora - inicio, 1)
            print(f"   {total_logs} logs cortados (~{por_hora:.0f}/h), {depositados} depositados, "
                  f"inventário {n}/28, árvores {estados}")
            LOG.ev("status", pos=pos, n=n, estados=estados, copas=copas, alvo=alvo,
                   mov=[round(m, 2) for m in movimentos])

        # B) Inventário cheio (ou depósito decidido com as árvores cortadas) -> sobe ao banco
        if n >= INVENTARIO_CHEIO:
            alvo = None
            if not subir():
                time.sleep(1.0)
            continue
        # C) Cortando: confere se a árvore continua em pé
        if alvo is not None:
            x, y = visao.JOGADOR
            rec = cv2.cvtColor(img[y - 60:y + 40, x - 50:x + 50], cv2.COLOR_BGR2GRAY).astype(np.int16)
            if rec_anterior is not None:
                movimentos.append(float(np.abs(rec - rec_anterior).mean()))
            rec_anterior = rec
            ocioso = len(movimentos) == movimentos.maxlen and max(movimentos) < MOVIMENTO_PARADO
            tocos_seguidos = tocos_seguidos + 1 if estados[alvo] == "toco" else 0
            if tocos_seguidos >= 2:
                tocos_seguidos = 0
                LOG.ev("arvore_caiu", arvore=alvo, pos=pos, copa=copas[alvo])
                print(f"🪓 Magic tree {alvo} cortada")
                caiu_em[alvo] = agora
                alvo = None
            elif ocioso or agora - ultimo_log > SEM_LOG_RECLICAR:
                LOG.ev("reclicar", arvore=alvo, estado=estados[alvo], ocioso=ocioso,
                       mov=round(max(movimentos), 2) if movimentos else None)
                ultimo_log = time.time()
                movimentos.clear()
                rec_anterior = None
                if not cortar(alvo):
                    alvo = None
                continue
            else:
                time.sleep(0.4)
                continue

        # D) Escolhe a árvore: a em pé à vista; senão a mais próxima que pode estar em pé
        ordem = sorted(ARVORES, key=lambda a: tela.distancia(ARVORES[a]["ponto"], pos))
        vivas = [a for a in ordem if estados[a] == "viva"]
        if vivas:
            if cortar(vivas[0]):
                alvo = vivas[0]
                ultimo_log = time.time()
                movimentos.clear()
                rec_anterior = None
                tocos_seguidos = 0
                print(f"🌳 Cortando a Magic tree {alvo}")
            else:
                time.sleep(0.5)
            continue
        for nome in ordem:
            if estados[nome] == "toco" and tela.distancia(pos, ARVORES[nome]["ponto"]) <= PERTO_DA_ARVORE:
                caiu_em.setdefault(nome, agora)
        incertas = [a for a in ordem if estados[a] is None]
        if incertas:
            if tela.distancia(pos, ARVORES[incertas[0]]["ponto"]) > PERTO_DA_ARVORE:
                ir_para(ARVORES[incertas[0]]["ponto"], tolerancia=CHEGADA)
            else:
                time.sleep(0.5)  # no ponto, mas a leitura da copa ficou no meio-termo
            continue

        # E) Todas cortadas: espera perto da que caiu primeiro (o banco é só com o inventário cheio)
        primeira = min(ARVORES, key=lambda a: caiu_em.get(a, 0))
        if tela.distancia(pos, ARVORES[primeira]["ponto"]) > PERTO_DA_ARVORE:
            ir_para(ARVORES[primeira]["ponto"], tolerancia=CHEGADA)
            continue
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        principal()
    except KeyboardInterrupt:
        print("\n🛑 Execução finalizada pelo usuário.")
    except Exception as e:
        if type(e).__name__ == "FailSafeException":
            print("\n🛑 Mouse no canto da tela: bot encerrado (failsafe).")
        else:
            raise
