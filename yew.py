"""Corta Yew em Edgeville (2 árvores ao sul do banco) e deposita os logs.

Uso:  python yew.py
Requisitos: janela maximizada, bússola no norte, zoom no mínimo, machado EQUIPADO
(o depósito usa "Deposit inventory", que guarda tudo o que estiver no inventário).
Comece com o personagem em frente à cabine do banco (ponto BANCO).

O jogador é localizado pelo minimapa (ver visao.localizar). As posições abaixo são
em tiles, relativas ao ponto BANCO: x cresce para leste, y cresce para sul.
"""
import time
from collections import deque

import cv2
import numpy as np

import tela
import visao

# ========== Mapa ==========
BANCO = (0, 0)  # tile onde o personagem fica para usar a cabine
CABINE_NA_TELA = (42, -5)  # onde a cabine aparece, em px a partir do personagem
OBSERVATORIO = (-7, 10)  # tile de onde as duas árvores aparecem na tela
ARVORES = {
    "norte": (-8.0, 8.0),  # centro da copa, em tiles
    "sul": (-8.0, 19.7),
}
REFERENCIAS = visao.carregar_referencias({
    "banco": ("minimap_edgeville_banco.png", (0, 0)),
    "yew_n": ("minimap_edgeville_yew_n.png", (-7, 10)),
})

# ========== Interface do banco ==========
BOTAO_DEPOSITAR = (1013, 920)  # "Deposit inventory"
BOTAO_FECHAR = (1069, 62)
TITULO_BANCO = visao._carregar("bank_title_osrs.png")
REGIAO_TITULO = (680, 40, 890, 86)

# ========== Parâmetros ==========
INVENTARIO_CHEIO = 28
# Com as duas árvores cortadas só vale ir ao banco se o inventário for encher na
# próxima árvore (cada uma rende ~8-12 logs); com menos que isso, espera a árvore voltar.
MIN_LOGS_DEPOSITO_OCIOSO = 20
# Fração de "cor de copa" na posição da árvore. Medido: em pé 0,72-0,91;
# toco 0,00 (sul) e 0,24 (norte, onde o chão em volta é mais verde).
COPA_VIVA = 0.50
COPA_TOCO = 0.40
SEM_LOG_RECLICAR = 60.0  # s sem ganhar log com a árvore em pé -> clica nela de novo
# Animação do personagem (diferença média entre frames): cortando ~2,0-2,3; parado ~0,3
MOVIMENTO_PARADO = 1.0
# Medido: a yew volta ~126 s depois de cair (ainda toco aos 124 s, em pé aos 128 s).
# Uma árvore fora de vista só é dada como toco sem olhar se caiu há bem menos que isso;
# perto do retorno o bot vai até o ponto de observação e confere com os próprios olhos.
RESPAWN_MINIMO = 100.0
ESPERA_CAMINHADA = 45.0
# Onde procurar a árvore em volta do ponto previsto (px)
VARREDURA = [(0, 0), (0, -30), (-30, 0), (30, 0), (0, 30), (-30, -30), (30, -30), (0, -60)]

LOG = tela.Registro("yew")


# ========== Leituras ==========
def posicao():
    pos, score, ref = visao.localizar(tela.capturar(visao.MINIMAPA), REFERENCIAS)
    return pos


def logs_no_inventario(img=None):
    rec = tela.capturar(visao.INVENTARIO) if img is None else visao.recortar(img, visao.INVENTARIO)
    return visao.slots_ocupados(rec)


def banco_aberto():
    rec = tela.capturar(REGIAO_TITULO)
    res = cv2.matchTemplate(rec, TITULO_BANCO, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) >= 0.9


def ponto_arvore(nome, pos):
    copa = ARVORES[nome]
    return visao.ponto_tela((copa[0] - pos[0], copa[1] - pos[1]))


def estado_arvore(nome, img, pos):
    """'viva', 'toco' ou None (fora da tela / indefinido). Devolve também a fração de copa."""
    p = ponto_arvore(nome, pos)
    # Perto da borda de cima a previsão erra demais (a árvore norte vista da sul caía
    # em y=85 e era lida como toco mesmo em pé): ali o estado fica "desconhecido".
    if not (45 <= p[0] < 1835 and 130 <= p[1] < 1120):
        return None, None
    # o chat é transparente e deixa ver a árvore; inventário e minimapa não
    if any(x1 <= p[0] < x2 and y1 <= p[1] < y2 for x1, y1, x2, y2 in visao.AREAS_INTERFACE[:2]):
        return None, None
    f = visao.fracao_copa(img, p)
    if f >= COPA_VIVA:
        return "viva", f
    if f <= COPA_TOCO:
        return "toco", f
    return None, f


# ========== Movimento ==========
def parado(amostras=3, intervalo=0.35):
    """O personagem parou? (mesma posição em leituras seguidas)"""
    ultima = posicao()
    for _ in range(amostras - 1):
        time.sleep(intervalo)
        atual = posicao()
        if atual != ultima:
            return False
        ultima = atual
    return ultima is not None


def ir_para(alvo, tolerancia=0.6, timeout=ESPERA_CAMINHADA):
    """Anda até o tile `alvo` clicando no minimapa.

    Se o alvo estiver além do alcance do minimapa, o primeiro clique vai na direção
    dele; assim que o alvo entra no alcance (ainda andando), o clique final é dado
    no tile exato — o personagem não chega a parar no meio do caminho.
    """
    inicio = time.time()
    ultima, parado_n, cliques = None, 0, 0
    clique_final = False
    while time.time() - inicio < timeout:
        if not tela.jogo_em_foco():
            time.sleep(0.5)
            continue
        pos = posicao()
        if pos is None:
            time.sleep(0.4)
            continue
        falta = (alvo[0] - pos[0], alvo[1] - pos[1])
        if abs(falta[0]) <= tolerancia and abs(falta[1]) <= tolerancia:
            LOG.ev("ir_para", ok=True, alvo=alvo, dur=round(time.time() - inicio, 2), cliques=cliques)
            return True
        parado_n = parado_n + 1 if pos == ultima else 0
        no_alcance = np.hypot(*falta) * visao.MM_PX_TILE <= visao.MM_RAIO_CLIQUE
        if ultima is None or parado_n >= 2 or (no_alcance and not clique_final):
            def ponto_atual():
                p = posicao()
                if p is None:
                    return None
                return visao.ponto_minimapa((alvo[0] - p[0], alvo[1] - p[1]))

            tela.clicar_minimapa(visao.ponto_minimapa(falta), recalcular=ponto_atual)
            tela.mover(tela.PONTO_NEUTRO)
            cliques += 1
            parado_n = 0
            clique_final = no_alcance
            LOG.ev("clique_minimapa", de=pos, falta=falta, final=bool(no_alcance))
        ultima = pos
        time.sleep(0.45)
    LOG.ev("ir_para", ok=False, alvo=alvo, dur=round(time.time() - inicio, 2), cliques=cliques,
           pos=posicao())
    return False


def esperar_parar(timeout=25.0):
    inicio = time.time()
    time.sleep(0.8)
    while time.time() - inicio < timeout:
        if parado():
            return True
    return False


# ========== Ações ==========
def cortar(nome):
    """Clica na árvore (conferindo o texto "Chop down Yew tree") e espera chegar nela."""
    pos = posicao()
    if pos is None:
        return False
    base = ponto_arvore(nome, pos)
    for dx, dy in VARREDURA:
        p = (base[0] + dx, base[1] + dy)
        if not (20 <= p[0] < 1860 and 60 <= p[1] < 1150):
            continue
        tela.mover(p)
        chave, lat, scores = tela.esperar_texto(["yew"])
        if chave is None:
            continue
        clique = tela.clicar_conferindo(p)
        tela.mover(tela.PONTO_NEUTRO)
        LOG.ev("cortar", arvore=nome, ok=clique != "chao", clique=clique, desvio=(dx, dy),
               de=pos)
        if clique == "chao":
            return False
        esperar_parar()
        LOG.ev("chegou", arvore=nome, pos=posicao())
        return True
    tela.mover(tela.PONTO_NEUTRO)
    LOG.ev("cortar", arvore=nome, ok=False, clique=None, de=pos, previsto=base)
    return False


def depositar():
    """Vai ao banco, guarda o inventário e fecha a interface."""
    inicio = time.time()
    antes = logs_no_inventario()
    if not ir_para(BANCO, tolerancia=0.1):
        return False
    time.sleep(0.6)
    aberto = False
    for dx, dy in [(0, 0), (8, 5), (-8, -10), (15, -15), (0, -40), (0, 20)]:
        p = (visao.JOGADOR[0] + CABINE_NA_TELA[0] + dx, visao.JOGADOR[1] + CABINE_NA_TELA[1] + dy)
        tela.mover(p)
        chave, _, _ = tela.esperar_texto(["banco"], 0.35)
        if chave is None:
            continue
        tela.clicar_conferindo(p)
        tela.mover(tela.PONTO_NEUTRO)
        aberto, _ = tela.esperar(banco_aberto, 5.0, 0.1)
        break
    if not aberto:
        tela.mover(tela.PONTO_NEUTRO)
        LOG.ev("depositar", ok=False, motivo="cabine", pos=posicao())
        return False
    tela.mover(BOTAO_DEPOSITAR)
    time.sleep(0.15)
    tela.clicar()
    tela.mover(tela.PONTO_NEUTRO)
    vazio, _ = tela.esperar(lambda: logs_no_inventario() == 0, 3.0, 0.1)
    fechou, com_esc = fechar_banco()
    # Ao fechar o banco o jogo mostra por um instante o inventário antigo (ainda cheio):
    # espera a contagem voltar a zero e ficar estável antes de devolver o controle.
    tela.esperar(lambda: logs_no_inventario() == 0, 2.0, 0.1)
    time.sleep(0.3)
    LOG.ev("depositar", ok=vazio and fechou, logs=antes, dur=round(time.time() - inicio, 2),
           esc=com_esc, depois=logs_no_inventario())
    return vazio and fechou


def fechar_banco():
    """Fecha a interface do banco: Esc e, se a tecla não estiver configurada, o botão X.
    Devolve (fechou, fechou com Esc)."""
    tela.tecla("esc")
    fechou, _ = tela.esperar(lambda: not banco_aberto(), 1.5, 0.1)
    if fechou:
        return True, True
    tela.mover(BOTAO_FECHAR)
    time.sleep(0.1)
    tela.clicar()
    tela.mover(tela.PONTO_NEUTRO)
    fechou, _ = tela.esperar(lambda: not banco_aberto(), 2.0, 0.1)
    return fechou, False


# ========== Loop ==========
def principal():
    alvo = None  # árvore que está sendo cortada
    ultimo_log = time.time()
    # O inventário só é lido com o jogo em foco: antes disso a "tela" pode ser outra
    # janela (lida como 28 slots ocupados, o que mandava o bot ao banco à toa).
    n_anterior = None
    total_logs = 0
    depositados = 0
    inicio = time.time()
    ultimo_status = 0.0
    em_foco = True
    problema_janela = None
    leitura_anterior = None
    caiu_em = {}  # árvore -> instante em que virou toco
    movimentos = deque(maxlen=14)  # ~6 s de amostras da animação do personagem
    rec_anterior = None
    tocos_seguidos = 0
    # Só mexe no banco com a janela no tamanho calibrado (senão o X do banco está em outro lugar)
    if tela.jogo_em_foco() and tela.atualizar_janela() is None and banco_aberto():
        fechar_banco()
    LOG.ev("inicio", pos=posicao())
    print("🌲 Yew iniciado. Pressione CTRL+C para parar. Log:", LOG.caminho)
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
        # A janela pode ter sido movida: alinha a captura; tamanho errado -> não age
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
        pos, score, ref = visao.localizar(visao.recortar(img, visao.MINIMAPA), REFERENCIAS)
        if pos is None:
            LOG.ev("perdido", score=round(score, 3))
            time.sleep(1.0)
            continue
        # Inventário: só aceita a contagem quando duas leituras seguidas concordam
        # (ao fechar o banco o painel é redesenhado e uma leitura isolada dava 28)
        n_lido = visao.slots_ocupados(visao.recortar(img, visao.INVENTARIO))
        if n_anterior is None:  # primeira contagem: espera duas leituras iguais
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
        estados = {}
        copas = {}
        for nome in ARVORES:
            estados[nome], copas[nome] = estado_arvore(nome, img, pos)
            if estados[nome] == "viva":
                caiu_em.pop(nome, None)
            elif estados[nome] == "toco":
                caiu_em.setdefault(nome, agora)
            elif nome in caiu_em and agora - caiu_em[nome] < RESPAWN_MINIMO:
                estados[nome] = "toco"  # fora de vista, mas caiu há pouco: ainda não voltou

        if agora - ultimo_status >= 30:
            ultimo_status = agora
            por_hora = 3600 * total_logs / max(agora - inicio, 1)
            print(f"   {total_logs} logs cortados (~{por_hora:.0f}/h), {depositados} depositados, "
                  f"inventário {n}/28, árvores {estados}")
            LOG.ev("status", pos=pos, n=n, estados=estados, copas=copas, alvo=alvo,
                   mov=[round(m, 2) for m in movimentos])

        # 1) Inventário cheio -> banco
        if n >= INVENTARIO_CHEIO:
            alvo = None
            if depositar():
                depositados += n
                n_anterior = leitura_anterior = 0
                print(f"🏦 {n} logs depositados (total {depositados})")
            continue

        # 2) Cortando: confere se a árvore continua em pé
        if alvo is not None:
            # animação do personagem: diferença entre frames seguidos em volta dele
            x, y = visao.JOGADOR
            rec = cv2.cvtColor(img[y - 60:y + 40, x - 50:x + 50], cv2.COLOR_BGR2GRAY).astype(np.int16)
            if rec_anterior is not None:
                movimentos.append(float(np.abs(rec - rec_anterior).mean()))
            rec_anterior = rec
            ocioso = (len(movimentos) == movimentos.maxlen and max(movimentos) < MOVIMENTO_PARADO)
            # só dá a árvore como caída depois de duas leituras seguidas de toco
            tocos_seguidos = tocos_seguidos + 1 if estados[alvo] == "toco" else 0
            if tocos_seguidos >= 2:
                tocos_seguidos = 0
                LOG.ev("arvore_caiu", arvore=alvo, pos=pos)
                print(f"🪓 Yew {alvo} cortada — trocando de árvore")
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

        # 3) Escolhe uma árvore em pé (a mais próxima primeiro)
        ordem = sorted(ARVORES, key=lambda a: tela.distancia(ARVORES[a], pos))
        vivas = [a for a in ordem if estados[a] == "viva"]
        if vivas:
            if cortar(vivas[0]):
                alvo = vivas[0]
                ultimo_log = time.time()
                movimentos.clear()
                rec_anterior = None
                print(f"🌲 Cortando a yew {alvo}")
            continue

        # 4) Alguma árvore fora de vista -> vai até o ponto de onde as duas aparecem
        if any(estados[a] is None for a in ARVORES) and tela.distancia(pos, OBSERVATORIO) > 1.5:
            ir_para(OBSERVATORIO, tolerancia=1.0)
            continue

        # 5) As duas cortadas: deposita só se o inventário for encher na próxima árvore;
        #    senão espera no ponto de onde as duas aparecem, para ver a primeira que voltar
        if n >= MIN_LOGS_DEPOSITO_OCIOSO:
            if depositar():
                depositados += n
                n_anterior = leitura_anterior = 0
                print(f"🏦 {n} logs depositados enquanto as árvores voltam (total {depositados})")
            continue
        if tela.distancia(pos, OBSERVATORIO) > 1.5:
            ir_para(OBSERVATORIO, tolerancia=1.0)
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
