"""Peças comuns dos bots de corte (yew.py, magic.py): andar pelo minimapa, contar o
inventário e usar a interface do banco. Pressupõe o que visao.py pressupõe para a navegação:
janela calibrada, bússola no norte e zoom no mínimo."""
import time

import cv2
import numpy as np

import tela
import visao

# ========== Interface do banco ==========
BOTAO_DEPOSITAR = (1013, 920)  # "Deposit inventory"
BOTAO_ESVAZIAR = (967, 920)  # "Empty containers": esvazia no banco a cesta de logs (log/forestry basket)
BOTAO_FECHAR = (1069, 62)
TITULO_BANCO = visao._carregar("bank_title_osrs.png")
REGIAO_TITULO = (680, 40, 890, 86)

ESPERA_CAMINHADA = 45.0


def logs_no_inventario(img=None):
    rec = tela.capturar(visao.INVENTARIO) if img is None else visao.recortar(img, visao.INVENTARIO)
    return visao.slots_ocupados(rec)


def garantir_mochila(img):
    """Abre a aba da mochila se outra estiver aberta (com a aba de equipamento a contagem do
    inventário lia sempre 12 e o bot nunca ia ao banco). True se precisou clicar."""
    if visao.mochila_aberta(img):
        return False
    x1, y1, x2, y2 = visao.ABA_MOCHILA
    tela.mover(((x1 + x2) // 2, (y1 + y2) // 2))
    time.sleep(0.15)
    tela.clicar()
    tela.mover(tela.PONTO_NEUTRO)
    time.sleep(0.5)
    return True


def banco_aberto():
    rec = tela.capturar(REGIAO_TITULO)
    res = cv2.matchTemplate(rec, TITULO_BANCO, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) >= 0.9


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


def depositar_tudo(esvaziar_recipientes=False):
    """Com o banco aberto: "Deposit inventory", confere o inventário vazio e fecha.
    `esvaziar_recipientes`: antes, clica em "Empty containers" (a cesta de logs, mesmo equipada,
    vai direto para o banco; visto ao vivo: "You empty all of your containers into the bank").
    Devolve (vazio, fechou, fechou com Esc)."""
    if esvaziar_recipientes:
        tela.mover(BOTAO_ESVAZIAR)
        time.sleep(0.15)
        tela.clicar()
        time.sleep(0.6)
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
    return vazio, fechou, com_esc


class Navegador:
    """Localiza o jogador pelo minimapa (visao.localizar) e anda clicando nele."""

    def __init__(self, referencias, log, score_min=None):
        self.referencias, self.log, self.score_min = referencias, log, score_min

    def posicao(self):
        pos, _, _ = visao.localizar(tela.capturar(visao.MINIMAPA), self.referencias, self.score_min)
        return pos

    def parado(self, amostras=3, intervalo=0.35):
        """O personagem parou? (mesma posição em leituras seguidas)"""
        ultima = self.posicao()
        for _ in range(amostras - 1):
            time.sleep(intervalo)
            atual = self.posicao()
            if atual != ultima:
                return False
            ultima = atual
        return ultima is not None

    def ir_para(self, alvo, tolerancia=0.6, timeout=ESPERA_CAMINHADA):
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
            pos = self.posicao()
            if pos is None:
                time.sleep(0.4)
                continue
            falta = (alvo[0] - pos[0], alvo[1] - pos[1])
            if abs(falta[0]) <= tolerancia and abs(falta[1]) <= tolerancia:
                self.log.ev("ir_para", ok=True, alvo=alvo, dur=round(time.time() - inicio, 2), cliques=cliques)
                return True
            parado_n = parado_n + 1 if pos == ultima else 0
            no_alcance = np.hypot(*falta) * visao.MM_PX_TILE <= visao.MM_RAIO_CLIQUE
            if ultima is None or parado_n >= 2 or (no_alcance and not clique_final):
                def ponto_atual():
                    p = self.posicao()
                    if p is None:
                        return None
                    return visao.ponto_minimapa((alvo[0] - p[0], alvo[1] - p[1]))

                tela.clicar_minimapa(visao.ponto_minimapa(falta), recalcular=ponto_atual)
                tela.mover(tela.PONTO_NEUTRO)
                cliques += 1
                parado_n = 0
                clique_final = no_alcance
                self.log.ev("clique_minimapa", de=pos, falta=falta, final=bool(no_alcance))
            ultima = pos
            time.sleep(0.45)
        self.log.ev("ir_para", ok=False, alvo=alvo, dur=round(time.time() - inicio, 2), cliques=cliques,
                    pos=self.posicao())
        return False

    def esperar_parar(self, timeout=25.0):
        inicio = time.time()
        time.sleep(0.8)
        while time.time() - inicio < timeout:
            if self.parado():
                return True
        return False
