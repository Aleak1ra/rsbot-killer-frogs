"""Volta ao jogo depois de uma desconexão (ex.: logout por inatividade).

O bot percebe que saiu do jogo quando a moldura do orbe de vida some (visao.diferenca_caixa_vida)
por mais de FORA_DO_JOGO_SEG com a janela em foco e do tamanho certo. Aí, a cada passo:
- grava uma foto de cada tela diferente em logs/telas/ (para calibrar os botões);
- se reconhecer uma tela de senha (moldes "_*.png" em templates/login/), não clica em nada e avisa;
- se reconhecer um botão (moldes "*.png" em templates/login/, ex.: "Try again", "Play Now",
  "Click here to play"), clica no centro dele.
Nunca digita nada. Sem moldes na pasta, só grava as fotos e avisa.
"""
import glob
import os
import time

import cv2
import numpy as np

import visao

PASTA_MOLDES = os.path.join(visao.TEMPLATE_DIR, "login")
PASTA_FOTOS = os.path.join(visao.BASE_DIR, "logs", "telas")
FORA_DO_JOGO_SEG = 10.0  # interface sumida por mais que isso = fora do jogo (não é transição)
LIMIAR_BOTAO = 0.90
ESPERA_DEPOIS_CLIQUE = 8.0  # s para a tela seguinte aparecer (e entre duas tentativas no mesmo botão)
MAX_CLIQUES_HORA = 12  # mais que isso numa hora: algo está errado, para de clicar e avisa
MAX_FOTOS = 30
DIFERENCA_FOTO = 6.0  # média da diferença (0-255) na miniatura para considerar outra tela


def carregar_moldes(pasta=None):
    """{nome: imagem cinza}. "_nome.png" = tela onde NÃO se clica (ex.: pedido de senha)."""
    moldes = {}
    for caminho in sorted(glob.glob(os.path.join(pasta or PASTA_MOLDES, "*.png"))):
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            moldes[os.path.splitext(os.path.basename(caminho))[0]] = img
    return moldes


def achar_moldes(img, moldes, limiar=LIMIAR_BOTAO):
    """[(nome, (x, y) do centro, nota)] dos moldes achados no quadro, do mais certo ao menos."""
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    achados = []
    for nome, molde in moldes.items():
        if cinza.shape[0] < molde.shape[0] or cinza.shape[1] < molde.shape[1]:
            continue
        res = cv2.matchTemplate(cinza, molde, cv2.TM_CCOEFF_NORMED)
        _, nota, _, (x, y) = cv2.minMaxLoc(res)
        if nota >= limiar:
            achados.append((nome, (x + molde.shape[1] // 2, y + molde.shape[0] // 2), float(nota)))
    return sorted(achados, key=lambda a: -a[2])


class Reconexao:
    """Estado da volta ao jogo. `clicar_em(pos)` move o mouse para `pos` (coordenadas da
    captura) e clica; `log(nome, **dados)` registra eventos; `agora()` dá o tempo (s)."""

    def __init__(self, clicar_em, log, agora=time.time, moldes=None, pasta_fotos=PASTA_FOTOS):
        self.clicar_em, self.log, self.agora = clicar_em, log, agora
        # moldes=None: lê templates/login/ e relê quando a pasta muda (moldes novos entram sem
        # reiniciar o bot, que perderia o estado da pausa por vida)
        self.recarregar = moldes is None
        self.assinatura = None
        self.moldes = {} if moldes is None else moldes
        self._recarregar()
        self.pasta_fotos = pasta_fotos
        self.fora_desde = None
        self.ultima_miniatura = None
        self.fotos = 0
        self.cliques = []  # instantes dos cliques (janela de 1 h)
        self.proximo_passo = 0.0
        self.avisou = set()

    def no_jogo(self):
        """Chamar a cada quadro com a interface no lugar."""
        if self.fora_desde is not None:
            self.log("de_volta_ao_jogo", seg=round(self.agora() - self.fora_desde))
            self.avisou.discard("fora")
        self.fora_desde = None

    def _recarregar(self):
        if not self.recarregar:
            return
        arquivos = sorted(glob.glob(os.path.join(PASTA_MOLDES, "*.png")))
        assinatura = tuple((a, os.path.getmtime(a)) for a in arquivos)
        if assinatura != self.assinatura:
            self.assinatura = assinatura
            self.moldes = carregar_moldes()
            if arquivos:
                self.log("moldes_login", nomes=sorted(self.moldes))

    def fora_do_jogo(self, img):
        """Chamar a cada quadro sem a interface no lugar (janela em foco e do tamanho certo).
        Devolve o que fez: None (ainda pode ser transição), "foto", "senha", "clique:<nome>",
        "limite" ou "esperando"."""
        agora = self.agora()
        if self.fora_desde is None:
            self.fora_desde = agora
        if agora - self.fora_desde < FORA_DO_JOGO_SEG or agora < self.proximo_passo:
            return None
        self._recarregar()
        if "fora" not in self.avisou:
            self.avisou.add("fora")
            self.log("fora_do_jogo", seg=round(agora - self.fora_desde), moldes=len(self.moldes))
            print("\n🔌 Fora do jogo (tela de login/desconectado?). "
                  + ("Tentando voltar pelos botões conhecidos." if self.moldes else
                     "Sem moldes dos botões ainda: só gravo as telas em logs/telas/."))
        acao = "foto" if self._foto(img) else "esperando"
        achados = achar_moldes(img, self.moldes)
        if any(nome.startswith("_") for nome, _, _ in achados):
            if "senha" not in self.avisou:
                self.avisou.add("senha")
                self.log("reconectar_senha", telas=[n for n, _, _ in achados])
                print("\n🔐 O jogo pede login com senha: não digito nada. Entre no jogo e o bot continua.")
            self.proximo_passo = agora + 5.0
            return "senha"
        botoes = [a for a in achados if not a[0].startswith("_")]
        if not botoes:
            self.proximo_passo = agora + 2.0
            return acao
        self.cliques = [t for t in self.cliques if agora - t < 3600]
        if len(self.cliques) >= MAX_CLIQUES_HORA:
            if "limite" not in self.avisou:
                self.avisou.add("limite")
                self.log("reconectar_limite", cliques=len(self.cliques))
                print(f"\n⚠️ Já tentei voltar ao jogo {len(self.cliques)} vezes na última hora: paro de clicar.")
            self.proximo_passo = agora + 60.0
            return "limite"
        nome, pos, nota = botoes[0]
        self.clicar_em(pos)
        self.cliques.append(agora)
        self.log("reconectar_clique", botao=nome, pos=list(pos), nota=round(nota, 3))
        print(f"🔌 Voltando ao jogo: cliquei em '{nome}'.")
        self.proximo_passo = agora + ESPERA_DEPOIS_CLIQUE
        return f"clique:{nome}"

    def _foto(self, img):
        """Grava o quadro se for uma tela diferente da última gravada."""
        if self.fotos >= MAX_FOTOS:
            return False
        mini = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img.shape[1] // 8, img.shape[0] // 8),
                          interpolation=cv2.INTER_AREA).astype(np.int16)
        if self.ultima_miniatura is not None and \
                np.abs(mini - self.ultima_miniatura).mean() < DIFERENCA_FOTO:
            return False
        self.ultima_miniatura = mini
        self.fotos += 1
        os.makedirs(self.pasta_fotos, exist_ok=True)
        nome = time.strftime("%Y%m%d_%H%M%S") + f"_{self.fotos:02d}.png"
        cv2.imwrite(os.path.join(self.pasta_fotos, nome), img)
        self.log("tela_gravada", arquivo=nome)
        return True
