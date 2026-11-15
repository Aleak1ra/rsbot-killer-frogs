"""Detecção por imagem: funções puras (recebem um frame BGR, devolvem o que foi visto).

Ficam separadas do bot para poderem ser testadas offline com frames salvos
(ver testar_visao.py).
"""
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")


def _carregar(nome, cinza=False):
    caminho = os.path.join(TEMPLATE_DIR, nome)
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE if cinza else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Template não encontrado: {caminho}")
    return img


# Texto de ação do cliente oficial (canto superior esquerdo), como máscara das
# cores do texto: independe do cenário que estiver atrás das letras.
# "giant" = "Attack Giant frog (level-13)", "hillgiant" = "Attack Hill Giant (level-28)",
# "take" = "Take Big bones".
TEXTOS = {
    "giant": _carregar("text_mask_giant_osrs.png", cinza=True),
    "hillgiant": _carregar("text_mask_hillgiant_osrs.png", cinza=True),
    "take": _carregar("text_mask_take_osrs.png", cinza=True),
    "banco": _carregar("text_mask_bank_booth_osrs.png", cinza=True),  # "Bank Bank booth"
    "yew": _carregar("text_mask_yew_osrs.png", cinza=True),  # "Chop down Yew tree"
}
ICONE_OSSO = _carregar("inventory_bone_osrs.png")
# Linha "Take Big bones" do menu de botão direito (média de 20 menus)
MENU_TAKE = _carregar("menu_take_big_bones_osrs.png", cinza=True)
# Rótulo branco "Big bones" desenhado sobre o item no chão (média de 15 ocorrências)
ROTULO_BLUR = (5, 5)
ROTULO_OSSO = cv2.GaussianBlur(_carregar("big_bones_label_osrs.png", cinza=True), ROTULO_BLUR, 0)

# ---------- Parâmetros calibráveis ----------
FROG_LOWER = np.array([59, 225, 45])
FROG_UPPER = np.array([63, 255, 80])
SAPO_AREA_MIN = 400
SAPO_AREA_MAX = 5000

# Limiares calibrados com amostras confirmadas pelo próprio jogo:
#   texto:  corretos >= 0,997   incorretos <= 0,70
#   rótulo: corretos >= 0,90    falsos     <= 0,62
TEXTO_THRESHOLD = 0.88
ROTULO_THRESHOLD = 0.76
ROTULO_BRILHO_MIN = 160
ROTULO_OFFSET_OSSO = (63, 25)  # do canto do rótulo até o osso no chão
ICONE_THRESHOLD = 0.85

# Regiões fixas da interface (x1, y1, x2, y2) dentro da captura da janela
FAIXA_TEXTO_ACAO = (0, 25, 720, 80)
PAINEL_OPONENTE = (17, 94, 194, 116)
INVENTARIO = (1570, 780, 1880, 1125)
# Áreas onde não existe mundo 3D: candidatos ali são descartados
AREAS_INTERFACE = [
    (1570, 740, 1940, 1170),  # inventário e abas
    (1600, 30, 1940, 290),  # minimapa e orbes
    (0, 950, 660, 1170),  # chat
    (0, 0, 1940, 40),  # barra de título
    (10, 60, 200, 125),  # painel do oponente
]


# ---------- Perfis de monstro ----------
# cor: faixa HSV do corpo; area: tamanho aceito do contorno (px²); fechar: une pedaços
# separados por objetos na frente (0 = não usa); barra: caixa onde procurar a barra de
# vida de quem já está em luta (meia largura, acima, abaixo do centro);
# raio_mira / area_mira: recorte e contorno mínimo ao corrigir a mira antes do clique;
# espera_loot: tempo máximo (s) para o osso aparecer depois da barra de vida zerar;
# loot_menu: o monstro larga vários itens na mesma pilha, então o osso é pego direto
# pelo menu de botão direito (sem conferir antes o texto de ação).
PERFIS = {
    "sapo": {
        "nome": "Giant frog",
        "texto": "giant",
        "cor": (FROG_LOWER, FROG_UPPER),
        "area": (SAPO_AREA_MIN, SAPO_AREA_MAX),
        "fechar": 0,
        "barra": (45, 85, 10),
        "raio_mira": 70,
        "area_mira": 150,
        "espera_loot": 3.0,  # medido: aparece em 1,8-2,0 s
        "loot_menu": False,
    },
    # Pele do Hill Giant: H 13-16, S ~127. O chão da caverna fica em H 19-20, S 153-185.
    "hillgiant": {
        "nome": "Hill Giant",
        "texto": "hillgiant",
        "cor": (np.array([12, 105, 60]), np.array([17, 140, 235])),
        "area": (500, 9000),
        "fechar": 5,
        "barra": (75, 120, 45),
        "raio_mira": 110,
        "area_mira": 300,
        "espera_loot": 4.5,  # medido: aparece em ~3,0 s
        "loot_menu": True,
    },
}


def mascara_monstro(img, perfil):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, perfil["cor"][0], perfil["cor"][1])
    if perfil["fechar"]:
        k = np.ones((perfil["fechar"], perfil["fechar"]), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def recortar(img, regiao):
    x1, y1, x2, y2 = regiao
    return img[y1:y2, x1:x2]


def na_interface(pos):
    x, y = pos
    return any(x1 <= x < x2 and y1 <= y < y2 for x1, y1, x2, y2 in AREAS_INTERFACE)


def picos(res, threshold, raio_x, raio_y, limite=30):
    """Máximos locais distintos de um resultado de matchTemplate: [(x, y, score)]."""
    res = res.copy()
    pontos = []
    while len(pontos) < limite:
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < threshold:
            break
        pontos.append((max_loc[0], max_loc[1], float(max_val)))
        cv2.rectangle(
            res,
            (max_loc[0] - raio_x, max_loc[1] - raio_y),
            (max_loc[0] + raio_x, max_loc[1] + raio_y),
            -1.0,
            -1,
        )
    return pontos


# ---------- Texto de ação ----------
def mascara_texto(img):
    """Pixels com as cores do texto de ação: branco, amarelo (NPC), verde-limão (nível),
    laranja (item) e ciano (objeto do cenário)."""
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    branco = (r > 200) & (g > 200) & (b > 200)
    amarelo = (r > 200) & (g > 200) & (b < 90)
    verde = (r < 90) & (g > 200) & (b < 90)
    laranja = (r > 200) & (g > 100) & (g < 190) & (b < 110)
    ciano = (r < 90) & (g > 200) & (b > 200)
    m = (branco | amarelo | verde | laranja | ciano).astype(np.uint8) * 255
    return cv2.GaussianBlur(m, (3, 3), 0)


def score_texto(faixa, chave):
    """Semelhança (0-1) entre a faixa do texto de ação e o template `chave`."""
    res = cv2.matchTemplate(mascara_texto(faixa), TEXTOS[chave], cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def faixa_texto(img):
    return recortar(img, FAIXA_TEXTO_ACAO)


# Só a palavra "Take" do texto de ação (do molde "Take Big bones"): diz se há um item do
# chão sob o cursor, qualquer que seja. Medido nas 353 faixas salvas: "Take ..." >= 0,998;
# "Attack ..." e "Walk here" <= 0,70.
TAKE_ACAO = TEXTOS["take"][:, 1:41].copy()
TAKE_ACAO_THRESHOLD = 0.9


def tem_take_acao(faixa):
    """O texto de ação (recorte FAIXA_TEXTO_ACAO) começa com "Take"? (item sob o cursor)"""
    return float(cv2.matchTemplate(mascara_texto(faixa), TAKE_ACAO, cv2.TM_CCOEFF_NORMED).max()) \
        >= TAKE_ACAO_THRESHOLD


# ---------- Menu de botão direito ----------
# Medido em 20 menus: a linha "Take Big bones" dá >= 0,97; "Examine Big bones" <= 0,86.
MENU_THRESHOLD = 0.91


def achar_take_no_menu(img_regiao):
    """Procura a linha "Take Big bones" num menu aberto.

    Devolve ((x, y) do centro da linha dentro do recorte, score) ou (None, score).
    """
    tpl = MENU_TAKE
    res = cv2.matchTemplate(mascara_texto(img_regiao), tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < MENU_THRESHOLD:
        return None, float(score)
    h, w = tpl.shape[:2]
    return (loc[0] + w // 2, loc[1] + h // 2), float(score)


# ---------- Vida do jogador ----------
REGIAO_VIDA = (1618, 114, 1644, 132)  # número ao lado do orbe de Hitpoints
# Caixa marrom onde o número fica. Antes de ler a cor, o bot confere que a caixa está
# exatamente ali, comparando só a moldura dela: a faixa dos dígitos fica de fora, e o
# orbe (x >= 1645) também, porque ele esvazia quando a vida cai e muda de cor com
# veneno; com ele na comparação, a leitura sumia justamente com a vida baixa.
# Medido em 313 imagens: caixa no lugar difere 0,0 (uma imagem 0,05). Em 685 milhões
# de posições de 316 imagens, a menor diferença fora dela foi 1,94 (a própria caixa
# 1 px ao lado), 2,07 (caixa do orbe de especial, mesmo desenho) e 5,8 no cenário.
CAIXA_VIDA = (1610, 108, 1648, 136)
CAIXA_VIDA_REF = _carregar("orbe_vida_caixa.png").astype(np.float32)
CAIXA_VIDA_MASCARA = _carregar("orbe_vida_mascara.png", cinza=True) > 127
CAIXA_VIDA_LIMIAR = 1.5
VIDA_PIXELS_MIN = 8  # o número tem 34-59 pixels coloridos com 2 dígitos
VIDA_MATIZ_ESPALHAMENTO_MAX = 8  # o número tem uma cor só (medido: até 4)


def diferenca_caixa_vida(img):
    """Quanto o recorte difere da caixa do orbe de vida (0 = idêntico)."""
    rec = recortar(img, CAIXA_VIDA)
    if rec.shape != CAIXA_VIDA_REF.shape:
        return float("inf")
    return float(np.abs(rec.astype(np.float32) - CAIXA_VIDA_REF)[CAIXA_VIDA_MASCARA].mean())


def vida_pct(img):
    """Fração de vida do jogador (0-1), lida pela COR do número do orbe.

    O jogo pinta o número num degradê contínuo: verde (0,255,0) com vida cheia,
    amarelo (255,255,0) em 50% e vermelho (255,0,0) em 0%. Em HSV isso é linear:
    vida = matiz / 60 (matiz do OpenCV). Conferido: 72/74 -> matiz 58 -> 97%.

    Devolve None se não tiver certeza de estar olhando o orbe: caixa fora do lugar
    (janela deslocada, algo por cima), número ausente ou com mais de uma cor.
    """
    if diferenca_caixa_vida(img) > CAIXA_VIDA_LIMIAR:
        return None
    hsv = cv2.cvtColor(recortar(img, REGIAO_VIDA), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    cor = hsv[(hsv[:, 1] > 180) & (hsv[:, 2] > 150)]
    if len(cor) < VIDA_PIXELS_MIN:
        return None
    matiz = cor[:, 0].astype(np.int16)
    matiz[matiz > 150] = 0  # vermelho "dá a volta" no círculo de matiz
    p10, p50, p90 = np.percentile(matiz, [10, 50, 90])
    if p90 - p10 > VIDA_MATIZ_ESPALHAMENTO_MAX:
        return None
    return float(min(p50 / 60.0, 1.0))


class VigiaVida:
    """Decide quando pausar e retomar pela vida, sem reagir a uma leitura isolada.

    Pausa depois de `confirmacoes` leituras seguidas abaixo de `minima` e retoma depois
    de `confirmacoes` leituras seguidas em `retomar` ou mais. Uma leitura no meio do
    caminho zera a contagem; None (orbe não confirmado) não conta nem zera.
    """

    def __init__(self, minima=0.40, retomar=0.60, confirmacoes=3):
        self.minima, self.retomar, self.confirmacoes = minima, retomar, confirmacoes
        self.pausado = False
        self.seguidas = 0  # leituras seguidas pedindo a troca de estado

    def ler(self, vida):
        """Registra uma leitura; devolve "pausar", "retomar" ou None."""
        if vida is None:
            return None
        pede_troca = vida >= self.retomar if self.pausado else vida < self.minima
        self.seguidas = self.seguidas + 1 if pede_troca else 0
        if self.seguidas < self.confirmacoes:
            return None
        self.pausado = not self.pausado
        self.seguidas = 0
        return "pausar" if self.pausado else "retomar"

    def descartar(self):
        """Esquece leituras pendentes que não se confirmaram (leitura isolada)."""
        self.seguidas = 0


# ---------- Geometria da janela ----------
# Calibração: janela maximizada num monitor 1920x1200 com escala de 125%. A captura
# tem 1938x1168 (inclui as bordas invisíveis do Windows e a barra de título) e a área
# do jogo começa em (9, 38) dentro dela, com 1920x1121. Todas as posições fixas deste
# arquivo valem nesse sistema de coordenadas.
CAPTURA_CALIBRADA = (1938, 1168)
CLIENTE_CALIBRADO = (9, 38, 1920, 1121)  # x, y, largura, altura


def alinhar_captura(cliente):
    """Onde capturar para que as posições calibradas valham, dada a área do jogo na tela.

    `cliente` = (x, y, largura, altura) da área do jogo em coordenadas de tela.
    Devolve (left, top, problema): a janela pode estar em qualquer lugar; `problema`
    é None se o tamanho bater com a calibração, ou um texto explicando o que mudou.
    """
    x, y, w, h = cliente
    cx, cy, cw, ch = CLIENTE_CALIBRADO
    if w <= 0 or h <= 0:
        return x - cx, y - cy, "janela do jogo minimizada"
    if (w, h) != (cw, ch):
        return x - cx, y - cy, (f"área do jogo com {w}x{h} (calibrada para {cw}x{ch}): "
                                "maximize a janela do jogo")
    return x - cx, y - cy, None


# ---------- X de clique ----------
CLIQUE_RAIO = 18  # o X tem ~16 px; a caixa lida é 36x36 em volta do clique
CLIQUE_MIN_PIXELS = 20  # medido: o X tem 42 px no 1º estágio e 64 no 2º


def cores_clique(img):
    """(pixels amarelos, pixels vermelhos) puros — as cores do X que o jogo desenha no clique."""
    p = img.astype(np.int16)
    b, g, r = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    amarelo = int(((r > 220) & (g > 220) & (b < 60)).sum())
    vermelho = int(((r > 220) & (g < 40) & (b < 40)).sum())
    return amarelo, vermelho


def tipo_clique(antes, depois):
    """Compara a região do clique antes e depois: 'chao' (X amarelo: o clique caiu no
    terreno, "Walk here"), 'alvo' (X vermelho: pegou num NPC/item) ou None (sem X ainda)."""
    am0, ve0 = cores_clique(antes)
    am1, ve1 = cores_clique(depois)
    if am1 - am0 >= CLIQUE_MIN_PIXELS:
        return "chao"
    if ve1 - ve0 >= CLIQUE_MIN_PIXELS:
        return "alvo"
    return None


# ---------- Painel do oponente ----------
def painel(img_painel):
    """(pixels vermelhos, pixels verdes) da barra de vida do oponente."""
    p = img_painel.astype(np.int16)
    b, g, r = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    vermelho = int(((r > 150) & (g < 40) & (b < 40)).sum())
    verde = int(((g > 150) & (r < 40) & (b < 40)).sum())
    return vermelho, verde


def estado_combate(img):
    """'livre' (sem painel), 'lutando' (oponente com vida) ou 'morto' (barra zerada).

    O painel ainda fica alguns segundos na tela depois que o sapo morre, todo
    vermelho; sem verde o combate já acabou.
    """
    vermelho, verde = painel(recortar(img, PAINEL_OPONENTE))
    if vermelho + verde <= 1500:
        return "livre"
    return "lutando" if verde > 50 else "morto"


def fracao_vida_oponente(img):
    vermelho, verde = painel(recortar(img, PAINEL_OPONENTE))
    total = vermelho + verde
    return verde / total if total > 1500 else None


# ---------- Sapos ----------
def detectar_sapos(img, perfil=None):
    """Candidatos ao monstro do perfil pela cor: [{'pos': (x, y), 'area': a, 'caixa': (w, h)}]."""
    perfil = perfil or PERFIS["sapo"]
    mask = mascara_monstro(img, perfil)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sapos = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (perfil["area"][0] < area < perfil["area"][1]):
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        pos = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        if na_interface(pos):
            continue
        _, _, w, h = cv2.boundingRect(c)
        sapos.append({"pos": pos, "area": int(area), "caixa": (int(w), int(h))})
    return sapos


def sapo_mais_proximo(img_local, ponto, perfil=None):
    """Centro do sapo mais perto de `ponto` num recorte pequeno (coordenadas do recorte).

    Usado para corrigir a mira logo antes do clique: o sapo anda, e a posição do
    ciclo já tem ~250 ms quando o texto de ação é confirmado.
    """
    perfil = perfil or PERFIS["sapo"]
    mask = mascara_monstro(img_local, perfil)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    melhor, melhor_d = None, None
    for c in contours:
        if cv2.contourArea(c) < perfil["area_mira"]:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        p = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        d = float(np.hypot(p[0] - ponto[0], p[1] - ponto[1]))
        if melhor_d is None or d < melhor_d:
            melhor, melhor_d = p, d
    return melhor, melhor_d


def barra_vida_perto(img, pos, largura=45, acima=85, abaixo=10):
    """Pixels de barra de vida (verde/vermelho puros) logo acima de um candidato.

    Um NPC com barra sobre a cabeça já está lutando com alguém.
    """
    x, y = int(pos[0]), int(pos[1])
    h, w = img.shape[:2]
    r = img[max(y - acima, 0):min(y + abaixo, h), max(x - largura, 0):min(x + largura, w)]
    if r.size == 0:
        return 0
    r = r.astype(np.int16)
    b, g, v = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    verde = (g > 230) & (v < 30) & (b < 30)
    vermelho = (v > 230) & (g < 30) & (b < 30)
    return int(verde.sum() + vermelho.sum())


# ---------- Ossos ----------
def mascara_rotulos(img):
    """(texto branco dos rótulos do chão, mesma máscara desfocada para os moldes)."""
    binm = img.min(axis=2) > ROTULO_BRILHO_MIN
    return binm, cv2.GaussianBlur(binm.astype(np.uint8) * 255, ROTULO_BLUR, 0)


def detectar_ossos_chao(img, threshold=None, origem=(0, 0), mascara=None):
    """Ossos no chão a partir do rótulo 'Big bones': [{'pos': (x, y), 'score': s}].

    `img` pode ser um recorte da janela; nesse caso `origem` é o canto dele, e as
    posições devolvidas continuam em coordenadas da janela. `mascara`: o resultado de
    mascara_rotulos(img), se já tiver sido calculado.
    """
    threshold = ROTULO_THRESHOLD if threshold is None else threshold
    _, mask = mascara_rotulos(img) if mascara is None else mascara
    res = cv2.matchTemplate(mask, ROTULO_OSSO, cv2.TM_CCOEFF_NORMED)
    ossos = []
    for x, y, score in picos(res, threshold, 40, 12):
        pos = (
            origem[0] + x + ROTULO_OFFSET_OSSO[0],
            origem[1] + y + ROTULO_OFFSET_OSSO[1],
        )
        if not na_interface(pos) and dentro_do_jogo(pos):
            ossos.append({"pos": pos, "score": round(score, 3)})
    return ossos


# ---------- Itens do chão: runas e moedas ----------
# Cada item é achado pelo rótulo branco no chão e pego pela linha "Take ..." do menu.
#
# Runas: o rótulo termina com a palavra "rune" ("Nature rune x 6 (1020 gp)", "Mind rune
# (3 gp)"), então um molde só serve para todas. Medido em 431 imagens: rótulos de runa
# >= 0,876; no mundo 3D o resto <= 0,843 ("Limpwurt root"), barrado também porque "rune"
# não tem letra acima da altura do "x" nem abaixo da linha. No menu (45 menus): "rune"
# >= 0,895 nas linhas de runa e <= 0,80 nas outras com esse filtro ("root" dava 0,87).
#
# Moedas ("Coins x 52"): medido em 679 imagens, rótulos visíveis >= 0,83; o que mais se
# parece é "Cosmic rune" (<= 0,819; e pegar uma runa por engano não faz mal). Cobertos
# por outro rótulo ficam em ~0,81 e só são pegos quando ficam livres. No menu: "Take
# Coins (N)" >= 0,94; as outras linhas <= 0,73.
#
# altura_x/colunas: filtro de letras altas/pernas (None = não usa; "Coins" tem o C e o i).
ITENS_CHAO = {
    "runa": {
        "rotulo": cv2.GaussianBlur(_carregar("rune_label_osrs.png", cinza=True), ROTULO_BLUR, 0),
        "limiar": 0.85, "altura_x": (6, 13), "colunas": (3, 31),
        "menu": _carregar("menu_rune_word_osrs.png", cinza=True),
        "menu_limiar": 0.85, "menu_altura_x": (7, 13), "menu_colunas": (3, 36),
    },
    "moedas": {
        "rotulo": cv2.GaussianBlur(_carregar("coins_label_osrs.png", cinza=True), ROTULO_BLUR, 0),
        "limiar": 0.84, "altura_x": None, "colunas": None,
        "menu": _carregar("menu_coins_word_osrs.png", cinza=True),
        "menu_limiar": 0.85, "menu_altura_x": None, "menu_colunas": None,
    },
}
# Nomes antigos (testes e quem calibrou as runas)
ROTULO_RUNA = ITENS_CHAO["runa"]["rotulo"]
RUNA_ALTURA_X, RUNA_COLUNAS = ITENS_CHAO["runa"]["altura_x"], ITENS_CHAO["runa"]["colunas"]
MENU_RUNA = ITENS_CHAO["runa"]["menu"]
ROTULO_ENTRE_LINHAS = 22  # px entre rótulos empilhados sobre a mesma pilha
ROTULO_ITEM = (-3, 23)  # do centro/topo do rótulo de baixo da pilha até o item (medido nos ossos)
ROTULO_MAX_PILHA = 8
ROTULO_LARGURA_MIN = 45  # px; o rótulo mais curto ("Coins x 5") tem ~60
# Na descida pela pilha, o rótulo de baixo pode estar até isso fora do centro calculado
# para o de cima (que sai torto quando encosta em outra pilha); o centro passa a ser o dele.
ROTULO_DESCIDA_TOLERANCIA = 25
ROTULO_DEGRAU_INICIO = (-2, 1)  # px em volta do degrau onde a linha de baixo pode começar
# Palavras do rótulo: dentro da palavra os buracos têm < 4 px; entre palavras, 5-8 px
# (medido: 7-8 px em 572 de 590). Mais que isso já é o rótulo de outra pilha.
ROTULO_BURACO_PALAVRA = 4
ROTULO_ESPACO_MAX = 10
# Estrutura do rótulo em volta da palavra casada: quantas palavras antes dela e quantas
# depois, com e sem o "x N" de quantidade. Runa: "<Nome> rune [x N] (P gp)"; moedas:
# "Coins [x N]".
ESTRUTURA_ROTULO = {"runa": (1, 4, 2), "moedas": (0, 2, 0)}
DENTRO_MARGEM = 6  # px da borda da área do jogo: clique mais perto que isso é recusado


def dentro_do_jogo(pos, margem=DENTRO_MARGEM):
    """O ponto (coordenadas da captura calibrada) está dentro da área do jogo?

    Fora dela o clique cairia na moldura da janela ou na barra de tarefas do Windows.
    """
    cx, cy, cw, ch = CLIENTE_CALIBRADO
    return cx + margem <= pos[0] < cx + cw - margem and cy + margem <= pos[1] < cy + ch - margem


def _palavras(faixa, x1, x2):
    """Palavras (início, fim) com tinta em faixa[x1:x2] (buracos curtos ficam dentro)."""
    cols = np.flatnonzero(faixa[x1:x2]) + x1
    if not len(cols):
        return []
    quebras = np.flatnonzero(np.diff(cols) > ROTULO_BURACO_PALAVRA)
    inicios = np.concatenate(([cols[0]], cols[quebras + 1]))
    fins = np.concatenate((cols[quebras], [cols[-1]]))
    return list(zip(inicios.tolist(), fins.tolist()))


def _centro_rotulo(binm, topo, x_meio, item, limites):
    """Centro x do rótulo do item cuja palavra casada passa por x_meio, ou None se incerto.

    Usa a estrutura do rótulo (ESTRUTURA_ROTULO) em vez da extensão bruta da linha, que
    engolia o rótulo de outra pilha encostado ao lado e tirava o clique da pilha. O "x"
    da quantidade se reconhece por ser estreito e não subir à altura das maiúsculas (o
    "(" sobe). None quando falta um pedaço ou o rótulo encosta na borda (`limites`).
    """
    antes, depois_com_x, depois_sem_x = ESTRUTURA_ROTULO[item]
    faixa = binm[topo:topo + 13].any(axis=0)
    ws = _palavras(faixa, max(x_meio - 320, 0), min(x_meio + 360, len(faixa)))
    k = next((i for i, (a, b) in enumerate(ws) if a <= x_meio <= b), None)
    if k is None:
        return None
    junto = lambda i: ws[i + 1][0] - ws[i][1] <= ROTULO_ESPACO_MAX  # noqa: E731
    t1 = ws[k + 1] if k + 1 < len(ws) and junto(k) else None
    eh_x = (t1 is not None and t1[1] - t1[0] <= 9
            and not binm[topo:topo + 4, t1[0]:t1[1] + 1].any())
    i0, i1 = k - antes, k + (depois_com_x if eh_x else depois_sem_x)
    if i0 < 0 or i1 >= len(ws) or not all(junto(i) for i in range(i0, i1)):
        return None
    esq, dir_ = ws[i0][0], ws[i1][1]
    if esq <= limites[0] + 1 or dir_ >= limites[1] - 2:
        return None  # cortado na borda: o centro visível não é o do rótulo
    return (esq + dir_) // 2


def so_altura_x(binm, x, y, altura_x, colunas, folga=1, k=5):
    """A palavra casada em (x, y) não tem tinta acima da altura-x nem abaixo da base?

    Separa "rune" de palavras parecidas com letra alta ou com perna ("root", "bones").
    """
    topo, base = altura_x
    h, w = binm.shape
    c1, c2 = max(x + colunas[0], 0), min(x + colunas[1], w)
    acima = binm[max(y + topo - k, 0):max(y + topo - 1, 0), c1:c2].sum()
    abaixo = binm[min(y + base + 2, h):min(y + base + k + 1, h), c1:c2].sum()
    return acima <= folga and abaixo <= folga


def _extensao_linha(binm, topo, x_ref, buraco=8, largura_max=320):
    """(esquerda, direita) da linha de texto que começa na altura `topo` e passa por x_ref."""
    faixa = binm[topo:topo + 13].any(axis=0)
    esq = dir_ = x_ref
    vazio = 0
    while esq > 0 and vazio <= buraco and x_ref - esq < largura_max:
        esq -= 1
        vazio = 0 if faixa[esq] else vazio + 1
    esq += vazio
    vazio = 0
    while dir_ < len(faixa) - 1 and vazio <= buraco and dir_ - x_ref < largura_max:
        dir_ += 1
        vazio = 0 if faixa[dir_] else vazio + 1
    dir_ -= vazio
    return esq, dir_


def _centro_abaixo(binm, topo, centro, tolerancia=8):
    """Centro do rótulo da mesma pilha logo abaixo (texto largo, centrado a até
    `tolerancia` px), ou None.

    Tem que ser texto: o desenho do osso no chão também é branco e fica exatamente onde
    estaria a próxima linha.
    """
    prox = topo + ROTULO_ENTRE_LINHAS
    if prox + 13 > binm.shape[0] or not binm[prox + 4:prox + 13, max(centro - 30, 0):centro + 30].any():
        return None
    e2, d2 = _extensao_linha(binm, prox, centro)
    c2 = (e2 + d2) // 2
    if d2 - e2 < ROTULO_LARGURA_MIN or abs(c2 - centro) > tolerancia:
        return None
    # Tem que começar no degrau: a primeira linha com tinta das linhas da mesma pilha fica a
    # -1/0 px dele (medido nas 72 descidas das amostras). Um rótulo de outra pilha logo
    # abaixo começa fora dele (visto em jogo: o desenho do osso a -4 px e o texto a +6;
    # a descida passava do item e o clique caía 22 px abaixo, num chão vazio).
    linhas = binm[prox - 4:prox + 14, e2:d2 + 1].sum(axis=1)
    inicio = next((i - 4 for i, v in enumerate(linhas) if v >= 3), None)
    if inicio is None or not ROTULO_DEGRAU_INICIO[0] <= inicio <= ROTULO_DEGRAU_INICIO[1]:
        return None
    return c2


def detectar_itens_chao(img, item, origem=(0, 0), ossos=None, mascara=None):
    """Itens no chão pelo rótulo: [{'pos': (x, y) da pilha, 'score': s, 'rotulo': (x, y)}].

    `item`: chave de ITENS_CHAO ("runa", "moedas"). O item costuma cair na pilha do osso:
    os rótulos se empilham (um a cada 22 px) e o item fica embaixo do último. `pos` é
    onde clicar para abrir o menu da pilha. `ossos`: resultado de detectar_ossos_chao
    para a mesma imagem (calculado se faltar); `mascara`: resultado de mascara_rotulos(img).
    """
    cfg = ITENS_CHAO[item]
    binm, mask = mascara_rotulos(img) if mascara is None else mascara
    res = cv2.matchTemplate(mask, cfg["rotulo"], cv2.TM_CCOEFF_NORMED)
    # Bordas para o rótulo não ser considerado cortado: as da imagem e as da área do jogo
    cx, _, cw, _ = CLIENTE_CALIBRADO
    limites = (max(0, cx - origem[0]), min(binm.shape[1], cx + cw - origem[0]))
    achados = []
    for x, y, score in picos(res, cfg["limiar"], 20, 8):
        if na_interface((origem[0] + x, origem[1] + y)):
            continue  # texto do chat ou de outro painel, não rótulo do chão
        if cfg["altura_x"] and not so_altura_x(binm, x, y, cfg["altura_x"], cfg["colunas"]):
            continue
        topo = y + 1  # topo das maiúsculas da linha do rótulo
        x_meio = x + cfg["rotulo"].shape[1] // 2
        centro = _centro_rotulo(binm, topo, x_meio, item, limites)
        if centro is None:
            continue  # incerto: melhor não clicar do que clicar em outra pilha
        # desce pela pilha enquanto houver outro rótulo dela logo abaixo. Os rótulos da
        # mesma pilha são centrados no mesmo item: se o de cima saiu torto (o fim dele
        # encostou no rótulo de outra pilha, visto em jogo com "Coins x 52" e "Iron
        # kiteshield"), o de baixo corrige o centro.
        for _ in range(ROTULO_MAX_PILHA):
            c2 = _centro_abaixo(binm, topo, centro, ROTULO_DESCIDA_TOLERANCIA)
            if c2 is None:
                break
            topo += ROTULO_ENTRE_LINHAS
            centro = c2
        pos = (origem[0] + centro + ROTULO_ITEM[0], origem[1] + topo + ROTULO_ITEM[1])
        # Rótulo de outra pilha encavalado no de baixo atrapalha a descida; se houver um
        # osso logo abaixo na mesma coluna (até uma linha), a pilha é a dele.
        if ossos is None:
            ossos = detectar_ossos_chao(img, origem=origem, mascara=(binm, mask))
        abaixo = [o["pos"] for o in ossos if abs(o["pos"][0] - pos[0]) <= ROTULO_DESCIDA_TOLERANCIA
                  and -6 <= o["pos"][1] - pos[1] <= ROTULO_ENTRE_LINHAS + 6]
        if abaixo:
            pos = max(abaixo, key=lambda p: p[1])
        if not na_interface(pos) and dentro_do_jogo(pos):
            achados.append({"pos": pos, "score": round(score, 3),
                            "rotulo": (origem[0] + x, origem[1] + y)})
    return achados


def detectar_runas_chao(img, origem=(0, 0), ossos=None, mascara=None):
    return detectar_itens_chao(img, "runa", origem, ossos, mascara)


# Linha "Take <item>" do menu de botão direito. "Take" medido em 45 menus: >= 0,87 nas
# linhas certas e <= 0,71 no resto (inclusive "Examine").
MENU_TAKE_PALAVRA = _carregar("menu_take_word_osrs.png", cinza=True)
MENU_TAKE_THRESHOLD = 0.80


def achar_take_item_no_menu(img_regiao, item):
    """Procura no menu aberto uma linha "Take <item>" (e não "Examine <item>").

    Devolve ((x, y) do "Take" dessa linha dentro do recorte, score do item) ou
    (None, nota): a maior nota da palavra do item vista numa linha "Take" (mesmo abaixo
    do limiar ou barrada pelo filtro de altura), ou a do "Take" se nenhuma linha "Take"
    apareceu (~0,6 = menu sem "Take", por exemplo clique num ponto sem itens).
    """
    cfg = ITENS_CHAO[item]
    mt = mascara_texto(img_regiao)
    binm = mt > 100
    th, tw = MENU_TAKE_PALAVRA.shape
    rh, rw = cfg["menu"].shape
    rt = cv2.matchTemplate(mt, MENU_TAKE_PALAVRA, cv2.TM_CCOEFF_NORMED)
    melhor, melhor_score = None, 0.0
    nota_item = None
    for x, y, _ in picos(rt, MENU_TAKE_THRESHOLD, 30, 8, limite=12):
        y1, x1 = max(y - 3, 0), x + tw
        faixa = mt[y1:y + th + 3, x1:x1 + 300]
        if faixa.shape[0] < rh or faixa.shape[1] < rw:
            continue
        rr = cv2.matchTemplate(faixa, cfg["menu"], cv2.TM_CCOEFF_NORMED)
        nota_item = max(nota_item or 0.0, float(rr.max()))
        for rx, ry, score in picos(rr, cfg["menu_limiar"], 15, 4, limite=4):
            if score <= melhor_score:
                continue
            if cfg["menu_altura_x"] and not so_altura_x(binm, x1 + rx, y1 + ry, cfg["menu_altura_x"],
                                                          cfg["menu_colunas"]):
                continue
            melhor, melhor_score = (x + tw // 2, y + th // 2), float(score)
    if melhor is None:
        if nota_item is not None:
            return None, nota_item
        return None, float(rt.max()) if rt.size else 0.0
    return melhor, melhor_score


def menu_tem_take(img_regiao):
    """O menu aberto tem alguma linha "Take ..."? Sem nenhuma, o clique caiu num quadrado
    do chão sem itens (o menu só tem "Walk here" e "Cancel")."""
    if img_regiao is None or img_regiao.shape[0] < MENU_TAKE_PALAVRA.shape[0] \
            or img_regiao.shape[1] < MENU_TAKE_PALAVRA.shape[1]:
        return False
    res = cv2.matchTemplate(mascara_texto(img_regiao), MENU_TAKE_PALAVRA, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) >= MENU_TAKE_THRESHOLD


def achar_take_runa_no_menu(img_regiao):
    return achar_take_item_no_menu(img_regiao, "runa")


def detectar_ossos_inventario(img_inventario):
    """Centros dos ícones de osso, em coordenadas da janela (recebe o recorte INVENTARIO)."""
    res = cv2.matchTemplate(img_inventario, ICONE_OSSO, cv2.TM_CCOEFF_NORMED)
    h, w = ICONE_OSSO.shape[:2]
    x1, y1 = INVENTARIO[0], INVENTARIO[1]
    pontos = [
        (x1 + x + w // 2, y1 + y + h // 2)
        for x, y, _ in picos(res, ICONE_THRESHOLD, w // 2, h // 2)
    ]
    # ordem de leitura (linha, coluna) para enterrar sempre na mesma sequência
    return sorted(pontos, key=lambda p: (p[1] // 20, p[0]))


# =====================================================================
# Navegação e coleta (bots que andam entre pontos fixos, ex.: yew.py)
# Pressupõe câmera fixa: bússola no norte e zoom no mínimo.
# =====================================================================

# ---------- Minimapa: localização do jogador ----------
MINIMAPA = (1670, 30, 1880, 250)  # recorte da janela que contém o minimapa
MM_CENTRO = (108, 115)  # posição do jogador dentro do recorte
MM_PX_TILE = 5  # 4 px por tile do jogo x 1,25 da escala do Windows
MM_RAIO_PATCH = 20
MM_RAIO_CLIQUE = 80  # alcance útil de um clique no minimapa (px)
MM_SCORE_MIN = 0.75


def carregar_referencias(refs):
    """refs: {nome: (arquivo em templates/, (tile_x, tile_y))} -> lista pronta para localizar()."""
    return [(nome, tile, _carregar(arquivo)) for nome, (arquivo, tile) in refs.items()]


def localizar(img_minimapa, referencias):
    """Posição do jogador em tiles, comparando o minimapa atual com fotos de referência.

    Cada referência é o minimapa fotografado num tile conhecido. O miolo dela é
    procurado no minimapa atual; o quanto ele aparece deslocado do centro diz a
    quantos tiles o jogador está daquele ponto. Devolve ((x, y), score, nome) da
    melhor referência, ou (None, score, nome) se nenhuma bater.
    """
    cx, cy = MM_CENTRO
    r = MM_RAIO_PATCH
    melhor = (None, 0.0, None)
    for nome, tile, ref in referencias:
        patch = ref[cy - r:cy + r, cx - r:cx + r]
        res = cv2.matchTemplate(img_minimapa, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > melhor[1]:
            dx, dy = loc[0] + r - cx, loc[1] + r - cy
            pos = (tile[0] - dx / MM_PX_TILE, tile[1] - dy / MM_PX_TILE)
            melhor = (pos, float(score), nome)
    if melhor[1] < MM_SCORE_MIN:
        return None, melhor[1], melhor[2]
    return melhor


def ponto_minimapa(delta_tiles):
    """Ponto da janela para clicar no minimapa e andar `delta_tiles` (limitado ao alcance)."""
    dx, dy = delta_tiles[0] * MM_PX_TILE, delta_tiles[1] * MM_PX_TILE
    d = float(np.hypot(dx, dy))
    if d > MM_RAIO_CLIQUE:
        dx, dy = dx * MM_RAIO_CLIQUE / d, dy * MM_RAIO_CLIQUE / d
    return (MINIMAPA[0] + MM_CENTRO[0] + int(round(dx)), MINIMAPA[1] + MM_CENTRO[1] + int(round(dy)))


# ---------- Tela principal: onde um tile aparece ----------
JOGADOR = (938, 585)  # onde o personagem fica na tela
PX_TILE = 50  # tamanho aproximado de um tile no zoom mínimo


def ponto_tela(delta_tiles):
    """Ponto aproximado da janela onde aparece o tile a `delta_tiles` do jogador."""
    return (int(JOGADOR[0] + delta_tiles[0] * PX_TILE), int(JOGADOR[1] + delta_tiles[1] * PX_TILE))


def visivel(pos, margem=25):
    """O ponto está na área 3D da janela (fora da interface)?"""
    x, y = pos
    if not (margem <= x < 1880 - margem and 45 + margem <= y < 1160 - margem):
        return False
    return not na_interface(pos)


# ---------- Inventário: slots ocupados ----------
INVENTARIO_VAZIO = _carregar("inventory_empty_osrs.png")
SLOT_ORIGEM = (76, 37)  # centro do 1º slot dentro do recorte INVENTARIO
SLOT_PASSO = (52.5, 45)
SLOT_MEIA = (15, 13)
SLOT_DIFERENCA_MIN = 12


def slots_ocupados(img_inventario):
    """Quantos dos 28 slots têm algum item (compara com a foto do inventário vazio)."""
    n = 0
    for linha in range(7):
        for coluna in range(4):
            x = int(SLOT_ORIGEM[0] + coluna * SLOT_PASSO[0])
            y = int(SLOT_ORIGEM[1] + linha * SLOT_PASSO[1])
            a = img_inventario[y - SLOT_MEIA[1]:y + SLOT_MEIA[1], x - SLOT_MEIA[0]:x + SLOT_MEIA[0]]
            b = INVENTARIO_VAZIO[y - SLOT_MEIA[1]:y + SLOT_MEIA[1], x - SLOT_MEIA[0]:x + SLOT_MEIA[0]]
            if float(cv2.absdiff(a, b).mean()) > SLOT_DIFERENCA_MIN:
                n += 1
    return n


# ---------- Árvores ----------
def fracao_copa(img, pos, raio=40):
    """Fração de pixels com cor de copa (verde escuro) numa caixa em volta de `pos`.

    Árvore em pé: copa cobre a caixa. Toco: só aparece o chão marrom (matiz 22-25).
    """
    x, y = int(pos[0]), int(pos[1])
    h, w = img.shape[:2]
    rec = img[max(y - raio, 0):min(y + raio, h), max(x - raio, 0):min(x + raio, w)]
    if rec.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rec, cv2.COLOR_BGR2HSV)
    copa = cv2.inRange(hsv, np.array([28, 90, 20]), np.array([42, 255, 95]))
    return float((copa > 0).mean())
