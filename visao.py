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
    """Pixels com as cores do texto de ação: branco, amarelo, verde-limão e laranja."""
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    branco = (r > 200) & (g > 200) & (b > 200)
    amarelo = (r > 200) & (g > 200) & (b < 90)
    verde = (r < 90) & (g > 200) & (b < 90)
    laranja = (r > 200) & (g > 100) & (g < 190) & (b < 110)
    m = (branco | amarelo | verde | laranja).astype(np.uint8) * 255
    return cv2.GaussianBlur(m, (3, 3), 0)


def score_texto(faixa, chave):
    """Semelhança (0-1) entre a faixa do texto de ação e o template `chave`."""
    res = cv2.matchTemplate(mascara_texto(faixa), TEXTOS[chave], cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def faixa_texto(img):
    return recortar(img, FAIXA_TEXTO_ACAO)


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


def vida_pct(img):
    """Fração de vida do jogador (0-1), lida pela COR do número do orbe.

    O jogo pinta o número num degradê contínuo: verde (0,255,0) com vida cheia,
    amarelo (255,255,0) em 50% e vermelho (255,0,0) em 0%. Em HSV isso é linear:
    vida = matiz / 60 (matiz do OpenCV). Conferido: 72/74 -> matiz 58 -> 97%.
    Devolve None se o número não estiver visível.
    """
    hsv = cv2.cvtColor(recortar(img, REGIAO_VIDA), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    cor = hsv[(hsv[:, 1] > 180) & (hsv[:, 2] > 150)]
    if len(cor) < 8:
        return None
    return float(min(np.median(cor[:, 0]) / 60.0, 1.0))


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
def detectar_ossos_chao(img, threshold=None, origem=(0, 0)):
    """Ossos no chão a partir do rótulo 'Big bones': [{'pos': (x, y), 'score': s}].

    `img` pode ser um recorte da janela; nesse caso `origem` é o canto dele, e as
    posições devolvidas continuam em coordenadas da janela.
    """
    threshold = ROTULO_THRESHOLD if threshold is None else threshold
    mask = ((img.min(axis=2) > ROTULO_BRILHO_MIN).astype(np.uint8)) * 255
    mask = cv2.GaussianBlur(mask, ROTULO_BLUR, 0)
    res = cv2.matchTemplate(mask, ROTULO_OSSO, cv2.TM_CCOEFF_NORMED)
    ossos = []
    for x, y, score in picos(res, threshold, 40, 12):
        pos = (
            origem[0] + x + ROTULO_OFFSET_OSSO[0],
            origem[1] + y + ROTULO_OFFSET_OSSO[1],
        )
        if not na_interface(pos):
            ossos.append({"pos": pos, "score": round(score, 3)})
    return ossos


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
