"""Teste de regressão da detecção, usando as amostras gravadas pelo bot.

  python main.py --amostras     # grava frames em amostras/ enquanto joga
  python testar_visao.py        # confere a detecção contra essas amostras (até 4 processos)
  python testar_visao.py -j 1   # um teste por vez: use com o bot ligado (não rouba CPU dele)

Cada amostra foi rotulada pelo próprio jogo: o bot só grava "sapo_ok"/"osso_ok"
depois que o texto de ação confirmou o que havia naquela posição. Rode este
teste sempre que mexer em visao.py ou trocar um template.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

import visao

PASTA = os.path.join(visao.BASE_DIR, "amostras")
TOLERANCIA_OSSO = 20  # px
TOLERANCIA_SAPO = 30


def perto(pos, pontos, tol):
    return any(np.hypot(pos[0] - p[0], pos[1] - p[1]) <= tol for p in pontos)


def testar_frames():
    caminho = os.path.join(PASTA, "rotulos.jsonl")
    if not os.path.exists(caminho):
        return []
    for nome in visao.perfis_disponiveis():  # amostras gravadas com perfis aprendidos também
        visao.carregar_perfil(nome)
    falhas, n = [], {"osso_ok": 0, **{f"{p}_ok": 0 for p in visao.PERFIS}}
    piores = {"osso_ok": 1.0}
    with open(caminho, encoding="utf-8") as f:
        rotulos = [json.loads(linha) for linha in f if linha.strip()]
    for r in rotulos:
        if r["classe"] not in n:
            continue
        img = cv2.imread(os.path.join(PASTA, r["arquivo"]))
        if img is None:
            continue
        n[r["classe"]] += 1
        if r["classe"] == "osso_ok":
            ossos = visao.detectar_ossos_chao(img)
            achados = [o for o in ossos if perto(r["pos"], [o["pos"]], TOLERANCIA_OSSO)]
            if not achados:
                falhas.append(f"{r['arquivo']}: osso em {r['pos']} não detectado")
            else:
                piores["osso_ok"] = min(piores["osso_ok"], achados[0]["score"])
        else:
            perfil = visao.PERFIS[r["classe"][:-3]]
            sapos = [s["pos"] for s in visao.detectar_sapos(img, perfil)]
            if not perto(r["pos"], sapos, TOLERANCIA_SAPO):
                falhas.append(f"{r['arquivo']}: {perfil['nome']} em {r['pos']} não detectado")
    print("Frames confirmados:", ", ".join(f"{v} {k[:-3]}" for k, v in n.items()))
    if n["osso_ok"]:
        print(f"  menor score de rótulo verdadeiro: {piores['osso_ok']:.3f} "
              f"(limiar {visao.ROTULO_THRESHOLD})")
    return falhas


def testar_faixas():
    esperado = {"sapo-giant": "giant", "sapo-hillgiant": "hillgiant", "take-ok": "take",
                "sapo-None": None, "take-rej": None}
    falhas, n = [], 0
    # só os moldes embutidos: os aprendidos (perfil_*) são conferidos em testar_perfis_aprendidos
    chaves = [c for c in visao.TEXTOS if not c.startswith("perfil_")]
    margens = {c: [] for c in chaves}
    for caminho in sorted(glob.glob(os.path.join(PASTA, "faixas", "*.png"))):
        nome = os.path.basename(caminho)
        classe = nome.split("_")[1]
        if classe not in esperado:
            continue
        faixa = cv2.imread(caminho)
        scores = {c: visao.score_texto(faixa, c) for c in chaves}
        lido = max(scores, key=scores.get)
        if scores[lido] < visao.TEXTO_THRESHOLD:
            lido = None
        n += 1
        for c, s in scores.items():
            if classe != "take-rej":
                margens[c].append((s, esperado[classe] == c))
        # "take-rej": o osso estava sob outra coisa (ex.: um sapo em cima dele);
        # qualquer leitura serve, desde que não seja "take".
        if classe == "take-rej" and lido != "take":
            continue
        if lido != esperado[classe]:
            falhas.append(f"{nome}: esperado {esperado[classe]}, lido {lido} "
                          f"({ {c: round(s, 2) for c, s in scores.items()} })")
    print(f"Faixas de texto: {n}")
    for c, valores in margens.items():
        certos = [s for s, ok in valores if ok]
        errados = [s for s, ok in valores if not ok]
        if certos and errados:
            print(f"  {c}: corretos >= {min(certos):.3f}   incorretos <= {max(errados):.3f}   "
                  f"(limiar {visao.TEXTO_THRESHOLD})")
    return falhas


def testar_cliques():
    """Cada amostra é a região do clique antes|depois; o nome traz o tipo lido ao vivo."""
    falhas, n = [], {"alvo": 0, "chao": 0, "None": 0}
    for caminho in sorted(glob.glob(os.path.join(PASTA, "cliques", "*.png"))):
        nome = os.path.basename(caminho)
        esperado = nome.rsplit("_", 1)[1].replace(".png", "")
        par = cv2.imread(caminho)
        meio = par.shape[1] // 2
        lido = str(visao.tipo_clique(par[:, :meio], par[:, meio:]))
        n[esperado] = n.get(esperado, 0) + 1
        if lido != esperado:
            falhas.append(f"{nome}: esperado {esperado}, lido {lido}")
    print(f"Cliques: {n['alvo']} com X vermelho, {n['chao']} com X amarelo, {n['None']} sem leitura")
    return falhas


def testar_menus():
    """Recortes de menus abertos; "ok" = o bot achou a linha e o osso entrou no inventário."""
    falhas, n = [], 0
    menus = glob.glob(os.path.join(PASTA, "menus", "*_ok.png")) + \
        glob.glob(os.path.join(PASTA, "menus", "*_ok_osso.png"))  # antigos sem sufixo + gravados agora
    for caminho in sorted(menus):
        n += 1
        ponto, score = visao.achar_take_no_menu(cv2.imread(caminho))
        if ponto is None:
            falhas.append(f"{os.path.basename(caminho)}: linha Take Big bones não achada ({score:.2f})")
    print(f"Menus: {n} com a linha \"Take Big bones\"")
    return falhas


# ---------- Vida (orbe de Hitpoints) ----------
CAPTURA = visao.CAPTURA_CALIBRADA[::-1]  # (altura, largura) de um frame inteiro
TOLERANCIA_VIDA = 0.025
# Caixas com o mesmo desenho da do orbe de vida (canto x, y). A moldura é idêntica; o
# que as separa é pouco (2,07 no especial), por isso o limiar fica bem abaixo disso.
OUTRAS_CAIXAS = {(1650, 222): "orbe de especial", (1610, 150): "orbe de oração",
                 (1622, 191): "orbe de corrida"}
# Onde o orbe de vida é pintado: o preenchimento esvazia de cima para baixo conforme a
# vida cai e muda de cor com veneno. Nada daqui pode influenciar a leitura.
ORBE_VIDA = (1646, 95, 1700, 146)


SEM_FRAMES = (f"nenhum frame {CAPTURA[1]}x{CAPTURA[0]} em amostras/: os testes de vida não "
              "testaram nada (rode `python main.py --amostras` com a janela maximizada)")


def frames_jogo(passo=1):
    """Frames inteiros da janela (tamanho calibrado) gravados em amostras/."""
    frames = []
    for caminho in sorted(glob.glob(os.path.join(PASTA, "*.png")))[::passo]:
        img = cv2.imread(caminho)
        if img is not None and img.shape[:2] == CAPTURA:
            frames.append((os.path.basename(caminho), img))
    return frames


def deslocar(img, dx, dy, buf=None):
    """Simula a captura deslocada: o que a área do orbe mostra se a janela andar (dx, dy).

    Só a região em volta do orbe é copiada (é tudo o que vida_pct lê); o resto fica preto.
    """
    buf = np.zeros_like(img) if buf is None else buf
    x1, y1, x2, y2 = 1590, 95, 1670, 150
    buf[y1:y2, x1:x2] = img[y1 + dy:y2 + dy, x1 + dx:x2 + dx]
    return buf


def pintar_numero(img, matiz):
    """Repinta o número do orbe com outro matiz (o jogo usa matiz = 60 x vida)."""
    out = img.copy()
    x1, y1, x2, y2 = visao.REGIAO_VIDA
    rec = out[y1:y2, x1:x2]
    hsv = cv2.cvtColor(rec, cv2.COLOR_BGR2HSV)
    numero = (hsv[..., 1] > 180) & (hsv[..., 2] > 150)
    hsv[..., 0][numero] = matiz
    rec[numero] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[numero]
    return out


def mexer_no_orbe(img, como, semente=0):
    """Simula o que o orbe faz em jogo: esvaziar, ficar verde (veneno) ou qualquer coisa."""
    out = img.copy()
    x1, y1, x2, y2 = ORBE_VIDA
    rec = out[y1:y2, x1:x2]
    if como == "vazio":
        rec[:] = (20, 18, 22)
    elif como == "veneno":
        b, g, r = (rec[..., i].astype(int) for i in range(3))
        vermelho = r > g + 40
        rec[vermelho] = rec[vermelho][:, [0, 2, 1]]  # troca vermelho por verde
    elif como == "ruido":
        rec[:] = np.random.default_rng(semente).integers(0, 256, rec.shape, dtype=np.uint8)
    return out


def testar_vida_amostras():
    """Todo frame gravado em jogo tem o orbe no lugar: a vida tem que ser lida."""
    falhas, lidas, difs = [], [], []
    frames = frames_jogo()
    if not frames:
        return [SEM_FRAMES]
    for nome, img in frames:
        vida = visao.vida_pct(img)
        difs.append(visao.diferenca_caixa_vida(img))
        if vida is None:
            falhas.append(f"{nome}: vida não lida (caixa {difs[-1]:.2f})")
        else:
            lidas.append(vida)
    if lidas:
        print(f"Vida: {len(lidas)} frames lidos ({min(lidas):.0%} a {max(lidas):.0%}); "
              f"caixa no lugar difere no máximo {max(difs):.2f} (limiar {visao.CAIXA_VIDA_LIMIAR})")
    return falhas


def testar_vida_baixa():
    """Vida baixa simulada (número repintado) com o orbe em vários estados."""
    falhas, n = [], 0
    casos = [(round(60 * v), v) for v in (1.0, 0.75, 0.6, 0.5, 0.45, 0.39, 0.28, 0.15, 0.05, 0.0)]
    casos.append((179, 0.0))  # vermelho "do outro lado" do círculo de matiz
    frames = frames_jogo(passo=15)
    if not frames:
        return [SEM_FRAMES]
    for nome, img in frames:
        for como in ("igual", "vazio", "veneno", "ruido"):
            base = mexer_no_orbe(img, como, semente=n)
            for matiz, esperado in casos:
                n += 1
                vida = visao.vida_pct(pintar_numero(base, matiz))
                if vida is None or abs(vida - esperado) > TOLERANCIA_VIDA:
                    falhas.append(f"{nome} (orbe {como}, matiz {matiz}): esperado "
                                  f"{esperado:.0%}, lido {vida if vida is None else f'{vida:.0%}'}")
        # Número com duas cores não é o número do orbe (o jogo pinta tudo de uma cor só)
        bicolor = pintar_numero(img, 17)
        x1, y1, x2, y2 = visao.REGIAO_VIDA
        meio = (x1 + x2) // 2
        bicolor[:, meio:] = pintar_numero(img, 55)[:, meio:]
        n += 1
        vida = visao.vida_pct(bicolor)
        if vida is not None:
            falhas.append(f"{nome}: número com duas cores lido como {vida:.0%} (esperado nada)")
    print(f"Vida baixa simulada: {n} leituras (número repintado; orbe cheio, vazio, "
          "envenenado e com ruído; número com duas cores)")
    return falhas


def testar_vida_deslocada():
    """O conserto em si: com a janela fora do lugar, vida_pct não pode devolver nada.

    Todos os deslocamentos até ±60 px (inclui os 43 px da sessão de 14/11 21:10 que
    pausou o bot à toa). A ±1 px o número continua dentro da área lida: ali só se exige
    que, se ler, leia certo.
    """
    falhas, n = [], 0
    frames = frames_jogo(passo=45)
    if not frames:
        return [SEM_FRAMES]
    for nome, img in frames:
        certa = visao.vida_pct(img)
        buf = np.zeros_like(img)
        lidas_fora = 0
        for dy in range(-60, 61):
            for dx in range(-60, 61):
                if dx == 0 and dy == 0:
                    continue
                n += 1
                vida = visao.vida_pct(deslocar(img, dx, dy, buf))
                if vida is None:
                    continue
                if abs(dx) > 1 or abs(dy) > 1:
                    lidas_fora += 1
                    if lidas_fora <= 3:
                        falhas.append(f"{nome}: janela deslocada ({dx}, {dy}) leu vida {vida:.0%}")
                elif abs(vida - certa) > TOLERANCIA_VIDA:
                    falhas.append(f"{nome}: deslocada ({dx}, {dy}) leu {vida:.0%}, certo {certa:.0%}")
        if lidas_fora > 3:
            falhas.append(f"{nome}: ... {lidas_fora} deslocamentos leram alguma vida")
    print(f"Janela deslocada: {n} leituras (±60 px em {len(frames)} frames)")
    return falhas


def mapa_caixa(img):
    """Diferença da caixa do orbe em TODAS as posições da imagem (canto x, y)."""
    m = visao.CAIXA_VIDA_MASCARA
    ref = visao.CAIXA_VIDA_REF.astype(np.uint8)
    h, w = m.shape
    hs, ws = img.shape[0] - h + 1, img.shape[1] - w + 1
    acc = np.zeros((hs, ws), np.int32)
    canais = cv2.split(img)
    pixels = list(zip(*np.where(m)))
    for y, x in pixels:
        for c in range(3):
            v = canais[c][y:y + hs, x:x + ws]
            acc += cv2.absdiff(v, np.full_like(v, ref[y, x, c]))
    return acc / (3.0 * len(pixels))


def testar_vida_cenario():
    """Em nenhuma outra posição da imagem a moldura da caixa do orbe pode 'bater'.

    Calcula a diferença da caixa em todas as posições de cada frame (o cenário inteiro,
    as caixas dos outros orbes, que têm o mesmo desenho, e a própria caixa deslocada).
    Só a posição certa pode passar (±1 px, onde o número continua dentro da área lida e
    a leitura tem que sair igual). O mapa é conferido contra visao.diferenca_caixa_vida.
    """
    falhas, n_janelas = [], 0
    menor = {"resto da imagem": float("inf"), **{nome: float("inf") for nome in OUTRAS_CAIXAS.values()}}
    cx, cy = visao.CAIXA_VIDA[:2]
    frames = frames_jogo(passo=30)
    if not frames:
        return [SEM_FRAMES]
    for nome, img in frames:
        mapa = mapa_caixa(img)
        n_janelas += mapa.size
        verdadeira = visao.vida_pct(img)
        # o mapa deste teste tem que dar o mesmo que a função usada pelo bot
        for (px, py) in [(cx, cy), (cx + 5, cy), (cx, cy + 3)] + list(OUTRAS_CAIXAS):
            real = visao.diferenca_caixa_vida(deslocar(img, px - cx, py - cy))
            if abs(real - mapa[py, px]) > 1e-3:
                falhas.append(f"{nome}: mapa_caixa ({mapa[py, px]:.3f}) difere de "
                              f"visao.diferenca_caixa_vida ({real:.3f}) em ({px}, {py})")
        ys, xs = np.where(mapa <= visao.CAIXA_VIDA_LIMIAR)
        for x, y in zip(xs, ys):
            dx, dy = int(x) - cx, int(y) - cy
            if abs(dx) > 1 or abs(dy) > 1:
                onde = OUTRAS_CAIXAS.get((int(x), int(y)), "cenário")
                falhas.append(f"{nome}: {onde} em ({x}, {y}) passa por caixa do orbe de vida "
                              f"(diferença {mapa[y, x]:.2f})")
                continue
            # janela 1 px fora do lugar: se a caixa passar, a leitura tem que sair certa
            vida = visao.vida_pct(deslocar(img, dx, dy))
            if vida is not None and abs(vida - verdadeira) > TOLERANCIA_VIDA:
                falhas.append(f"{nome}: deslocado ({dx}, {dy}) leu {vida:.0%}, "
                              f"certo {verdadeira:.0%}")
        fora = mapa.copy()
        fora[cy - 1:cy + 2, cx - 1:cx + 2] = np.inf
        for (ox, oy), orbe in OUTRAS_CAIXAS.items():
            menor[orbe] = min(menor[orbe], float(fora[oy, ox]))
            fora[oy - 1:oy + 2, ox - 1:ox + 2] = np.inf
        menor["resto da imagem"] = min(menor["resto da imagem"], float(fora.min()))
    print(f"Vida no cenário: {n_janelas} posições em {len(frames)} frames; menor diferença "
          f"fora da caixa certa: " + ", ".join(f"{k} {v:.2f}" for k, v in menor.items())
          + f" (limiar {visao.CAIXA_VIDA_LIMIAR})")
    return falhas


def testar_vigia_vida():
    """Pausa/retomada: uma leitura isolada nunca pausa nem retoma o bot."""
    casos = [
        ("leitura baixa isolada", [0.9, 0.28, 0.9, 0.9], [None] * 4),
        ("três baixas seguidas", [0.35, 0.3, 0.2], [None, None, "pausar"]),
        ("None não zera a contagem", [0.3, None, 0.3, None, 0.3],
         [None, None, None, None, "pausar"]),
        ("leitura boa no meio zera", [0.3, 0.3, 0.5, 0.3, 0.3], [None] * 5),
        ("40% exatos não é baixa", [0.40] * 4, [None] * 4),
        ("pausado: 59% não retoma", [0.3, 0.3, 0.3, 0.59, 0.59, 0.59, 0.59],
         [None, None, "pausar", None, None, None, None]),
        ("pausado: retoma com 3 leituras >= 60%", [0.3, 0.3, 0.3, 0.6, 0.9, 0.7],
         [None, None, "pausar", None, None, "retomar"]),
        ("pausado: leitura alta isolada não retoma", [0.3, 0.3, 0.3, 0.9, 0.3, 0.9, 0.9],
         [None, None, "pausar", None, None, None, None]),
        ("ciclo completo duas vezes", [0.1] * 3 + [1.0] * 3 + [0.1] * 3,
         [None, None, "pausar", None, None, "retomar", None, None, "pausar"]),
        # main.py descarta a leitura que não se confirmou na hora: não sobra contagem
        ("descartada não soma depois", [0.3, 0.3, "descartar", 0.3, 0.3],
         [None, None, None, None, None]),
        ("descartar pausado não retoma", [0.3, 0.3, 0.3, 0.9, 0.9, "descartar", 0.9],
         [None, None, "pausar", None, None, None, None]),
    ]
    falhas = []
    for nome, leituras, esperado in casos:
        vigia = visao.VigiaVida(0.40, 0.60, 3)
        obtido = [vigia.descartar() if v == "descartar" else vigia.ler(v) for v in leituras]
        if obtido != esperado:
            falhas.append(f"VigiaVida ({nome}): esperado {esperado}, obtido {obtido}")
    print(f"Pausa por vida: {len(casos)} sequências")
    return falhas


# ---------- Itens do chão: runas e moedas ----------
# Gabaritos em amostras/runas/ (conferidos a olho nas montagens): esperado.json (runas)
# e esperado_moedas.json, para os quadros de amostras/runas/ e os de amostras/ ("../").
GABARITOS_CHAO = {"runa": "esperado.json", "moedas": "esperado_moedas.json"}
# Menus gravados com a linha "Take <item>": y do "Take" esperado. Os outros não podem ter.
MENUS_COM_ITEM = {
    "runa": {"151929_001_ok.png": 374, "152556_019_ok.png": 397, "152805_003_ok.png": 374,
             "152957_008_ok.png": 374, "153257_016_ok.png": 374},
    "moedas": {"152126_008_ok.png": 374, "152251_011_ok.png": 374, "152330_013_ok.png": 374,
               "152412_015_ok.png": 374, "152456_016_ok.png": 374, "152623_022_ok.png": 374,
               "152723_001_ok.png": 374, "152921_006_ok.png": 374, "152940_007_ok.png": 374,
               "152957_008_ok.png": 398, "153019_009_ok.png": 374, "153033_010_ok.png": 374,
               "153220_015_ok.png": 374, "153350_018_ok.png": 374, "153413_019_ok.png": 398,
               "153436_020_ok.png": 374},
}
TOLERANCIA_ROTULO = 4
TOLERANCIA_PILHA = 4


def testar_item_chao(item):
    """Rótulos do item no chão contra o gabarito: todos os visíveis, nenhum a mais, e a
    pilha certa (na pilha de um osso, o clique vai no mesmo ponto do osso)."""
    pasta = os.path.join(PASTA, "runas")
    caminho = os.path.join(pasta, GABARITOS_CHAO[item])
    if not os.path.exists(caminho):
        return [f"sem amostras/runas/{GABARITOS_CHAO[item]}: o teste de {item} não testou nada"]
    with open(caminho, encoding="utf-8") as f:
        esperado = json.load(f)
    falhas, n_rot, n_pilha_osso, encobertas = [], 0, 0, 0
    perto = lambda a, b, tol: abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol  # noqa: E731
    for nome, exp in esperado.items():
        img = cv2.imread(os.path.join(pasta, nome))
        if img is None:
            falhas.append(f"runas/{nome}: imagem não encontrada")
            continue
        mascara = visao.mascara_rotulos(img)
        ossos = visao.detectar_ossos_chao(img, mascara=mascara)
        achados = visao.detectar_itens_chao(img, item, ossos=ossos, mascara=mascara)
        for e, p in zip(exp["rotulos"], exp["pilhas"]):
            n_rot += 1
            r = next((r for r in achados if perto(r["rotulo"], e, TOLERANCIA_ROTULO)), None)
            if r is None:
                falhas.append(f"{item} em runas/{nome}: rótulo em {e} não detectado")
            elif not perto(r["pos"], p, TOLERANCIA_PILHA):
                falhas.append(f"{item} em runas/{nome}: pilha em {r['pos']}, esperado {p}")
        for r in achados:
            if not any(perto(r["rotulo"], e, TOLERANCIA_ROTULO) for e in exp["rotulos"]):
                falhas.append(f"{item} em runas/{nome}: rótulo a mais em {r['rotulo']}")
            o = [o["pos"] for o in ossos if abs(o["pos"][0] - r["pos"][0]) <= 12
                 and abs(o["pos"][1] - r["pos"][1]) <= 12]
            if o:
                n_pilha_osso += 1
                if not perto(r["pos"], o[0], 3):
                    falhas.append(f"{item} em runas/{nome}: pilha {r['pos']} longe do osso {o[0]}")
        encobertas += len(exp.get("encobertas", []))
    print(f"{item.capitalize()} no chão: {n_rot} rótulos em {len(esperado)} frames ({n_pilha_osso} na "
          f"pilha de um osso); {encobertas} encoberto(s) por outro rótulo, não exigido(s)")
    return falhas


def testar_item_menu(item):
    """Linha "Take <item>" nos menus gravados: só onde tem, e nunca no "Examine".

    Menus antigos (HHMMSS_NNN_ok.png, de osso): gabarito MENUS_COM_ITEM. Menus gravados
    pela coleta (HHMMSS_NNN_ok_<item>.png): o bot achou e clicou a linha daquele item,
    então ela tem que ser achada de novo; sobre os outros itens não se afirma nada.
    """
    falhas, n = [], 0
    esperados = MENUS_COM_ITEM[item]
    itens = tuple(visao.ITENS_CHAO) + ("osso",)  # sufixos dos menus gravados pela coleta
    for caminho in sorted(glob.glob(os.path.join(PASTA, "menus", "*.png"))):
        nome = os.path.basename(caminho)
        de_coleta = nome[:-4].endswith(itens)
        if de_coleta and not nome.endswith(f"_ok_{item}.png"):
            continue
        n += 1
        ponto, score = visao.achar_take_item_no_menu(cv2.imread(caminho), item)
        if de_coleta:
            if ponto is None:
                falhas.append(f"menus/{nome}: linha Take <{item}> não achada ({score:.2f})")
        elif nome in esperados:
            if ponto is None:
                falhas.append(f"menus/{nome}: linha Take <{item}> não achada ({score:.2f})")
            elif abs(ponto[1] - esperados[nome]) > 3:
                falhas.append(f"menus/{nome}: escolheu a linha em y={ponto[1]}, esperado {esperados[nome]}")
        elif ponto is not None:
            falhas.append(f"menus/{nome}: achou Take <{item}> onde não tem ({score:.2f} em {ponto})")
    print(f"Menus ({item}): {n}, {len(esperados)} com a linha Take")
    if n == 0:
        falhas.append(f"sem menus gravados em amostras/menus: o teste de {item} no menu não testou nada")
    return falhas


def testar_rotulo_vizinho():
    """Rótulo de outra pilha encostado ou encavalado no da runa (montado com rótulos reais).

    Antes o centro vinha da extensão da linha branca, que engolia o vizinho e tirava o
    clique da pilha (60-75 px). Agora: ou acha a pilha certa, ou ignora a runa. "Certa" =
    a até 12 px (um quarto do quadrado do chão, ~50 px: o menu da pilha abre de qualquer
    ponto do quadrado); encavalado 10 px por cima, o centro sai 7-9 px.
    """
    base = cv2.imread(os.path.join(PASTA, "runas", "230455_0051.png"))
    fonte = cv2.imread(os.path.join(PASTA, "152957_osso_ok_8.png"))
    if base is None or fonte is None:
        return ["sem as imagens do teste de rótulo vizinho (runas/230455_0051, 152957_osso_ok_8)"]
    certa = (942, 617)  # pilha da "Law rune x 2 (238 gp)" sem vizinho
    rot = fonte[411:433, 732:876]  # "Iron arrow x 3 (9 gp)"
    tinta = (rot.min(axis=2) > 160) | (rot.max(axis=2) < 25)  # letras e sombra
    binm, _ = visao.mascara_rotulos(base)
    linhas = np.flatnonzero(binm[591:613, 871:1020].any(axis=1))
    topo, esq, dir_ = 591 + linhas[0], 871, 1019  # extensão do rótulo da runa
    falhas, n, achou = [], 0, 0
    for lado in ("esquerda", "direita"):
        for folga in (-20, -10, 4, 6, 7, 8):  # negativo = encavalado
            for dy in (0, -5, 5):
                img = base.copy()
                x0 = esq - folga - rot.shape[1] if lado == "esquerda" else dir_ + folga + 1
                y0 = topo - 1 + dy
                reg = img[y0:y0 + rot.shape[0], x0:x0 + rot.shape[1]]
                reg[tinta] = rot[tinta]
                n += 1
                runas = [r for r in visao.detectar_itens_chao(img, "runa") if abs(r["rotulo"][1] - 593) <= 6]
                for r in runas:
                    achou += 1
                    if abs(r["pos"][0] - certa[0]) > 12 or abs(r["pos"][1] - certa[1]) > 12:
                        falhas.append(f"rótulo vizinho à {lado} (folga {folga}, dy {dy}): pilha em "
                                      f"{r['pos']}, certa {certa}")
    print(f"Rótulo vizinho: {n} montagens, runa achada no lugar certo em {achou} (nas outras, ignorada)")
    return falhas


def testar_menu_tem_take():
    """Menu em que o bot achou e clicou a linha (*_ok*.png) tem "Take"; os de clique em
    quadrado vazio gravados como falha (_sem, sem nenhum Take) não. É o que decide a
    segunda tentativa uma linha abaixo (main.pegar_pelo_menu_pilha)."""
    falhas, n = [], 0
    vazios = {"151931_002_sem.png", "151949_004_sem.png", "152559_020_sem.png"}
    for caminho in sorted(glob.glob(os.path.join(PASTA, "menus", "*.png"))):
        nome = os.path.basename(caminho)
        if "_ok" not in nome and nome not in vazios:
            continue
        n += 1
        tem = visao.menu_tem_take(cv2.imread(caminho))
        if tem != ("_ok" in nome):
            falhas.append(f"menus/{nome}: menu_tem_take = {tem}")
    print(f"Menu com/sem Take: {n} menus")
    return falhas


def testar_take_acao():
    """visao.tem_take_acao nas faixas do texto de ação: "Take <qualquer item>" sim; "Attack
    ..." e "Walk here" não. É o que guia a procura do item com o mouse (main.procurar_take)."""
    falhas, n = [], 0
    for caminho in sorted(glob.glob(os.path.join(PASTA, "faixas", "*.png"))):
        nome = os.path.basename(caminho)
        classe = nome.split("_")[1]
        if classe == "take-rej":
            continue  # "Take" de outro item por cima ou nada: não se sabe qual
        n += 1
        tem = visao.tem_take_acao(cv2.imread(caminho))
        if tem != (classe == "take-ok"):
            falhas.append(f"faixas/{nome}: tem_take_acao = {tem}")
    print(f"Take no texto de ação: {n} faixas")
    return falhas


def testar_rotulo_novo():
    """visao.rotulo_novo (loot de monstro sem "Big bones"): um rótulo de moedas real que
    aparece conta; o mesmo quadro sem mudança e um borrão branco do tamanho do número de um
    hitsplat (até 25 px) não contam."""
    caminho = os.path.join(PASTA, "runas", "esperado_moedas.json")
    if not os.path.isfile(caminho):
        return []
    with open(caminho, encoding="utf-8") as f:
        esperado = json.load(f)
    falhas, n = [], 0
    for arq, exp in sorted(esperado.items()):
        img = cv2.imread(os.path.join(PASTA, "runas", arq))
        if img is None or not exp["rotulos"]:
            continue
        binm = visao.mascara_rotulos(img)[0]
        for x, y in exp["rotulos"]:
            n += 1
            # o rótulo inteiro (a palavra "Coins" e o resto da linha à direita) ainda não existia
            antes = binm.copy()
            antes[max(y - 2, 0):y + 14, max(x - 4, 0):x + 90] = False
            if not visao.rotulo_novo(antes, binm):
                falhas.append(f"runas/{arq}: rótulo de moedas em {(x, y)} não contou como novo")
        if visao.rotulo_novo(binm, binm):
            falhas.append(f"runas/{arq}: quadro igual contou como rótulo novo")
        for largura in (12, 20, 25):
            agora = binm.copy()
            agora[600:612, 900:900 + largura] = True
            if visao.rotulo_novo(binm, agora):
                falhas.append(f"runas/{arq}: borrão de {largura} px contou como rótulo novo")
        # os rótulos do chão que se mexem um pouco (tela assentando) não são loot novo
        for dx, dy in ((0, 1), (2, 0), (4, 0)):
            mexido = np.zeros_like(binm)
            mexido[dy:, dx:] = binm[:binm.shape[0] - dy, :binm.shape[1] - dx]
            if visao.rotulo_novo(binm[:-8, :-8], mexido[:-8, :-8]):
                falhas.append(f"runas/{arq}: rótulos mexidos ({dx}, {dy}) px contaram como novos")
    # par real (1 s, nenhum item novo; o rótulo 'Limpwurt root' mudou 1-2 px): visto pela revisão
    a = cv2.imread(os.path.join(PASTA, "runas", "231124_0112.png"))
    b = cv2.imread(os.path.join(PASTA, "runas", "231125_0113.png"))
    cx, cy = visao.CAPTURA_CALIBRADA[0] // 2, visao.CAPTURA_CALIBRADA[1] // 2
    area = (slice(cy - 350, cy + 330), slice(cx - 390, cx + 330))  # o recorte de main.area_loot
    if a is not None and b is not None and visao.rotulo_novo(visao.mascara_rotulos(a[area])[0],
                                                             visao.mascara_rotulos(b[area])[0]):
        falhas.append("runas/231124_0112 -> 231125_0113: rótulo que mudou 1-2 px contou como novo")
    print(f"Rótulo novo (loot sem osso): {n} rótulos")
    return falhas


def testar_matiz_vermelho():
    """ensino.aprender_cor com monstro vermelho: o matiz do OpenCV dá a volta em 0/179. Gira o
    matiz de quadros reais de Hill Giant para a pele cair na volta e confere que o perfil
    aprendido acha todos os exemplos sem acender muito mais coisa que o aprendido sem giro."""
    import ensino
    caminho = os.path.join(PASTA, "rotulos.jsonl")
    if not os.path.isfile(caminho):
        return []
    with open(caminho, encoding="utf-8") as f:
        rots = [json.loads(linha) for linha in f if linha.strip()]
    rots = [r for r in rots if r["classe"] == "hillgiant_ok"]
    rots = rots[::max(len(rots) // 4, 1)][:4]
    exemplos = [(cv2.imread(os.path.join(PASTA, r["arquivo"])), tuple(r["pos"])) for r in rots]
    exemplos = [(img, pos) for img, pos in exemplos if img is not None]
    if len(exemplos) < 2:
        return []

    def girar(img, d):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + d) % 180).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def avaliar(exs):
        p = ensino.aprender_cor(exs)
        perfil = {"cor": p["cor"], "area": p["area"], "fechar": ensino.FECHAR, "desloc_matiz": p["desloc_matiz"]}
        achados, cands = 0, 0
        for img, pos in exs:
            c = visao.detectar_sapos(img, perfil)
            cands += len(c)
            achados += perto(pos, [s["pos"] for s in c], TOLERANCIA_SAPO)
        return p, achados, cands

    falhas = []
    _, _, cands_base = avaliar([(girar(img, 0), pos) for img, pos in exemplos])
    for d in (-10, -14, -17, 166):
        p, achados, cands = avaliar([(girar(img, d), pos) for img, pos in exemplos])
        if not p["desloc_matiz"]:
            falhas.append(f"matiz girado {d}: aprendeu sem girar o matiz (caixa {p['cor']})")
        if achados < len(exemplos):
            falhas.append(f"matiz girado {d}: achou {achados}/{len(exemplos)} exemplos")
        if cands > 1.5 * cands_base + 2:
            falhas.append(f"matiz girado {d}: {cands} candidatos contra {cands_base} sem giro")
    print(f"Monstro vermelho (matiz na volta 0/179): {len(exemplos)} exemplos, 4 giros")
    return falhas


def testar_reconexao():
    """reconexao: cada molde de botão em templates/login/ é achado na tela real dele (amostras/
    telas/<nome>.png), com o clique dentro do botão, e não aparece nas outras telas nem nos
    quadros de jogo (lá o bot não pode clicar em nada)."""
    import reconexao
    moldes = reconexao.carregar_moldes()
    pasta = os.path.join(PASTA, "telas")
    if not moldes or not os.path.isdir(pasta):
        return []
    falhas = []
    telas = {os.path.splitext(f)[0]: cv2.imread(os.path.join(pasta, f)) for f in os.listdir(pasta)}
    for nome, img in telas.items():
        achados = reconexao.achar_moldes(img, moldes)
        if nome in moldes and (not achados or achados[0][0] != nome):
            falhas.append(f"telas/{nome}: o botão {nome} não foi o achado ({achados})")
        outros = [a for a in achados if a[0] != nome]
        if outros:
            falhas.append(f"telas/{nome}: outro botão achado nela ({outros})")
    jogo = frames_jogo(passo=6)
    for caminho, img in jogo:
        achados = reconexao.achar_moldes(img, moldes)
        if achados:
            falhas.append(f"{os.path.basename(caminho)}: botão de login achado num quadro de jogo ({achados})")
    print(f"Volta ao jogo: {len(moldes)} botões, {len(telas)} telas, {len(jogo)} quadros de jogo")
    return falhas


# ---------- Perfis aprendidos (ensinar.py) ----------
def testar_perfis_aprendidos():
    """Cada perfis/<nome>.json: o molde do texto tem que recusar as faixas "Take Big bones"
    salvas em amostras/faixas (take-ok; as take-rej às vezes mostram "Attack Giant frog") e
    aceitar as faixas do próprio ensino; o perfil tem que achar o monstro em todos os quadros
    em que foi ensinado (perfis/<nome>_exemplos, que fica fora do git)."""
    import ensino  # só aqui: o resto dos testes não depende dele
    falhas, n_perfis, n_ex, sem_ex = [], 0, 0, []
    negativas = [cv2.imread(c) for c in sorted(glob.glob(os.path.join(PASTA, "faixas", "*take-ok*.png")))]
    negativas = [f for f in negativas if f is not None]
    # o gerador só aceita texto de ataque
    aceitas = sum(ensino.gerar_molde_texto(f)[0] is not None for f in negativas)
    if aceitas:
        falhas.append(f"ensino.gerar_molde_texto aceitou {aceitas} faixa(s) 'Take ...' sem 'Attack'")
    for nome in visao.perfis_disponiveis():
        if nome in visao.PERFIS and not visao.PERFIS[nome].get("aprendido"):
            continue
        perfil = visao.carregar_perfil(nome)
        n_perfis += 1
        limiar = visao.limiar_texto(perfil["texto"])
        pior = max((visao.score_texto(f, perfil["texto"]) for f in negativas), default=0.0)
        if pior >= limiar:
            falhas.append(f"perfis/{nome}: uma faixa 'Take ...' passa no molde ({pior:.3f})")
        pasta = os.path.join(visao.PASTA_PERFIS, f"{nome}_exemplos")
        lista = os.path.join(pasta, "exemplos.json")
        if not os.path.isfile(lista):
            sem_ex.append(nome)
            continue
        with open(lista, encoding="utf-8") as f:
            exemplos = json.load(f)
        for e in exemplos:
            img = cv2.imread(os.path.join(pasta, e["arquivo"]))
            n_ex += 1
            if not perto(e["pos"], [s["pos"] for s in visao.detectar_sapos(img, perfil)], TOLERANCIA_SAPO):
                falhas.append(f"perfis/{nome}: exemplo {e['arquivo']} em {e['pos']} não detectado")
        for c in sorted(glob.glob(os.path.join(pasta, "faixa_*.png"))):
            s = visao.score_texto(cv2.imread(c), perfil["texto"])
            if s < limiar:
                falhas.append(f"perfis/{nome}: {os.path.basename(c)} dá {s:.3f} no próprio molde")
    extra = f" (sem quadros de ensino, só o molde testado: {', '.join(sem_ex)})" if sem_ex else ""
    print(f"Perfis aprendidos: {n_perfis}, {n_ex} exemplos{extra}")
    return falhas


def testar_moldes_cruzados():
    """Moldes aprendidos sem o nível (visao.cortar_nivel): o corte foi feito (molde mais curto
    que o gravado; "Take Big bones", sem nível, fica inteiro); a faixa de cada monstro passa no
    próprio molde com outro nível (tudo depois do fim do molde apagado, onde ficam os dígitos);
    e não passa no molde de outro monstro (um corte curto demais aceitaria)."""
    falhas, n_faixas, cruzadas = [], 0, []
    if visao.cortar_nivel(visao.TEXTOS["take"]).shape != visao.TEXTOS["take"].shape:
        falhas.append("cortar_nivel cortou o molde 'Take Big bones', que não tem nível")
    perfis, faixas = {}, {}
    for nome in visao.perfis_disponiveis():
        p = visao.carregar_perfil(nome)
        if not p.get("aprendido"):
            continue
        perfis[nome] = p
        with open(os.path.join(visao.PASTA_PERFIS, f"{nome}.json"), encoding="utf-8") as f:
            molde_arq = json.load(f)["texto"]["molde"]
        gravado = cv2.imread(os.path.join(visao.PASTA_PERFIS, molde_arq), cv2.IMREAD_GRAYSCALE)
        tpl = visao.TEXTOS[p["texto"]]
        if gravado is not None and tpl.shape[1] >= gravado.shape[1]:
            falhas.append(f"perfis/{nome}: o nível não foi cortado do molde ({gravado.shape[1]} px)")
        else:  # o último traço do molde cortado é o hífen (até 3 px de altura), não um dígito
            t = tpl > 100
            fim = ini = np.flatnonzero(t.any(axis=0))[-1]
            while ini > 0 and t[:, ini - 1].any():
                ini -= 1
            linhas = np.flatnonzero(t[:, ini:fim + 1].any(axis=1))
            if linhas[-1] - linhas[0] > 2:
                falhas.append(f"perfis/{nome}: o molde cortado não termina no '-' de \"(level-\"")
        caminhos = sorted(glob.glob(os.path.join(visao.PASTA_PERFIS, f"{nome}_exemplos", "faixa_*.png")))
        lidas = [(os.path.basename(c), cv2.imread(c)) for c in caminhos]
        faixas[nome] = [(arq, f) for arq, f in lidas if f is not None]
    for a, pa in perfis.items():
        chave = pa["texto"]
        tpl, limiar = visao.TEXTOS[chave], visao.limiar_texto(chave)
        for arq, faixa in faixas[a]:
            n_faixas += 1
            # outro nível: apaga a faixa a partir de onde o molde termina nela
            m = visao.mascara_texto(faixa, chave in visao.TEXTO_NIVEL_QUALQUER_COR)
            _, _, _, loc = cv2.minMaxLoc(cv2.matchTemplate(m, tpl, cv2.TM_CCOEFF_NORMED))
            outro = faixa.copy()
            outro[:, loc[0] + tpl.shape[1]:] = 0
            s = visao.score_texto(outro, chave)
            if s < limiar:
                falhas.append(f"perfis/{a}: {arq} sem os dígitos do nível dá {s:.3f} (limiar {limiar})")
            for b, pb in perfis.items():
                if pb["nome"].lower() == pa["nome"].lower():
                    continue  # o mesmo monstro ensinado duas vezes
                s = visao.score_texto(faixa, pb["texto"])
                cruzadas.append(s)
                if s >= visao.limiar_texto(pb["texto"]):
                    falhas.append(f"perfis/{a}: {arq} passa no molde de {b} ({s:.3f})")
    if n_faixas:
        print(f"Moldes sem o nível: {len(perfis)} perfis, {n_faixas} faixas; outro nível aceito; "
              f"no molde de outro monstro <= {max(cruzadas, default=0.0):.3f} (limiar "
              f"{min(visao.limiar_texto(p['texto']) for p in perfis.values())})")
    else:
        print(f"Moldes sem o nível: {len(perfis)} perfis; faixas: sem dados (perfis/*_exemplos/)")
    return falhas


# ---------- Monstro pelo nome em ciano ("NPC Highlights") ----------
# amostras/nomes/<perfil>/: quadros copiados de logs/ com o monstro do perfil, e esperado.json com
# o centro de cada nome que o bot tem que achar (conferidos a olho; os nomes na faixa do texto
# de ação, no alto da tela, ficam de fora de propósito).
TOLERANCIA_NOME = 6  # px
TOLERANCIA_MIRA_NOME = 3


def caixa_mira(pos, raio, img):
    """O recorte de main.refinar_mira (tela.caixa): centrado em pos, sem sair do quadro."""
    x = min(max(int(pos[0]), raio), img.shape[1] - raio)
    y = min(max(int(pos[1]), raio), img.shape[0] - raio)
    return x - raio, y - raio, x + raio, y + raio


def testar_mira_por_nome():
    """Perfil com o nome do NPC (visao.detectar_por_nome): acha os nomes conferidos e nada a
    mais (o "Zombie" ao lado não conta); no recorte da correção da mira (main.refinar_mira),
    visao.sapo_mais_proximo devolve o mesmo ponto; e nos quadros de amostras/ (sem nomes em
    ciano) não acha nada."""
    pasta = os.path.join(PASTA, "nomes")
    gabaritos = sorted(glob.glob(os.path.join(pasta, "*", "esperado.json")))
    if not gabaritos:
        print("Monstro pelo nome: sem dados (amostras/nomes/<perfil>/esperado.json)")
        return []
    falhas, perfis, n_quadros, n_achados = [], [], 0, 0
    for caminho in gabaritos:
        nome = os.path.basename(os.path.dirname(caminho))
        try:
            perfil = visao.carregar_perfil(nome)
        except KeyError:
            falhas.append(f"nomes/{nome}: o perfil {nome} não existe")
            continue
        if perfil.get("nome_npc") is None:
            falhas.append(f"nomes/{nome}: o perfil não acha o monstro pelo nome")
            continue
        perfis.append(perfil)
        with open(caminho, encoding="utf-8") as f:
            esperado = json.load(f)
        for arq, nomes in esperado.items():
            img = cv2.imread(os.path.join(pasta, nome, arq))
            if img is None:
                falhas.append(f"nomes/{nome}/{arq}: imagem não encontrada")
                continue
            n_quadros += 1
            achados = visao.detectar_por_nome(img, perfil)
            for e in nomes:
                if not perto(e, [a["nome"] for a in achados], TOLERANCIA_NOME):
                    falhas.append(f"nomes/{nome}/{arq}: nome em {e} não achado")
            for a in achados:
                n_achados += 1
                if not perto(a["nome"], nomes, TOLERANCIA_NOME):
                    falhas.append(f"nomes/{nome}/{arq}: nome a mais em {tuple(map(int, a['nome']))}")
                x1, y1, x2, y2 = caixa_mira(a["pos"], perfil["raio_mira"], img)
                p, _ = visao.sapo_mais_proximo(img[y1:y2, x1:x2], (a["pos"][0] - x1, a["pos"][1] - y1), perfil)
                if p is None or not perto((p[0] + x1, p[1] + y1), [a["pos"]], TOLERANCIA_MIRA_NOME):
                    falhas.append(f"nomes/{nome}/{arq}: a mira em {a['pos']} foi para "
                                  f"{None if p is None else (p[0] + x1, p[1] + y1)}")
    frames = frames_jogo(passo=5)
    for perfil in perfis:
        for arq, img in frames:
            if visao.detectar_por_nome(img, perfil):
                falhas.append(f"{arq}: {perfil['nome']} achado pelo nome num quadro sem nomes em ciano")
    print(f"Monstro pelo nome: {n_achados} achados em {n_quadros} quadros, mira conferida em cada um; "
          f"{len(frames)} quadros sem nomes")
    return falhas


# ---------- Geometria da janela ----------
def testar_geometria():
    """Onde capturar dada a área do jogo na tela (medido: maximizada = (0, 29, 1920, 1121))."""
    casos = [
        ("maximizada (calibração)", (0, 29, 1920, 1121), (-9, -9), False),
        ("mesma janela movida", (300, 150, 1920, 1121), (291, 112), False),
        ("posição negativa (outro monitor)", (-1920, 29, 1920, 1121), (-1929, -9), False),
        ("restaurada, mais larga", (-45, 22, 1926, 1121), None, True),
        ("restaurada, menor", (100, 100, 1280, 720), None, True),
        ("1 px mais baixa", (0, 29, 1920, 1120), None, True),
        ("minimizada", (0, 0, 0, 0), None, True),
    ]
    falhas = []
    for nome, cliente, esperado, problema in casos:
        left, top, msg = visao.alinhar_captura(cliente)
        if (msg is not None) != problema:
            falhas.append(f"geometria ({nome}): problema esperado={problema}, obtido {msg!r}")
        if esperado is not None and (left, top) != esperado:
            falhas.append(f"geometria ({nome}): captura em {(left, top)}, esperado {esperado}")
    # A posição calibrada das coisas fixas tem que bater com a área do jogo
    cx, cy, cw, ch = visao.CLIENTE_CALIBRADO
    if (cx + cw + cx, cy + ch + cx) != visao.CAPTURA_CALIBRADA:
        falhas.append(f"geometria: captura {visao.CAPTURA_CALIBRADA} não fecha com a área "
                      f"do jogo {visao.CLIENTE_CALIBRADO} + bordas")
    print(f"Geometria da janela: {len(casos)} casos")
    return falhas


def testar_mochila_aberta():
    """visao.mochila_aberta: nos quadros de jogo a aba da mochila está aberta; nos de
    amostras/abas/ (outra aba aberta, como depois do F9 do ensino), não."""
    frames = frames_jogo(passo=5)
    if not frames:
        return [SEM_FRAMES]
    falhas = [f"{nome}: a aba da mochila aberta não foi reconhecida"
              for nome, img in frames if not visao.mochila_aberta(img)]
    outras = [(os.path.basename(c), cv2.imread(c)) for c in sorted(glob.glob(os.path.join(PASTA, "abas", "*.png")))]
    outras = [(nome, img) for nome, img in outras if img is not None]
    for nome, img in outras:
        if visao.mochila_aberta(img):
            falhas.append(f"abas/{nome}: outra aba aberta lida como a da mochila")
    extra = f"{len(outras)} com outra aba" if outras else "com outra aba: sem dados (amostras/abas/)"
    print(f"Aba da mochila: {len(frames)} quadros com ela aberta, {extra}")
    return falhas


def testar_semente():
    """Snape grass seed (item "semente"): não há foto do rótulo dela, então o teste monta
    rótulos com letras reais do quadro amostras/sementes/limpwurt_seed.png. Só "Snape grass
    seed" pode ser pega; as outras sementes, não. No menu, só a linha "Take" desconhecida."""
    caminho = os.path.join(PASTA, "sementes", "limpwurt_seed.png")
    img = cv2.imread(caminho)
    if img is None:
        print("Snape grass seed: sem dados (amostras/sementes/limpwurt_seed.png)")
        return []
    binm = visao.mascara_rotulos(img)[0]
    topo = lambda y1, y2, x1, x2: y1 + int(np.flatnonzero(binm[y1:y2, x1:x2].any(axis=1)).min())  # noqa: E731
    tops = {"limp": topo(494, 514, 870, 1000), "big": topo(516, 534, 878, 955), "earth": topo(560, 578, 1008, 1090)}
    letras = {  # letra: (linha de origem, x, largura); "S" = o "B" de "Big" (só a forma conta)
        "S": ("big", 884, 6), "n": ("big", 927, 6), "a": ("earth", 1021, 7), "p": ("limp", 898, 6),
        "e": ("limp", 949, 6), "g": ("big", 896, 7), "r": ("limp", 924, 4), "s": ("limp", 940, 7),
        "d": ("limp", 966, 7), "L": ("limp", 875, 6), "u": ("limp", 915, 7), "t": ("limp", 930, 4),
        "w": ("limp", 906, 7), "c": ("limp", 949, 5), "o": ("big", 918, 6), "i": ("limp", 883, 1),
        "m": ("limp", 887, 9), "R": ("big", 884, 6), "K": ("limp", 875, 6), "G": ("big", 884, 6),
        "M": ("limp", 887, 9),
    }

    # "anarr" real (de um "Grimy ranarr weed"), para montar "Ranarr seed": o caractere "#"
    outro = cv2.imread(os.path.join(PASTA, "sementes", "ranarr_marrentill.png"))
    anarr = None
    if outro is not None:
        b2 = visao.mascara_rotulos(outro)[0]
        anarr = b2[650:650 + 18, 1185:1222]

    # "Snape grass" real (16/11): o caractere "$"
    real = cv2.imread(os.path.join(PASTA, "sementes", "snape_real.png"))
    snape = None if real is None else visao.mascara_rotulos(real)[0][651:651 + 18, 848:934]

    def montar(texto, x0=700, y0=700, gap=2, espaco=7):
        out, x = img.copy(), x0
        for c in texto + " ":
            if c == " ":
                x += espaco - gap
                continue
            if c == "#":
                g, w = anarr, anarr.shape[1]
            elif c == "$":
                g, w = snape, snape.shape[1]
            else:
                linha, gx, w = letras[c]
                g = binm[tops[linha]:tops[linha] + 18, gx:gx + w]
            out[y0:y0 + 18, x:x + w][g] = (255, 255, 255)
            x += w + gap
        fim = binm[tops["limp"]:tops["limp"] + 18, 979:1025]  # o "(18 gp)" real
        out[y0:y0 + 18, x:x + fim.shape[1]][fim] = (255, 255, 255)
        return out, x + fim.shape[1]

    falhas = []
    if visao.detectar_itens_chao(img, "semente"):
        falhas.append("sementes: 'Limpwurt seed' real foi tomada por Snape grass seed")
    casos = [("$ seed", True), ("Limpwurt seed", False), ("Snape seed", False),
             ("grass seed", False), ("Swetcorn seed", False), ("Limpwurt grass seed", False),
             ("Snape Lass seed", False)]
    for gap in (2, 3):
        for texto, esperado in casos:
            quadro, x_fim = montar(texto, gap=gap)
            achados = [i for i in visao.detectar_itens_chao(quadro, "semente") if abs(i["rotulo"][1] - 699) <= 4]
            if bool(achados) != esperado:
                falhas.append(f"sementes: '{texto.replace('$', 'Snape grass')}' (folga {gap}) "
                              f"{'não foi pega' if esperado else 'foi pega'}")
            elif achados and abs(achados[0]["pos"][0] - ((700 + x_fim) // 2 - 3)) > 6:
                falhas.append(f"sementes: '{texto}' (folga {gap}) com a pilha fora do centro: {achados[0]['pos']}")
    # rótulo real (16/11): "Snape grass seed (8069 gp)" na pilha de um Big bones, com uma
    # "Jangerberry seed" perto; só a Snape grass seed, no lugar onde o bot a pegou (942, 695)
    if real is not None:
        a = visao.detectar_itens_chao(real, "semente")
        if len(a) != 1 or abs(a[0]["pos"][0] - 942) > 6 or abs(a[0]["pos"][1] - 695) > 6:
            falhas.append(f"sementes/snape_real.png: esperava 1 Snape grass seed em (942, 695), achou {a}")
        if visao.detectar_itens_chao(real, "semente_ranarr"):
            falhas.append("sementes/snape_real.png: Snape grass seed tomada por Ranarr seed")
    # Ranarr seed real (16/11, print do jogo: monitor inteiro, a captura começa 9 px antes): rótulo
    # AZUL-CLARO (item valioso) numa pilha com Steel bar e Harralander seed; o menu da pilha tem 3
    # "Take" e o bot tem que achar o "Take Ranarr seed"
    for nome in ("ranarr_tela1.png", "ranarr_tela2_menu.png"):
        mon = cv2.imread(os.path.join(PASTA, "sementes", nome))
        if mon is None:
            continue
        quadro = np.zeros((visao.CAPTURA_CALIBRADA[1], visao.CAPTURA_CALIBRADA[0], 3), np.uint8)
        quadro[9:, 9:9 + mon.shape[1]] = mon[:quadro.shape[0] - 9, :quadro.shape[1] - 9]
        a = visao.detectar_itens_chao(quadro, "semente_ranarr")
        if len(a) != 1 or abs(a[0]["pos"][0] - 829) > 8 or abs(a[0]["pos"][1] - 769) > 8:
            falhas.append(f"sementes/{nome}: esperava 1 Ranarr seed perto de (829, 769), achou {a}")
        if visao.detectar_itens_chao(quadro, "semente"):
            falhas.append(f"sementes/{nome}: Ranarr seed tomada por Snape grass seed")
        if nome == "ranarr_tela2_menu.png":
            p, nota = visao.achar_take_item_no_menu(quadro[759:984, 714:944], "semente_ranarr")
            if p is None or abs(p[1] - 65) > 4:
                falhas.append(f"sementes/{nome}: linha 'Take Ranarr seed' não achada no menu ({p}, {nota:.2f})")
    # menu real (16/11) de uma pilha Steel arrow + Snape grass seed + Big bones: a linha certa é a 2ª
    rec = cv2.imread(os.path.join(PASTA, "sementes", "snape_menu.png"))
    if rec is not None:
        p, _ = visao.achar_take_item_no_menu(rec, "semente")
        if p is None or abs(p[1] - 398) > 4:
            falhas.append(f"sementes/snape_menu.png: linha 'Take Snape grass seed' não achada ({p})")
        if visao.achar_take_item_no_menu(rec, "semente_ranarr")[0] is not None:
            falhas.append("sementes/snape_menu.png: achou 'Take Ranarr seed' num menu sem Ranarr")
    # rótulo real encavalado (16/11): "Snape grass see" + "Big bones (297 gp)" de outra pilha, com
    # letras 1 px para o lado; a semente estava em (937, 370)
    enc = cv2.imread(os.path.join(PASTA, "sementes", "snape_encavalada.png"))
    if enc is not None:
        a = visao.detectar_itens_chao(enc, "semente")
        if len(a) != 1 or abs(a[0]["pos"][0] - 937) > 8 or abs(a[0]["pos"][1] - 372) > 8:
            falhas.append(f"sementes/snape_encavalada.png: esperava 1 Snape grass seed perto de (937, 372), achou {a}")
    # Ranarr seed: "R" + o "anarr" real (folga entre letras 2-3 px, como a fonte do jogo; com 1 px nem o
    # "seed" passa); as parecidas não passam, e uma semente não passa pela outra
    n_ranarr = 0
    if anarr is not None:
        for gap in (2, 3):
            for texto, item, esperado in (("R# seed", "semente_ranarr", True), ("r# seed", "semente_ranarr", False),
                                          ("Kwuarm seed", "semente_ranarr", False), ("Guam seed", "semente_ranarr", False),
                                          ("$ seed", "semente_ranarr", False), ("R# seed", "semente", False)):
                quadro, x_fim = montar(texto, gap=gap)
                achados = [i for i in visao.detectar_itens_chao(quadro, item) if abs(i["rotulo"][1] - 699) <= 4]
                n_ranarr += 1
                nome = texto.replace("#", "anarr")
                if bool(achados) != esperado:
                    falhas.append(f"sementes: '{nome}' como {item} (folga {gap}) {'não foi pega' if esperado else 'foi pega'}")
                elif achados and abs(achados[0]["pos"][0] - ((700 + x_fim) // 2 - 3)) > 6:
                    falhas.append(f"sementes: '{nome}' (folga {gap}) com a pilha fora do centro: {achados[0]['pos']}")
    # quadro real com "Grimy ranarr weed", "Marrentill seed" e "Belladonna seed": nem runa nem semente
    if outro is not None:
        for item in ("runa", "semente", "semente_ranarr"):
            if visao.detectar_itens_chao(outro, item):
                falhas.append(f"sementes/ranarr_marrentill.png: achou '{item}' (ranarr/outras sementes)")
    # nenhuma semente nos quadros de runas e moedas
    outros = sorted(glob.glob(os.path.join(PASTA, "runas", "*.png")))
    for f in outros:
        quadro = cv2.imread(f)
        for item in ("semente", "semente_ranarr"):
            if visao.detectar_itens_chao(quadro, item):
                falhas.append(f"sementes: achou '{item}' em runas/{os.path.basename(f)}")
    # menu: os de osso/runa/moedas com outro item na pilha têm 1 linha desconhecida; sem "Take", nenhuma
    menus = sorted(glob.glob(os.path.join(PASTA, "menus", "*.png")))
    com = 0
    for f in menus:
        rec = cv2.imread(f)
        p, _ = visao.achar_take_desconhecido_no_menu(rec)
        com += p is not None
        if p is not None and not visao.menu_tem_take(rec):
            falhas.append(f"sementes: linha desconhecida num menu sem 'Take' (menus/{os.path.basename(f)})")
    print(f"Sementes: {2 * len(casos)} rótulos Snape e {n_ranarr} Ranarr montados, {len(outros)} quadros sem semente, "
          f"{len(menus)} menus ({com} com 1 linha 'Take' desconhecida)")
    return falhas


def testar_valiosos():
    """Itens de rótulo azul (detectar_valiosos_chao / achar_take_valioso_no_menu), sem molde do nome:
    a "Snapdragon seed (36K gp)" real de 16/11 é achada; a Ranarr seed (azul, mas com coletor
    próprio) e as Snape grass seeds (brancas) não; nenhum outro quadro das amostras tem rótulo
    azul. No menu, o nome lido no rótulo escolhe a linha certa e só ela."""
    falhas = []
    S = os.path.join(PASTA, "sementes")
    sd_img = cv2.imread(os.path.join(S, "snapdragon_tela.png"))
    rn_img = cv2.imread(os.path.join(S, "ranarr_tela2_menu.png"))  # monitor sem deslocar: menu e rótulos juntos
    if sd_img is None or rn_img is None:
        print("Valiosos: sem dados (amostras/sementes/snapdragon_tela.png e ranarr_tela2_menu.png)")
        return falhas
    sd = visao.detectar_valiosos_chao(sd_img)
    if len(sd) != 1 or abs(sd[0]["pos"][0] - 884) > 6 or abs(sd[0]["pos"][1] - 890) > 6:
        falhas.append(f"valiosos: esperava 1 Snapdragon seed perto de (884, 890), achou {[i['pos'] for i in sd]}")
    # pega ao vivo (16/11, no meio de uma luta): pilha Snapdragon seed + Limpwurt seed + Big bones
    pilha = cv2.imread(os.path.join(S, "snapdragon_pilha.png"))
    if pilha is not None:
        a = visao.detectar_valiosos_chao(pilha)
        if len(a) != 1 or abs(a[0]["pos"][0] - 831) > 6 or abs(a[0]["pos"][1] - 711) > 6:
            falhas.append(f"valiosos: esperava a pilha da Snapdragon em (831, 711), achou {[i['pos'] for i in a]}")
    if visao.detectar_valiosos_chao(rn_img):
        falhas.append("valiosos: a Ranarr seed (coletor próprio) entrou como item valioso genérico")
    # Ranarr pega ao vivo (17/11) com o rótulo azul encostado no "Mithril shortsword" de outra pilha,
    # numa pilha de 4 itens: o coletor dela acha; o genérico deixa para ele
    rp_img = cv2.imread(os.path.join(S, "ranarr_pilha.png"))
    if rp_img is not None:
        a = visao.detectar_itens_chao(rp_img, "semente_ranarr")
        if len(a) != 1 or abs(a[0]["pos"][0] - 932) > 6 or abs(a[0]["pos"][1] - 709) > 6:
            falhas.append(f"valiosos: Ranarr encavalada fora da pilha (932, 709): {[i['pos'] for i in a]}")
        if visao.detectar_valiosos_chao(rp_img):
            falhas.append("valiosos: a Ranarr encavalada (coletor próprio) entrou como item valioso genérico")
    # sem os coletores próprios, o mesmo rótulo azul desce pela pilha como o coletor dela (a descida
    # com o topo 1 px errado parava na 1ª linha), e o nome lido nele escolhe "Take Ranarr seed" (y=807)
    original = visao.detectar_itens_chao
    visao.detectar_itens_chao = lambda *a, **k: []
    try:
        rn = visao.detectar_valiosos_chao(rn_img)
        rp = visao.detectar_valiosos_chao(rp_img) if rp_img is not None else None
    finally:
        visao.detectar_itens_chao = original
    if rp is not None and (len(rp) != 1 or abs(rp[0]["pos"][0] - 932) > 6 or abs(rp[0]["pos"][1] - 709) > 6):
        falhas.append(f"valiosos: rótulo azul da Ranarr encavalada fora da pilha (932, 709): {[i['pos'] for i in rp]}")
    if len(rn) != 1 or abs(rn[0]["pos"][0] - 820) > 6 or abs(rn[0]["pos"][1] - 760) > 6:
        falhas.append(f"valiosos: rótulo azul da Ranarr fora da pilha dela (820, 760): {[i['pos'] for i in rn]}")
    else:
        p, nota = visao.achar_take_valioso_no_menu(rn_img, rn[0]["nome"])
        if p is None or abs(p[1] - 815) > 4:
            falhas.append(f"valiosos: nome da Ranarr não escolheu 'Take Ranarr seed' ({p}, {nota:.2f})")
    if sd:
        for nome in ("ranarr_tela2_menu.png", "snape_menu.png"):
            p, nota = visao.achar_take_valioso_no_menu(cv2.imread(os.path.join(S, nome)), sd[0]["nome"])
            if p is not None:
                falhas.append(f"valiosos: nome da Snapdragon escolheu uma linha em {nome} ({p}, {nota:.2f})")
    # nenhum rótulo azul nos outros quadros inteiros das amostras
    quadros = [f for f in sorted(glob.glob(os.path.join(PASTA, "**", "*.png"), recursive=True))
               if os.path.basename(f) not in ("snapdragon_tela.png", "snapdragon_pilha.png", "ranarr_pilha.png")]
    vistos = 0
    for f in quadros:
        img = cv2.imread(f)
        if img is None or img.shape[0] < 600 or img.shape[1] < 1000:
            continue  # recortes (menus, faixas, cliques)
        vistos += 1
        a = visao.detectar_valiosos_chao(img)
        if a:
            falhas.append(f"valiosos: rótulo azul inexistente em {os.path.relpath(f, PASTA)}: {[i['pos'] for i in a]}")
    print(f"Valiosos: Snapdragon real, Ranarr (pilha e menu) e {vistos} quadros sem rótulo azul")
    return falhas


# Na ordem da saída. Cada item: (função, argumentos...).
TESTES = [
    (testar_frames,), (testar_faixas,), (testar_cliques,), (testar_menus,),
    (testar_vida_amostras,), (testar_vida_baixa,), (testar_vida_deslocada,), (testar_vida_cenario,),
    (testar_vigia_vida,), (testar_geometria,), (testar_mochila_aberta,),
    *[t for item in GABARITOS_CHAO for t in ((testar_item_chao, item), (testar_item_menu, item))],
    (testar_semente,), (testar_valiosos,),
    (testar_rotulo_vizinho,), (testar_menu_tem_take,), (testar_take_acao,), (testar_rotulo_novo,),
    (testar_perfis_aprendidos,), (testar_moldes_cruzados,), (testar_mira_por_nome,),
    (testar_matiz_vermelho,), (testar_reconexao,),
]
# Os mais demorados (medido: 10-40 s cada; o resto, poucos segundos) começam primeiro.
LENTOS = (testar_item_chao, testar_rotulo_novo, testar_vida_cenario, testar_reconexao,
          testar_matiz_vermelho, testar_vida_amostras)
THREADS_POR_PROCESSO = 2  # o OpenCV de cada processo usaria todos os núcleos


def _args():
    p = argparse.ArgumentParser(description="Confere a detecção (visao.py) contra as amostras "
                                            "gravadas pelo bot em amostras/.")
    padrao = min(4, os.cpu_count() or 1)
    p.add_argument("-j", type=int, default=padrao, metavar="N",
                   help=f"testes em N processos ao mesmo tempo (padrão: {padrao}); "
                        "com o bot ligado, use -j 1")
    a = p.parse_args()
    if a.j < 1:
        p.error("-j precisa ser 1 ou mais")
    return a


def _iniciar_processo():
    cv2.setNumThreads(THREADS_POR_PROCESSO)


def _rodar(teste):
    """Roda um teste num processo à parte guardando o que ele imprime: (texto, falhas)."""
    fn, *args = teste
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            falhas = fn(*args)
        except Exception:  # um teste que quebra não derruba os outros
            print(traceback.format_exc(), end="")
            falhas = [f"{fn.__name__}({', '.join(map(repr, args))}): o teste quebrou (erro acima)"]
    return buf.getvalue(), falhas


def rodar_testes(testes, j):
    """Roda `testes` em j processos e devolve as falhas. A saída sai na ordem de `testes`;
    com j = 1, um teste por vez, imprimindo na hora (como antes)."""
    falhas = []
    if j == 1:
        for fn, *args in testes:
            falhas += fn(*args)
        return falhas
    ordem = sorted(testes, key=lambda t: LENTOS.index(t[0]) if t[0] in LENTOS else len(LENTOS))
    with ProcessPoolExecutor(j, initializer=_iniciar_processo) as ex:
        futuros = {t: ex.submit(_rodar, t) for t in ordem}
        for t in testes:
            texto, f = futuros[t].result()
            print(texto, end="", flush=True)
            falhas += f
    return falhas


if __name__ == "__main__":
    ARGS = _args()
    if not os.path.isdir(PASTA):
        sys.exit("Sem amostras: rode `python main.py --amostras` primeiro.")
    falhas = rodar_testes(TESTES, ARGS.j)
    if falhas:
        print(f"\n{len(falhas)} divergência(s):")
        for f in falhas:
            print("  -", f)
        sys.exit(1)
    print("\nTudo certo: a detecção bate com todas as amostras.")
