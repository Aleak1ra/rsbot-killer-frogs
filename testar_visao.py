"""Teste de regressão da detecção, usando as amostras gravadas pelo bot.

  python main.py --amostras     # grava frames em amostras/ enquanto joga
  python testar_visao.py        # confere a detecção contra essas amostras

Cada amostra foi rotulada pelo próprio jogo: o bot só grava "sapo_ok"/"osso_ok"
depois que o texto de ação confirmou o que havia naquela posição. Rode este
teste sempre que mexer em visao.py ou trocar um template.
"""
import glob
import json
import os
import sys

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
    margens = {c: [] for c in visao.TEXTOS}
    for caminho in sorted(glob.glob(os.path.join(PASTA, "faixas", "*.png"))):
        nome = os.path.basename(caminho)
        classe = nome.split("_")[1]
        if classe not in esperado:
            continue
        faixa = cv2.imread(caminho)
        scores = {c: visao.score_texto(faixa, c) for c in visao.TEXTOS}
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
    for caminho in sorted(glob.glob(os.path.join(PASTA, "menus", "*_ok.png"))):
        n += 1
        ponto, score = visao.achar_take_no_menu(cv2.imread(caminho))
        if ponto is None:
            falhas.append(f"{os.path.basename(caminho)}: linha Take Big bones não achada ({score:.2f})")
    print(f"Menus: {n} com a linha \"Take Big bones\"")
    return falhas


if __name__ == "__main__":
    if not os.path.isdir(PASTA):
        sys.exit("Sem amostras: rode `python main.py --amostras` primeiro.")
    falhas = testar_frames() + testar_faixas() + testar_cliques() + testar_menus()
    if falhas:
        print(f"\n{len(falhas)} divergência(s):")
        for f in falhas:
            print("  -", f)
        sys.exit(1)
    print("\nTudo certo: a detecção bate com todas as amostras.")
