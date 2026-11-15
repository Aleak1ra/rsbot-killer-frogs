"""Aprende o perfil de um monstro novo a partir de exemplos: (quadro, ponto do mouse sobre ele).

Só funções sobre imagens: não lê a tela nem mexe no mouse (quem faz isso é ensinar.py).
O perfil gerado é o mesmo molde dos feitos à mão em visao.PERFIS (cor, área, barra, mira...),
gravado em perfis/<nome>.json e carregado por visao.carregar_perfil.

Medido offline nas amostras de hoje (Hill Giant e Giant frog reaprendidos assim):
- texto: o molde gerado de UMA faixa de ataque dá >= 0,994 nas faixas certas e nenhum falso a
  0,95 (353 faixas);
- cor ("varinha mágica" em volta do ponto): 1 clique acha 96% (mediana) dos Hill Giants
  confirmados; 2 cliques, um em cada tom de pele, 100% (pior caso 86%);
- barra/mira deduzidas do tamanho do monstro: a decisão "já está lutando" bate com a feita à
  mão em 98,7% de 536 candidatos.
"""
import json
import os

import cv2
import numpy as np

import visao

# ---------- Texto de ação ----------
TOPO_FAIXA = 13  # linhas de cima da faixa (FAIXA_TEXTO_ACAO) que são da barra de título
BURACO_TEXTO = 14  # px sem tinta que separam o texto de ação do " / N more options"
# Palavra "Attack" (branca, a 1ª do texto de ação), média de 112 faixas de ataque. Medido em
# 353 faixas: "Attack ..." >= 0,999; "Take ...", "Walk here" etc. <= 0,64.
ATAQUE = visao._carregar("text_mask_attack_osrs.png", cinza=True)
ATAQUE_LIMIAR = 0.9
LIMIAR_TEXTO_APRENDIDO = 0.95


def _linha_texto(tinta):
    """(y1, y2) do primeiro bloco de linhas com tinta, ou None."""
    ys = np.flatnonzero(tinta.any(axis=1))
    if not len(ys):
        return None
    y1 = y2 = int(ys[0])
    while y2 + 1 < tinta.shape[0] and tinta[y2 + 1].any():
        y2 += 1
    return y1, y2


def gerar_molde_texto(faixa, com_nivel=True):
    """Molde do texto de ação "Attack <Nome> (level-N)" a partir de UMA faixa.

    `faixa`: recorte FAIXA_TEXTO_ACAO com o mouse sobre o monstro. Devolve (molde, nome_px,
    motivo): molde em visao.mascara_texto_geral (None se a faixa não serve; `motivo` diz por
    quê). Com com_nivel=False o molde para no nome (aceita o monstro em qualquer nível).
    """
    m = visao.mascara_texto_geral(faixa)
    tinta = m > 100
    tinta[:TOPO_FAIXA] = False
    linha = _linha_texto(tinta)
    if linha is None:
        return None, "nenhum texto de ação na tela"
    y1, y2 = linha
    p = faixa.astype(np.int16)
    branco = p.min(axis=2) > 200
    colorido = tinta & ~cv2.dilate(branco.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    cols = np.flatnonzero(tinta[y1:y2 + 1].any(axis=0))
    # o texto de ação é o 1º trecho contínuo (o " / 2 more options" vem depois de um buraco)
    fim_trecho = cols[0]
    for c in cols[1:]:
        if c - fim_trecho > BURACO_TEXTO:
            break
        fim_trecho = c
    cc = np.flatnonzero(colorido[y1:y2 + 1, cols[0]:fim_trecho + 1].any(axis=0)) + cols[0]
    if not len(cc):
        return None, "texto sem nome colorido (não é um monstro)"
    verbo = m[:, :int(cc[0])]
    if verbo.shape[1] < ATAQUE.shape[1] or \
            float(cv2.matchTemplate(verbo, ATAQUE, cv2.TM_CCOEFF_NORMED).max()) < ATAQUE_LIMIAR:
        return None, "o texto não começa com 'Attack' (o mouse está sobre o monstro?)"
    x_fim = int(cc[-1])
    if not com_nivel:
        # só o nome (amarelo), sem o "(level-N)", que tem outra cor
        b, g, r = p[:, :, 0], p[:, :, 1], p[:, :, 2]
        amarelo = (r > 200) & (g > 200) & (b < 90)
        ca = np.flatnonzero(amarelo[y1:y2 + 1, cols[0]:fim_trecho + 1].any(axis=0)) + cols[0]
        if len(ca):
            x_fim = int(ca[-1])
    molde = m[max(y1 - 2, 0):min(y2 + 3, m.shape[0]), max(int(cols[0]) - 2, 0):x_fim + 3].copy()
    return molde, None


def score_molde(faixa, molde):
    mm = visao.mascara_texto_geral(faixa)
    if mm.shape[0] < molde.shape[0] or mm.shape[1] < molde.shape[1]:
        return 0.0
    return float(cv2.matchTemplate(mm, molde, cv2.TM_CCOEFF_NORMED).max())


# ---------- Cor do corpo ("varinha mágica") ----------
R_MIOLO = 7  # px: disco em volta do ponto, que o texto de ação garantiu estar sobre o corpo
FOLGA_MATIZ = 3  # o mesmo monstro varia alguns tons de matiz (Hill Giant: 13-14 e 16-17)
VAZAMENTO_MAX = 0.002  # fração da área 3D em manchas enormes (o chão) tolerada
VAZAMENTO_FATOR = 4.0  # mancha maior que isso x a área do monstro conta como vazamento
MARGEM_CAIXA = (1, 6, 10)  # folga H, S, V em volta dos percentis 2-98 da região
FECHAR = 5
FATOR_AREA = (0.25, 4.0)
MATIZ_BORDA = 10  # cor a até isso da volta 0/179 do matiz (vermelhos): aprende com o matiz girado


def mascara_3d(shape):
    """255 onde há mundo 3D (área do jogo menos a interface)."""
    m = np.zeros(shape[:2], np.uint8)
    cx, cy, cw, ch = visao.CLIENTE_CALIBRADO
    m[cy:cy + ch, cx:cx + cw] = 255
    for x1, y1, x2, y2 in visao.AREAS_INTERFACE:
        m[y1:y2, x1:x2] = 0
    return m


def _comp_no_ponto(mask, pos, raio=12):
    """(máscara bool do componente que contém `pos` ou o mais perto a até `raio` px, stats)."""
    _, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x, y = pos
    x0, y0 = max(x - raio, 0), max(y - raio, 0)
    win = lab[y0:y + raio + 1, x0:x + raio + 1]
    ys, xs = np.nonzero(win)
    if not len(xs):
        return None, None
    i = int(np.argmin(np.hypot(xs + x0 - x, ys + y0 - y)))
    k = win[ys[i], xs[i]]
    return lab == k, st[k]


def _mascara(img, cor, fechar, desloc=0):
    return visao.mascara_monstro(img, {"cor": cor, "fechar": fechar, "desloc_matiz": desloc})


def _desloc_matiz(exemplos):
    """Giro do matiz para aprender monstros vermelhos: o matiz do OpenCV dá a volta em 0/179, e
    uma caixa HSV não atravessa essa borda. Se a cor dos exemplos estiver a até MATIZ_BORDA
    da volta, aprende (e detecta) com o matiz girado para pôr essa cor no meio da escala."""
    angulos = []
    yy, xx = np.ogrid[-R_MIOLO:R_MIOLO + 1, -R_MIOLO:R_MIOLO + 1]
    disco = (xx * xx + yy * yy) <= R_MIOLO * R_MIOLO
    for img, (x, y) in exemplos:
        miolo = img[y - R_MIOLO:y + R_MIOLO + 1, x - R_MIOLO:x + R_MIOLO + 1]
        if miolo.shape[:2] != disco.shape:
            continue
        h = cv2.cvtColor(miolo, cv2.COLOR_BGR2HSV)[..., 0][disco].astype(float)
        angulos.append(h * np.pi / 90.0)
    if not angulos:
        return 0
    a = np.concatenate(angulos)
    h0 = (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2.0) % 180.0
    if min(h0, 180.0 - h0) >= MATIZ_BORDA:
        return 0
    return int(round(90 - h0)) % 180


def _regiao_semente(hsv, pos, m3):
    """Pixels HSV da região de cor parecida ligada ao ponto, e (h0, s0, tamanho)."""
    x, y = pos
    yy, xx = np.ogrid[-R_MIOLO:R_MIOLO + 1, -R_MIOLO:R_MIOLO + 1]
    disco = (xx * xx + yy * yy) <= R_MIOLO * R_MIOLO
    miolo = hsv[y - R_MIOLO:y + R_MIOLO + 1, x - R_MIOLO:x + R_MIOLO + 1]
    if miolo.shape[:2] != disco.shape:
        return None, None
    core = miolo[disco].astype(int)
    h0, s0 = np.median(core[:, 0]), np.median(core[:, 1])
    # só o grupo de cor dominante do miolo (tira olho, roupa, chão que aparece)
    core = core[(np.abs(core[:, 0] - h0) <= 3) & (np.abs(core[:, 1] - s0) <= 30)]
    dh = max(3, int(np.ceil(np.percentile(np.abs(core[:, 0] - h0), 90))) + 1)
    ds = max(14, int(np.ceil(np.percentile(np.abs(core[:, 1] - s0), 90))) + 6)
    vmin = max(20, int(np.percentile(core[:, 2], 5)) - 60)
    H, S, V = (hsv[..., i].astype(int) for i in range(3))
    cand = ((np.abs(H - h0) <= dh) & (np.abs(S - s0) <= ds) & (V >= vmin)).astype(np.uint8) * 255
    cand &= m3
    reg, _ = _comp_no_ponto(cand, pos)
    if reg is None:
        return None, None
    return hsv[reg], (float(h0), float(s0), int(reg.sum()))


def _vazamento(hsv, cor, a_ref, m3, m3_area):
    """(fração da área 3D em manchas enormes, pixels HSV dessas manchas dentro da caixa).
    Só o mundo 3D conta (a interface não entra na detecção)."""
    cru = cv2.inRange(hsv, np.asarray(cor[0]), np.asarray(cor[1])) & m3
    m = cv2.morphologyEx(cru, cv2.MORPH_CLOSE, np.ones((FECHAR, FECHAR), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    grandes = [k for k in range(1, n) if st[k, 4] > VAZAMENTO_FATOR * a_ref]
    if not grandes:
        return 0.0, np.empty((0, 3), int)
    sel = np.isin(lab, grandes) & (cru > 0)
    return float(sum(int(st[k, 4]) for k in grandes)) / m3_area, hsv[sel].astype(int)


def _apertar(hsvs, regs, lo, hi, a_ref, m3, m3_area, travar_matiz=False):
    """Estreita a caixa enquanto ela acender manchas enormes (o chão) em algum exemplo.

    A cada passo corta a fatia (um lado de H, S ou V) que tira mais pixels das manchas por
    pixel perdido da região de cada exemplo (o pior exemplo manda: um monstro mais claro
    ou mais escuro que os outros não é sacrificado). Devolve (lo, hi, vazamento)."""
    lo, hi = lo.copy(), hi.copy()
    vz = 0.0
    for _ in range(80):
        res = [_vazamento(hsv, (lo, hi), a_ref, m3, m3_area) for hsv in hsvs]
        vz = max(r[0] for r in res)
        if vz <= VAZAMENTO_MAX:
            break
        vaz = np.concatenate([r[1] for r in res])
        melhor, melhor_nota = None, 0.0
        for d in range(3):
            if d == 0 and travar_matiz:
                continue
            passo = (1, 4, 8)[d]
            for lado in (0, 1):
                novo = lo[d] + passo if lado == 0 else hi[d] - passo
                if (lado == 0 and novo > hi[d]) or (lado == 1 and novo < lo[d]):
                    continue
                def sai(v, novo=novo, lado=lado):
                    """Fração de `v` que a fatia cortada tira."""
                    return float(((v < novo) if lado == 0 else (v > novo)).mean()) if len(v) else 0.0

                sai_vaz = sai(vaz[:, d])
                sai_reg = max(sai(r[:, d]) for r in regs)
                nota = sai_vaz / (sai_reg + 0.01)
                if nota > melhor_nota:
                    melhor, melhor_nota = (d, lado, novo), nota
        if melhor is None:
            break
        d, lado, novo = melhor
        if lado == 0:
            lo[d] = novo
        else:
            hi[d] = novo
    return lo, hi, vz


def _areas(exemplos, cor, desloc=0):
    areas, caixas = [], []
    for img, pos in exemplos:
        comp, st = _comp_no_ponto(_mascara(img, cor, FECHAR, desloc), pos)
        if comp is not None:
            areas.append(int(st[4]))
            caixas.append((int(st[2]), int(st[3])))
    return areas, caixas


def aprender_cor(exemplos):
    """Faixa HSV, área aceita e tamanho típico do monstro a partir de exemplos (img, (x, y)).

    Um exemplo por aparência diferente do monstro (ex.: os dois tons de pele do Hill Giant).
    Devolve dict com cor (lo, hi), area (min, max), area_ref, caixa_ref (w, h), vazamento e
    usados (quantos exemplos deram região), ou None se nenhum exemplo serviu.
    """
    if not exemplos:
        return None
    m3 = mascara_3d(exemplos[0][0].shape)
    m3_area = float((m3 > 0).sum())
    desloc = _desloc_matiz(exemplos)
    hsvs = [visao.hsv_monstro(img, desloc) for img, _ in exemplos]
    regs, info = [], []
    for hsv, (_, pos) in zip(hsvs, exemplos):
        r, i = _regiao_semente(hsv, pos, m3)
        if r is not None:
            regs.append(r)
            info.append(i)
    if not regs:
        return None
    px = np.concatenate(regs)
    lo = np.clip(np.floor(np.percentile(px, 2, axis=0) - MARGEM_CAIXA), 0, 255).astype(int)
    hi = np.clip(np.ceil(np.percentile(px, 98, axis=0) + MARGEM_CAIXA), 0, 255).astype(int)
    hi[0] = min(hi[0], 179)
    a_reg = float(np.median([i[2] for i in info]))
    regs = [r.astype(int) for r in regs]

    lo, hi, vz = _apertar(hsvs, regs, lo, hi, a_reg, m3, m3_area)
    areas, _ = _areas(exemplos, (lo, hi), desloc)
    a_comp = float(np.median(areas)) if areas else a_reg
    # folga de matiz em volta das cores-semente (tons do mesmo monstro que nenhum exemplo
    # mostrou), depois aperta só S e V
    h0s = [i[0] for i in info]
    lo[0] = max(0, min(lo[0], int(min(h0s)) - FOLGA_MATIZ))
    hi[0] = min(179, max(hi[0], int(max(h0s)) + FOLGA_MATIZ))
    lo, hi, vz = _apertar(hsvs, regs, lo, hi, a_comp, m3, m3_area, travar_matiz=True)
    areas, caixas = _areas(exemplos, (lo, hi), desloc)
    a_comp = float(np.median(areas)) if areas else a_reg
    # a componente da máscara pode vazar para o que encosta no monstro: limita pela região
    a = min(a_comp, max(a_reg, 1.0) * 1.5)
    caixa = tuple(int(v) for v in np.median(caixas, axis=0)) if caixas else (40, 40)
    return {"cor": (lo, hi), "area": (int(a * FATOR_AREA[0]), int(a * FATOR_AREA[1])),
            "area_ref": a, "caixa_ref": caixa, "vazamento": round(vz, 5), "usados": len(regs),
            "desloc_matiz": desloc}


def deduzir_parametros(caixa_ref, area_ref):
    """Barra, mira e área de mira proporcionais ao tamanho do monstro na tela.

    Conferido com os perfis feitos à mão: Hill Giant (75, 120, 45) -> (72, 116, 34) e sapo
    (45, 85, 10) -> (64, 82, 24), a mesma decisão "ocupado" em 98,7% e 99,3% dos candidatos;
    raio da mira 110 e 87 (à mão 110 e 70)."""
    w, h = caixa_ref
    return {
        "barra": (int(max(45, round(1.4 * w))), int(round(2.0 * h)), int(round(0.6 * h))),
        "raio_mira": int(max(70, round(1.9 * max(w, h)))),
        "area_mira": int(max(50, round(0.25 * area_ref))),
    }


# ---------- Perfil em arquivo ----------
def montar_perfil(nome_jogo, molde, exemplos, espera_loot=4.5, loot_menu=True, osso="Big bones"):
    """Dict do perfil (formato de perfis/<nome>.json, sem o molde) ou None.

    `osso`: "Big bones" (o único osso que o bot conhece: pega, enterra e espera por ele) ou
    None (monstro que larga outro osso ou nenhum: o bot só pega runas e moedas)."""
    cor = aprender_cor(exemplos)
    if cor is None:
        return None
    perfil = {
        "nome": nome_jogo,
        "cor": [list(map(int, cor["cor"][0])), list(map(int, cor["cor"][1]))],
        "area": list(cor["area"]),
        "fechar": FECHAR,
        "espera_loot": espera_loot,
        "loot_menu": loot_menu,
        "osso": osso,
        "desloc_matiz": cor["desloc_matiz"],
        "texto": {"limiar": LIMIAR_TEXTO_APRENDIDO, "altura": int(molde.shape[0]), "largura": int(molde.shape[1])},
        "aprendido": {"exemplos": len(exemplos), "area_ref": round(cor["area_ref"], 1),
                      "caixa_ref": list(cor["caixa_ref"]), "vazamento": cor["vazamento"]},
    }
    p = deduzir_parametros(cor["caixa_ref"], cor["area_ref"])
    perfil.update({k: list(v) if isinstance(v, tuple) else v for k, v in p.items()})
    return perfil


def salvar_perfil(nome, perfil, molde, exemplos=(), faixas=(), pasta=None):
    """Grava perfis/<nome>.json, o molde do texto e os quadros e faixas de ensino (para os
    testes: testar_visao.testar_perfis_aprendidos)."""
    pasta = pasta or visao.PASTA_PERFIS
    os.makedirs(pasta, exist_ok=True)
    perfil = dict(perfil, texto=dict(perfil["texto"], molde=f"{nome}_texto.png"))
    cv2.imwrite(os.path.join(pasta, f"{nome}_texto.png"), molde)
    if exemplos:
        pasta_ex = os.path.join(pasta, f"{nome}_exemplos")
        os.makedirs(pasta_ex, exist_ok=True)
        lista = []
        for i, (img, pos) in enumerate(exemplos, 1):
            arq = f"{i:02d}.png"
            cv2.imwrite(os.path.join(pasta_ex, arq), img)
            lista.append({"arquivo": arq, "pos": [int(pos[0]), int(pos[1])]})
        for i, faixa in enumerate(faixas, 1):
            cv2.imwrite(os.path.join(pasta_ex, f"faixa_{i:02d}.png"), faixa)
        perfil["exemplos"] = f"{nome}_exemplos"
        with open(os.path.join(pasta_ex, "exemplos.json"), "w", encoding="utf-8") as f:
            json.dump(lista, f, ensure_ascii=False, indent=1)
    caminho = os.path.join(pasta, f"{nome}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(perfil, f, ensure_ascii=False, indent=2)
    return caminho


def marcar_candidatos(img, perfil, exemplos=()):
    """Cópia do quadro com os candidatos do perfil (círculo verde) e os exemplos (x vermelho)."""
    out = img.copy()
    for s in visao.detectar_sapos(img, perfil):
        cv2.circle(out, s["pos"], 18, (0, 255, 0), 2)
    for _, pos in exemplos:
        cv2.drawMarker(out, tuple(map(int, pos)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    return out
