"""Aprende o perfil de um monstro novo a partir de exemplos: (quadro, ponto do mouse sobre ele).

Não lê a tela nem mexe no mouse (quem faz isso é main.py, no F9 do bot, ou ensinar.py): as
funções trabalham sobre imagens, e a classe Ensino junta os exemplos e grava o perfil.
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
import shutil

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
LIMIAR_TEXTO_APRENDIDO = visao.LIMIAR_TEXTO_SEM_NIVEL  # o molde perde os dígitos do nível ao carregar


def _linha_texto(tinta):
    """(y1, y2) do primeiro bloco de linhas com tinta, ou None."""
    ys = np.flatnonzero(tinta.any(axis=1))
    if not len(ys):
        return None
    y1 = y2 = int(ys[0])
    while y2 + 1 < tinta.shape[0] and tinta[y2 + 1].any():
        y2 += 1
    return y1, y2


def gerar_molde_texto(faixa):
    """Molde do texto de ação "Attack <Nome> (level-N)" a partir de UMA faixa.

    `faixa`: recorte FAIXA_TEXTO_ACAO com o mouse sobre o monstro. Devolve (molde, motivo):
    molde em visao.mascara_texto(faixa, nivel_qualquer_cor=True), ou None se a faixa não serve
    (`motivo` diz por quê).
    O molde vai até o fim do "(level-N)": parar no nome aceitaria outro monstro cujo nome começa
    igual ("Cow" pegaria "Cow calf"), visto na revisão. Os dígitos do nível são tirados ao usar
    (visao.cortar_nivel): o mesmo monstro de outro nível também serve.
    """
    m = visao.mascara_texto(faixa, True)
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
        return None, ("o texto do canto não começa com 'Attack': ponha o mouse em cima do monstro; se "
                      "aparecer 'Walk here', mude a opção NPC 'Attack' options para 'Left-click "
                      "where available' nas configurações do jogo")
    x_fim = int(cc[-1])
    molde = m[max(y1 - 2, 0):min(y2 + 3, m.shape[0]), max(int(cols[0]) - 2, 0):x_fim + 3].copy()
    return molde, None


def score_molde(faixa, molde):
    """Nota da faixa no molde como o bot vai usar (sem os dígitos do nível: visao.cortar_nivel)."""
    return visao.nota_molde(faixa, visao.cortar_nivel(molde), True)


# ---------- Cor do corpo ("varinha mágica") ----------
R_MIOLO = 7  # px: disco em volta do ponto, que o texto de ação garantiu estar sobre o corpo
FOLGA_MATIZ = 3  # o mesmo monstro varia alguns tons de matiz (Hill Giant: 13-14 e 16-17)
VAZAMENTO_MAX = 0.002  # fração da área 3D em manchas enormes (o chão) tolerada
VAZAMENTO_FATOR = 4.0  # mancha maior que isso x a área do monstro conta como vazamento
MARGEM_CAIXA = (1, 6, 10)  # folga H, S, V em volta dos percentis 2-98 da região
FECHAR = 5
FATOR_AREA = (0.25, 4.0)
S_CINZA = 40  # saturação abaixo disso: cor quase cinza, o matiz não decide
S_GRUPO = 20  # ±S em volta da semente: com 30 o chão encostado alargava a região (revisão)
REGIAO_MIN = 40  # px: região de cor menor que isso não serve de exemplo
REGIAO_MAX = 0.04  # fração da área 3D: região maior que isso é o chão, não o monstro
AREA_REF_MIN = 200  # px: piso da área de referência (a mínima aceita fica >= 50 px)
AMOSTRA_REGIAO = 2000  # px por exemplo nos percentis da caixa
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


def _regiao_semente(hsv, pos, m3, m3_area):
    """Pixels HSV da região de cor parecida ligada ao ponto, e (h0, s0, tamanho); (None,
    motivo) se o ponto não serve.

    Cor-semente: a dominante num disco de R_MIOLO px em volta do ponto (resiste a cair num
    detalhe, como a tanga do gigante). Se essa cor se espalhar pela tela (clique na borda do
    corpo: o disco é quase todo chão) ou não formar região, tenta a cor sob o cursor (5x5),
    que o texto de ação garante ser do monstro. Em monstro cinza o matiz é ruído e não conta."""
    x, y = pos
    alt, larg = hsv.shape[:2]
    if not (R_MIOLO <= x < larg - R_MIOLO and R_MIOLO <= y < alt - R_MIOLO):
        return None, "ponto na borda da tela"
    yy, xx = np.ogrid[-R_MIOLO:R_MIOLO + 1, -R_MIOLO:R_MIOLO + 1]
    disco = (xx * xx + yy * yy) <= R_MIOLO * R_MIOLO
    miolo = hsv[y - R_MIOLO:y + R_MIOLO + 1, x - R_MIOLO:x + R_MIOLO + 1][disco].astype(int)
    centro = hsv[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3).astype(int)
    H, S, V = (hsv[..., i].astype(int) for i in range(3))
    motivo = "cor do cursor sem vizinhos parecidos (mouse na borda do monstro?)"
    for semente in (miolo, centro):
        h0, s0 = float(np.median(semente[:, 0])), float(np.median(semente[:, 1]))
        cinza = s0 < S_CINZA
        perto = np.abs(miolo[:, 1] - s0) <= S_GRUPO
        if not cinza:
            perto &= np.abs(miolo[:, 0] - h0) <= 3
        core = miolo[perto]
        if len(core) < 10:
            continue
        dh = 90 if cinza else max(3, int(np.ceil(np.percentile(np.abs(core[:, 0] - h0), 90))) + 1)
        ds = max(14, int(np.ceil(np.percentile(np.abs(core[:, 1] - s0), 90))) + 6)
        vmin = max(20, int(np.percentile(core[:, 2], 5)) - 60)
        cand = ((np.abs(H - h0) <= dh) & (np.abs(S - s0) <= ds) & (V >= vmin)).astype(np.uint8) * 255
        cand &= m3
        reg, _ = _comp_no_ponto(cand, pos)
        n = 0 if reg is None else int(reg.sum())
        if n < REGIAO_MIN:
            motivo = f"região de cor pequena demais ({n} px): ponha o mouse no meio do corpo"
            continue
        if n > REGIAO_MAX * m3_area:
            motivo = "a cor do ponto se espalha pela tela (chão?): ponha o mouse no meio do corpo"
            continue
        return hsv[reg], (h0, s0, n)
    return None, motivo


def checar_exemplo(img, pos):
    """None se o ponto (mouse sobre o monstro) serve de exemplo de cor; senão o motivo."""
    m3 = mascara_3d(img.shape)
    hsv = visao.hsv_monstro(img, _desloc_matiz([(img, pos)]))
    r, motivo = _regiao_semente(hsv, pos, m3, float((m3 > 0).sum()))
    return None if r is not None else motivo


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
        r, i = _regiao_semente(hsv, pos, m3, m3_area)
        if r is not None:
            regs.append(r)
            info.append(i)
    if not regs:
        return None
    # cada exemplo pesa igual nos percentis (uma região grande não manda sozinha na caixa)
    sorteio = np.random.default_rng(0)
    px = np.concatenate([r if len(r) <= AMOSTRA_REGIAO else r[sorteio.choice(len(r), AMOSTRA_REGIAO, replace=False)]
                         for r in regs])
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
    a = max(min(a_comp, max(a_reg, 1.0) * 1.5), AREA_REF_MIN)
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


# ---------- Nome do NPC em ciano ("NPC Highlights" do cliente oficial) ----------
NOME_NPC_TOLERANCIA = (12, 8)  # px: o nome achado pelo molde tem que cair na linha achada


def aprender_nome_npc(exemplos):
    """Nome do NPC em ciano acima do monstro: {'molde', 'desloc': (dx, dy), 'achados': n} ou None.

    O molde vem do 1º exemplo com nome; o deslocamento (do centro do nome achado pelo molde, como
    o bot vai achar, até o ponto do mouse) é a mediana dos exemplos onde o molde achou o nome."""
    molde, descs = None, []
    for img, pos in exemplos:
        caixa = visao.achar_nome_acima(img, pos)
        if caixa is None:
            continue
        if molde is None:
            molde = visao.molde_nome_npc(img, caixa)
        cx, cy = caixa[0] + caixa[2] / 2, caixa[1] + caixa[3] / 2
        so_nome = {"nome_npc": (molde, (0, 0), visao.NOME_NPC_LIMIAR)}
        perto = [s["nome"] for s in visao.detectar_por_nome(img, so_nome, interface=False)
                 if abs(s["nome"][0] - cx) <= NOME_NPC_TOLERANCIA[0] and abs(s["nome"][1] - cy) <= NOME_NPC_TOLERANCIA[1]]
        if not perto:
            continue  # outro nome (outro NPC) ou nome encavalado: não entra no deslocamento
        n = min(perto, key=lambda q: abs(q[0] - cx) + abs(q[1] - cy))
        descs.append((pos[0] - n[0], pos[1] - n[1]))
    if molde is None or not descs:
        return None
    dx, dy = np.median(np.array(descs, dtype=float), axis=0)
    return {"molde": molde, "desloc": (int(round(dx)), int(round(dy))), "achados": len(descs)}


# ---------- Perfil em arquivo ----------
def montar_perfil(nome_jogo, molde, exemplos, espera_loot=4.5, loot_menu=True, osso="Big bones"):
    """Dict do perfil (formato de perfis/<nome>.json, sem o molde) ou None.

    `osso`: "Big bones" (o único osso que o bot conhece: pega, enterra e espera por ele) ou
    None (monstro que larga outro osso ou nenhum: o bot só pega runas e moedas).
    Com o nome do NPC em ciano na tela (aprender_nome_npc), o perfil acha o monstro pelo nome
    ('nome_npc'; o molde vai em '_molde_npc' até salvar_perfil) e a cor fica de reserva."""
    cor = aprender_cor(exemplos)
    nome = aprender_nome_npc(exemplos)
    if cor is None and nome is None:
        return None
    if cor is None:  # só o nome serve (a cor do corpo se confunde com o cenário)
        cor = {"cor": ((0, 0, 0), (0, 0, 0)), "area": (0, 0), "area_ref": 0.0, "caixa_ref": (40, 40),
               "vazamento": 0.0, "desloc_matiz": 0}
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
    if nome is not None:
        dx, dy = nome["desloc"]
        th, tw = nome["molde"].shape[:2]
        perfil["nome_npc"] = {"desloc": [dx, dy], "limiar": visao.NOME_NPC_LIMIAR, "achados": nome["achados"]}
        perfil["_molde_npc"] = nome["molde"]
        # a barra de vida de quem já luta fica logo abaixo do nome: de 15 px acima do centro
        # do nome até 30 px abaixo dele (o corpo, onde fica `pos`, está dy abaixo do nome)
        perfil["barra"] = [int(max(45, tw // 2 + 10)), int(dy + 15), int(30 - dy)]
        # a correção da mira procura o nome num recorte em volta do corpo: tem que caber
        perfil["raio_mira"] = int(max(perfil["raio_mira"], abs(dy) + th + 20, abs(dx) + tw))
    return perfil


def salvar_perfil(nome, perfil, molde, exemplos=(), faixas=(), pasta=None):
    """Grava perfis/<nome>.json, o molde do texto e os quadros e faixas de ensino (para os
    testes: testar_visao.testar_perfis_aprendidos)."""
    pasta = pasta or visao.PASTA_PERFIS
    os.makedirs(pasta, exist_ok=True)
    perfil = dict(perfil, texto=dict(perfil["texto"], molde=f"{nome}_texto.png"))
    molde_npc = perfil.pop("_molde_npc", None)
    if perfil.get("nome_npc") and molde_npc is not None:
        perfil["nome_npc"] = dict(perfil["nome_npc"], molde=f"{nome}_npc.png")
        cv2.imwrite(os.path.join(pasta, f"{nome}_npc.png"), molde_npc)
    pasta_ex = os.path.join(pasta, f"{nome}_exemplos")
    if os.path.isdir(pasta_ex):  # ensino refeito com o mesmo nome: nada do anterior fica
        shutil.rmtree(pasta_ex)
    cv2.imwrite(os.path.join(pasta, f"{nome}_texto.png"), molde)
    if exemplos:
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


# ---------- Sessão de ensino ----------
class Ensino:
    """Junta exemplos (quadro, ponto) e o molde do texto; reaprende a cada exemplo."""

    def __init__(self, nome, nome_jogo, espera_loot, osso="Big bones"):
        self.nome, self.nome_jogo = nome, nome_jogo
        self.espera_loot, self.osso = espera_loot, osso
        self.molde = None
        self.exemplos = []
        self.faixas = []
        self.perfil = None

    def adicionar(self, img, pos, faixa):
        """Novo exemplo. Devolve uma mensagem para o usuário."""
        if not visao.dentro_do_jogo(pos) or visao.na_interface(pos):
            return "exemplo recusado: o mouse está fora da área do jogo (ponha-o em cima do monstro)"
        molde, motivo = gerar_molde_texto(faixa)
        if molde is None:
            return f"exemplo recusado: {motivo}"
        if visao.achar_nome_acima(img, pos) is None:  # com o nome do NPC na tela a cor não decide
            motivo = checar_exemplo(img, pos)
            if motivo is not None:
                return f"exemplo recusado: {motivo}"
        if self.molde is None:
            self.molde = molde
        elif score_molde(faixa, self.molde) < LIMIAR_TEXTO_APRENDIDO:
            return "exemplo recusado: o texto não é o do mesmo monstro do 1º exemplo"
        self.exemplos.append((img, (int(pos[0]), int(pos[1]))))
        perfil = montar_perfil(self.nome_jogo, self.molde, self.exemplos, self.espera_loot,
                               osso=self.osso)
        if perfil is None:
            self.exemplos.pop()
            if not self.exemplos:
                self.molde = None
            return "exemplo recusado: não achei a cor do corpo nesse ponto (mouse na borda?)"
        self.perfil = perfil
        if not any(f is faixa for f in self.faixas):
            self.faixas.append(faixa)
        aviso = ""
        if len(self.exemplos) > 1 and min(np.hypot(pos[0] - p[0], pos[1] - p[1])
                                          for _, p in self.exemplos[:-1]) <= 25:
            aviso = " (mesmo lugar de um exemplo anterior: ensine OUTROS monstros, de preferência os que ficaram sem círculo)"
        return self.resumo() + aviso

    def desfazer(self):
        """Tira o último exemplo e reaprende com os que sobraram."""
        if not self.exemplos:
            return "nada para desfazer"
        self.exemplos.pop()
        if len(self.faixas) > len(self.exemplos):
            self.faixas.pop()
        if not self.exemplos:
            self.molde, self.perfil, self.faixas = None, None, []
            return "exemplo desfeito: nenhum exemplo agora"
        self.perfil = montar_perfil(self.nome_jogo, self.molde, self.exemplos, self.espera_loot,
                                    osso=self.osso)
        return "exemplo desfeito; " + self.resumo()

    def perfil_visao(self):
        """O perfil no formato de visao.PERFIS, para detectar com ele."""
        p = self.perfil
        pv = {"cor": (np.array(p["cor"][0]), np.array(p["cor"][1])),
              "area": tuple(p["area"]), "fechar": p["fechar"], "desloc_matiz": p.get("desloc_matiz", 0)}
        if p.get("nome_npc"):
            pv["nome_npc"] = (p["_molde_npc"], tuple(p["nome_npc"]["desloc"]), p["nome_npc"]["limiar"])
        return pv

    def acha(self, img, pos, raio=30, candidatos=None):
        """O perfil atual acha um monstro a até `raio` px (em x e em y) de `pos` em `img`?
        `candidatos`: o detectar_sapos de `img` já feito (para não detectar de novo)."""
        if candidatos is None:
            candidatos = visao.detectar_sapos(img, self.perfil_visao())
        return any(abs(s["pos"][0] - pos[0]) <= raio and abs(s["pos"][1] - pos[1]) <= raio
                   for s in candidatos)

    def resumo(self):
        pv = self.perfil_visao()
        partes = []
        for img, pos in self.exemplos:
            c = visao.detectar_sapos(img, pv)
            achou = self.acha(img, pos, candidatos=c)
            partes.append(f"{len(c)}{'' if achou else ' (este exemplo não foi achado!)'}")
        p = self.perfil
        if p.get("nome_npc"):
            n = p["nome_npc"]
            como = (f"pelo NOME em ciano (achado em {n['achados']} de {len(self.exemplos)} exemplo(s); "
                    f"corpo {n['desloc'][1]} px abaixo do nome)")
        else:
            como = (f"pela cor HSV {p['cor'][0]}..{p['cor'][1]}, área {p['area'][0]}-{p['area'][1]} px "
                    f"(sem nome de NPC em ciano na tela)")
        return (f"{len(self.exemplos)} exemplo(s); candidatos por quadro: {', '.join(partes)}. "
                f"Acha o monstro {como}; barra {p['barra']}, mira {p['raio_mira']} px")

    def conferir(self):
        """Grava perfis/<nome>_conferir.png com os candidatos do último quadro marcados."""
        os.makedirs(visao.PASTA_PERFIS, exist_ok=True)
        img = self.exemplos[-1][0]
        out = marcar_candidatos(img, self.perfil_visao(), [e for e in self.exemplos if e[0] is img])
        caminho = os.path.join(visao.PASTA_PERFIS, f"{self.nome}_conferir.png")
        cv2.imwrite(caminho, out)
        return caminho

    def salvar(self):
        return salvar_perfil(self.nome, self.perfil, self.molde, self.exemplos, self.faixas)
