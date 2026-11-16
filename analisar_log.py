"""Resume os tempos medidos numa sessão do bot.

Uso:  python analisar_log.py [logs/sessao_X.jsonl]   (sem argumento: a mais recente)
"""
import collections
import glob
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def carregar(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]


def pct(valores, ps=(50, 90, 100)):
    if not valores:
        return "-"
    return " / ".join(f"{np.percentile(valores, p):.2f}" for p in ps)


def intervalos(eventos, liga, desliga, ligado=False):
    """[(início, fim)] em que o estado ficou ligado: de `liga(e)` até `desliga(e)` (ou o fim
    da sessão). `ligado`: a sessão (t = 0) começa ligada."""
    ivs, desde = [], (0.0 if ligado else None)
    for e in eventos:
        if desde is None and liga(e):
            desde = e["t"]
        elif desde is not None and desliga(e):
            ivs.append((desde, e["t"]))
            desde = None
    if desde is not None and eventos:
        ivs.append((desde, eventos[-1]["t"]))
    return ivs


def subtrair(ivs, tirar):
    """Os intervalos `ivs` sem as partes que caem em algum de `tirar`."""
    out = []
    for a, b in ivs:
        for c, d in sorted(tirar):
            if d <= a or c >= b:
                continue
            if c > a:
                out.append((a, c))
            a = max(a, d)
        if a < b:
            out.append((a, b))
    return out


def dentro(a, b, ivs):
    """Segundos de [a, b] que caem nos intervalos `ivs`."""
    return sum(max(0.0, min(b, d) - max(a, c)) for c, d in ivs)


def resumir(eventos):
    por = {}
    for e in eventos:
        por.setdefault(e["ev"], []).append(e)
    # Tempo ativo: o jogo em primeiro plano, fora das pausas por vida baixa (o bot fica parado
    # esperando a vida voltar; a luta em andamento seguia contando e passava de 100%).
    em_foco = intervalos(eventos, lambda e: e["ev"] == "foco" and e["ok"],
                         lambda e: e["ev"] == "foco" and not e["ok"], ligado=True)
    pausas = intervalos(eventos, lambda e: e["ev"] == "vida_baixa", lambda e: e["ev"] == "vida_ok")
    ativos = subtrair(em_foco, pausas)
    foco = sum(b - a for a, b in em_foco)
    ativo = sum(b - a for a, b in ativos)
    minutos = max(ativo / 60, 1e-9)
    if pausas:
        print(f"Duração em foco: {foco:.0f}s; pausado por vida: {foco - ativo:.0f}s; "
              f"ativo: {ativo:.0f}s ({minutos:.1f} min)   [p50 / p90 / máx]")
    else:
        print(f"Duração em foco: {ativo:.0f}s ({minutos:.1f} min)   [p50 / p90 / máx]")

    atk = por.get("atacar", [])
    ok_atk = [e for e in atk if e["ok"]]
    print(f"\nAtaques: {len(atk)}  iniciaram combate: {len(ok_atk)}  "
          f"({len(ok_atk) / minutos:.2f}/min)")
    print(f"  clique -> combate (s): {pct([e['lat'] for e in ok_atk])}")
    print(f"  distância do alvo (px): {pct([e['dist'] for e in atk])}")
    print(f"  falhas: {len(atk) - len(ok_atk)}")
    if any("clique" in e for e in atk):
        print("  por X de clique (tipo: ataques / viraram combate / tempo perdido nas falhas):")
        for tipo in ("alvo", "chao", None):
            g = [e for e in atk if e.get("clique") == tipo]
            perdido = sum(e["lat"] for e in g if not e["ok"])
            print(f"    {str(tipo):5}: {len(g):3} / {sum(e['ok'] for e in g):3} / {perdido:.1f}s")
    if any(e.get("desloc") is not None for e in atk):
        for tipo in ("alvo", "chao"):
            d = [e["desloc"] for e in atk if e.get("clique") == tipo and e.get("desloc") is not None]
            print(f"    correção de mira antes do clique ({tipo}), px: {pct(d)}")
    cl = por.get("clique", [])
    if cl:
        lidos = [e for e in cl if e["tipo"]]
        print(f"  leitura do X: {len(lidos)}/{len(cl)} cliques lidos, latência (s): "
              f"{pct([e['lat'] for e in lidos])}")
    if atk and "barra" in atk[0]:
        print(f"  pixels de barra de vida sobre o alvo — ok: "
              f"{sorted(e['barra'] for e in ok_atk)}  falhou: "
              f"{sorted(e['barra'] for e in atk if not e['ok'])}")

    print(f"  candidatos pulados por já estarem em luta com outro jogador: "
          f"{len(por.get('sapo_ocupado', []))}")

    fim = por.get("fim_combate", [])
    # a luta [t - dur, t] sem a parte pausada por vida baixa
    dur_luta = [e["dur"] - dentro(e["t"] - e["dur"], e["t"], pausas) for e in fim]
    print(f"\nLutas terminadas: {len(fim)}  duração (s): {pct(dur_luta)}")
    print(f"  estados no fim: { {s: sum(e['estado'] == s for e in fim) for s in ('morto', 'livre')} }")
    print(f"  lutas não iniciadas pelo bot: {len(por.get('combate_externo', []))}")

    loot = por.get("loot", [])
    ok_loot = [e for e in loot if e["ok"]]
    print(f"\nLoot após morte: {len(ok_loot)}/{len(loot)} apareceu  latência (s): "
          f"{pct([e['lat'] for e in ok_loot])}")

    hs = por.get("hover_sapo", [])
    print(f"\nHover em candidatos a monstro: {len(hs)}")
    textos = sorted({e["texto"] for e in hs}, key=str)  # o molde lido (perfil_<nome> nos aprendidos)
    largura = max((len(str(t)) for t in textos), default=5)
    for texto in textos:
        g = [e for e in hs if e["texto"] == texto]
        print(f"  {str(texto):{largura}}: {len(g):3}  área: {pct([e['area'] for e in g], (0, 50, 100))}  "
              f"latência: {pct([e['lat'] for e in g])}")
        if texto is None:
            print(f"  {'':{largura}}  melhor score: {pct([e.get('score', e.get('giant', 0)) for e in g])}")
    aceitos = [e for e in hs if e["texto"] is not None]
    if aceitos:
        print(f"  score quando aceito (mín/p50): "
              f"{pct([e.get('score', e.get('giant', 0)) for e in aceitos], (0, 50))}")

    ho = por.get("hover_osso", [])
    ok_ho = [e for e in ho if e["ok"]]
    if ho:
        print(f"\nHover em ossos: {len(ho)}  confirmados: {len(ok_ho)}")
        print(f"  latência do texto (s): {pct([e['lat'] for e in ok_ho])}")
        print(f"  score do rótulo confirmado (mín/p50): {pct([e['score_rotulo'] for e in ok_ho], (0, 50))}")
        rej = [e for e in ho if not e["ok"]]
        if rej:
            print(f"  rejeitados: score rótulo {pct([e['score_rotulo'] for e in rej], (0, 50, 100))}  "
                  f"texto {pct([e['score_texto'] for e in rej], (0, 50, 100))}")

    pg = por.get("pegar_osso", [])
    ok_pg = [e for e in pg if e["ok"]]
    print(f"\nOssos pegos: {len(ok_pg)}/{len(pg)}  ({len(ok_pg) / minutos:.2f}/min)")
    print(f"  clique -> inventário (s): {pct([e['lat'] for e in ok_pg])}")
    if ok_pg:
        vel = [e["dist"] / e["lat"] for e in ok_pg if e["lat"] > 0.3]
        print(f"  px por segundo de caminhada: {pct(vel, (10, 50, 90))}")

    for via in ("direto", "menu"):
        g = [e for e in pg if e.get("via") == via]
        if g:
            print(f"  via {via}: {sum(e['ok'] for e in g)}/{len(g)}  "
                  f"latência (s): {pct([e['lat'] for e in g if e['ok']])}")

    # Runas e moedas (main.ColetaChao): falha sem 'motivo' = o inventário não mudou
    coletas = {item: por.get(f"pegar_{item}", []) for item in ("runa", "moedas", "semente")}
    extras = {nome: por.get(nome, []) for nome in ("mira_perdida", "menu_procura", "clique_cancelado")}
    if any(coletas.values()) or any(extras.values()):
        print("\nItens do chão:")
        for item, plural in (("runa", "runas"), ("moedas", "moedas"), ("semente", "Snape grass seeds")):
            g = coletas[item]
            if not g:
                continue
            ok = [e for e in g if e["ok"]]
            falhas = collections.Counter(e.get("motivo", "inventario_igual") for e in g if not e["ok"])
            pausas_item = len(por.get(f"{plural}_pausa", []))
            print(f"  {plural}: {len(ok)}/{len(g)}  ({len(ok) / minutos:.2f}/min)  clique -> inventário (s): "
                  f"{pct([e['lat'] for e in ok])}  falhas: {dict(falhas) or 0}"
                  + (f"  pausas: {pausas_item}" if pausas_item else ""))
        mp = extras["mira_perdida"]
        print(f"  mira perdida (a tela não parou): {len(mp)}"
              + (f" {dict(collections.Counter(e.get('item') for e in mp))}" if mp else ""))
        mpr = extras["menu_procura"]
        print(f"  procura do \"Take\" com o mouse: {sum(bool(e['achou']) for e in mpr)}/{len(mpr)} acharam")
        print(f"  cliques cancelados (sem foco ou o mouse mexido): {len(extras['clique_cancelado'])}")

    mn = por.get("menu", [])
    if mn:
        print(f"\nMenus abertos: {len(mn)}  (com a linha \"Take\" / abertos)")
        for item in sorted({e.get("item", "osso") for e in mn}, key=lambda i: (i != "osso", i)):
            g = [e for e in mn if e.get("item", "osso") == item]  # os antigos, sem 'item', eram de osso
            achou = [e for e in g if e["ok"]]
            print(f"  {item:6}: {len(achou):3}/{len(g):3}  score (mín/p50): "
                  f"{pct([e['score'] for e in achou], (0, 50))}  latência (s): {pct([e['lat'] for e in achou])}")

    en = por.get("enterrar", [])
    ok_en = [e for e in en if e["ok"]]
    print(f"\nEnterros: {len(ok_en)}/{len(en)}  ({len(ok_en) / minutos:.2f}/min)")
    print(f"  por motivo: { {m: sum(e.get('motivo') == m for e in ok_en) for m in sorted({e.get('motivo') for e in en})} }")

    vd = [e["pct"] for e in por.get("vida", []) if e["pct"] is not None]
    if vd:
        print(f"\nVida do jogador: início {100 * vd[0]:.0f}%  mínima {100 * min(vd):.0f}%  "
              f"fim {100 * vd[-1]:.0f}%   pausas por vida baixa: {len(por.get('vida_baixa', []))}")

    ci = por.get("ciclo", [])
    if ci:
        print(f"\nCusto por ciclo (ms): captura {1000 * np.median([e['captura'] for e in ci]):.0f}  "
              f"rótulos {1000 * np.median([e['rotulos'] for e in ci]):.0f}  "
              f"monstros {1000 * np.median([e['sapos'] for e in ci]):.0f}")
        print(f"  ciclos sem nenhum candidato a monstro: {sum(e['n_sapos'] == 0 for e in ci)}/{len(ci)}")

    # Para onde vai o tempo ativo
    t_luta = sum(dentro(e["t"] - e["dur"], e["t"], ativos) for e in fim)
    t_atk = sum(e["lat"] for e in atk)
    t_loot = sum(e["lat"] for e in loot)
    t_pegar = sum(e["lat"] for e in pg)
    t_itens = sum(e.get("lat", 0.0) for g in coletas.values() for e in g)
    t_ent = sum(e.get("lat", 0.0) for e in en if e.get("motivo") == "limite") + 0.25 * len(en)
    t_hover = sum(e["lat"] for e in hs) + sum(e["lat"] for e in ho)
    partes = [("lutando", t_luta), ("indo até o monstro", t_atk), ("esperando loot", t_loot),
              ("pegando osso", t_pegar), ("pegando runa/moedas", t_itens), ("enterrando", t_ent),
              ("conferindo texto", t_hover)]
    partes.append(("outros / ocioso", ativo - sum(v for _, v in partes)))
    print("\nDistribuição do tempo ativo:")
    for nome, v in partes:
        print(f"  {nome:19} {v:6.0f}s  {100 * v / max(ativo, 1e-9):4.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        caminho = sys.argv[1]
    else:
        arquivos = sorted(glob.glob(os.path.join(BASE_DIR, "logs", "sessao_*.jsonl")))
        if not arquivos:
            sys.exit("Nenhum log em logs/")
        caminho = arquivos[-1]
    print(caminho)
    resumir(carregar(caminho))
