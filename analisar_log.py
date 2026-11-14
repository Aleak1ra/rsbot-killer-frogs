"""Resume os tempos medidos numa sessão do bot.

Uso:  python analisar_log.py [logs/sessao_X.jsonl]   (sem argumento: a mais recente)
"""
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


def tempo_em_foco(eventos):
    """Segundos da sessão com o jogo em primeiro plano."""
    if not eventos:
        return 0.0
    fim = eventos[-1]["t"]
    total, desde = 0.0, 0.0
    for e in eventos:
        if e["ev"] == "foco":
            if not e["ok"] and desde is not None:
                total += e["t"] - desde
                desde = None
            elif e["ok"]:
                desde = e["t"]
    if desde is not None:
        total += fim - desde
    return total


def resumir(eventos):
    por = {}
    for e in eventos:
        por.setdefault(e["ev"], []).append(e)
    ativo = tempo_em_foco(eventos)
    minutos = max(ativo / 60, 1e-9)
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
    print(f"\nLutas terminadas: {len(fim)}  duração (s): {pct([e['dur'] for e in fim])}")
    print(f"  estados no fim: { {s: sum(e['estado'] == s for e in fim) for s in ('morto', 'livre')} }")
    print(f"  lutas não iniciadas pelo bot: {len(por.get('combate_externo', []))}")

    loot = por.get("loot", [])
    ok_loot = [e for e in loot if e["ok"]]
    print(f"\nLoot após morte: {len(ok_loot)}/{len(loot)} apareceu  latência (s): "
          f"{pct([e['lat'] for e in ok_loot])}")

    hs = por.get("hover_sapo", [])
    print(f"\nHover em candidatos a sapo: {len(hs)}")
    for texto in ("giant", "hillgiant", None):
        g = [e for e in hs if e["texto"] == texto]
        if not g:
            continue
        print(f"  {str(texto):5}: {len(g):3}  área: {pct([e['area'] for e in g], (0, 50, 100))}  "
              f"latência: {pct([e['lat'] for e in g])}")
        if texto is None:
            print(f"         melhor score giant: {pct([e.get('score', e.get('giant', 0)) for e in g])}  "
                  f"big: {pct([e.get('big', 0) for e in g])}")
    aceitos = [e for e in hs if e["texto"] is not None]
    if aceitos:
        print(f"  score giant quando aceito (mín/p50): {pct([e.get('score', e.get('giant', 0)) for e in aceitos], (0, 50))}")

    ho = por.get("hover_osso", [])
    ok_ho = [e for e in ho if e["ok"]]
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
    mn = por.get("menu", [])
    if mn:
        achou = [e for e in mn if e["ok"]]
        print(f"  menus abertos: {len(mn)}  com \"Take Big bones\": {len(achou)}  "
              f"score: {pct([e['score'] for e in achou], (0, 50))}  "
              f"latência (s): {pct([e['lat'] for e in achou])}")

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
              f"sapos {1000 * np.median([e['sapos'] for e in ci]):.0f}")
        print(f"  ciclos sem nenhum candidato a sapo: {sum(e['n_sapos'] == 0 for e in ci)}/{len(ci)}")

    # Para onde vai o tempo
    t_luta = sum(e["dur"] for e in fim)
    t_atk = sum(e["lat"] for e in atk)
    t_loot = sum(e["lat"] for e in loot)
    t_pegar = sum(e["lat"] for e in pg)
    t_ent = sum(e.get("lat", 0.0) for e in en if e.get("motivo") == "limite") + 0.25 * len(en)
    t_hover = sum(e["lat"] for e in hs) + sum(e["lat"] for e in ho)
    print("\nDistribuição do tempo em foco:")
    for nome, v in [("lutando", t_luta), ("indo até o sapo", t_atk),
                    ("esperando loot", t_loot), ("pegando osso", t_pegar),
                    ("enterrando", t_ent), ("conferindo texto", t_hover)]:
        print(f"  {nome:18} {v:6.0f}s  {100 * v / max(ativo, 1e-9):4.0f}%")
    resto = ativo - (t_luta + t_atk + t_loot + t_pegar + t_ent + t_hover)
    print(f"  {'outros / ocioso':18} {resto:6.0f}s  {100 * resto / max(ativo, 1e-9):4.0f}%")


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
