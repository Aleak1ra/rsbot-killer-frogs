"""Ensina um monstro novo ao bot, sem editar código.

Uso:  python ensinar.py "<nome no jogo>"   (ex.: python ensinar.py "Moss giant")
  O perfil fica em perfis/<nome>.json com o nome convertido como no main.py ("Moss giant" ->
  moss_giant). Um nome igual ao de um embutido no jogo (ex.: "Giant frog") passa a usar o
  perfil ensinado no --monstro, no lugar do embutido.
  1. Jogo maximizado e em primeiro plano, como para o bot.
  2. Ponha o mouse em cima do monstro (o texto do canto superior esquerdo deve mostrar
     "Attack <Nome> (level-N)") e aperte F9. O mouse não é movido por este programa.
     O terminal fica atrás do jogo: um bipe normal diz que o exemplo serviu; o bipe de erro,
     que foi recusado (o motivo sai no terminal). F7 e F10 também bipam.
  3. Confira perfis/<nome>_conferir.png: círculo verde = o que o bot acha que é o monstro,
     x vermelho = onde você ensinou. Se algum monstro ficou sem círculo (ex.: outro tom de
     pele), ponha o mouse nele e aperte F9 de novo. 2 a 4 exemplos costumam bastar.
     F7 desfaz o último exemplo (se o círculo verde apareceu no chão, por exemplo).
  4. F10 grava perfis/<nome>.json. Depois: python main.py --monstro "<nome no jogo>"
  Ctrl+C no terminal sai sem gravar. (Se o F9/F10 trocar a aba do painel no jogo, volte
  para a mochila antes de ligar o bot: ele lê os ossos no inventário.)

Opções:
  --espera-loot S    tempo máximo esperando o loot depois da morte (padrão 4,5 s)
  --nome-jogo "X"    nome mostrado nas mensagens (padrão: o nome digitado)
  --sem-osso         o monstro não larga "Big bones" (o único osso que o bot conhece):
                     o bot não pega nem enterra ossos e só junta runas e moedas
  --de-amostras CLASSE --faixa ARQ [--max N]: aprende de quadros já salvos
      (amostras/rotulos.jsonl, ex.: hillgiant_ok) e de uma faixa de texto salva, sem ler a tela.

O que o perfil guarda e como foi medido: ver ensino.py. Os perfis embutidos (sapo,
hillgiant) continuam em visao.PERFIS e não mudam.
"""
import argparse
import json
import os
import sys
import time

import cv2

import ensino
import visao


def _args():
    p = argparse.ArgumentParser(description="Ensina um monstro novo ao bot.")
    p.add_argument("nome", help='o monstro como no jogo, ex.: "Moss giant" (vira perfis/moss_giant.json)')
    p.add_argument("--espera-loot", type=float, default=4.5)
    p.add_argument("--nome-jogo", default=None)
    p.add_argument("--sem-osso", action="store_true")
    p.add_argument("--de-amostras", default=None, metavar="CLASSE")
    p.add_argument("--faixa", default=None)
    p.add_argument("--max", type=int, default=4)
    a = p.parse_args()
    a.nome_arq = visao.nome_arquivo(a.nome)  # a mesma conversão do --monstro do main.py
    if not a.nome_arq:
        p.error("o nome precisa de pelo menos uma letra ou número")
    if a.nome_arq in visao.PERFIS:
        p.error(f"'{a.nome_arq}' é um perfil embutido; escolha outro nome")
    if a.de_amostras and not a.faixa:
        p.error("--de-amostras precisa de --faixa (uma faixa do texto de ataque salva)")
    return a


def de_amostras(e, classe, faixa_arq, maximo):
    faixa = cv2.imread(faixa_arq)
    if faixa is None:
        sys.exit(f"não consegui ler {faixa_arq}")
    caminho = os.path.join(visao.BASE_DIR, "amostras", "rotulos.jsonl")
    with open(caminho, encoding="utf-8") as f:
        rots = [json.loads(linha) for linha in f]
    rots = [r for r in rots if r["classe"] == classe]
    if not rots:
        sys.exit(f"nenhuma amostra da classe {classe} em {caminho}")
    passo = max(len(rots) // maximo, 1)  # espalhados pela sessão (aparências diferentes)
    for r in rots[::passo][:maximo]:
        img = cv2.imread(os.path.join(visao.BASE_DIR, "amostras", r["arquivo"]))
        if img is None:
            continue
        print(f"{r['arquivo']}: {e.adicionar(img, tuple(r['pos']), faixa)}")
    if e.perfil is None:
        sys.exit("nenhum exemplo serviu")
    print(f"conferir: {e.conferir()}")
    print(f"gravado: {e.salvar()}")


def ao_vivo(e):
    import keyboard

    import tela

    print(__doc__.split("Opções:")[0])
    print("Aguardando F9 (exemplo), F7 (desfaz o último) ou F10 (gravar); Ctrl+C sai sem gravar...")

    while True:
        time.sleep(0.02)
        if keyboard.is_pressed("F7"):
            tela.esperar_soltar("F7")
            msg = e.desfazer()
            print(msg)
            tela.bip("erro" if msg == "nada para desfazer" else "ok")
            if e.perfil is not None:
                print(f"Confira: {e.conferir()}")
            continue
        if keyboard.is_pressed("F10"):
            tela.esperar_soltar("F10")
            if e.perfil is None:
                tela.bip("erro")
                print("Nada para gravar ainda: ensine pelo menos um exemplo com F9.")
                continue
            print(f"Gravado: {e.salvar()}")
            tela.bip("ok")
            # o nome do jogo, se ele leva ao mesmo arquivo (um --nome-jogo diferente pode não levar)
            usar = e.nome_jogo if visao.nome_arquivo(e.nome_jogo) == e.nome else e.nome
            print(f'Para usar: python main.py --monstro "{usar}"')
            return
        leitura = tela.ler_f9(timeout=0)  # o mesmo quadro tem o monstro e o texto de ação
        if leitura is None:
            continue
        img, pos, problema = leitura
        if problema:
            msg = f"exemplo recusado: {problema}"
        else:
            try:
                msg = e.adicionar(img, pos, visao.recortar(img, visao.FAIXA_TEXTO_ACAO))
            except Exception as erro:  # um exemplo ruim não pode derrubar o ensino inteiro
                msg = f"exemplo recusado: erro ao aprender ({erro!r})"
        recusado = msg.startswith("exemplo recusado")
        print(tela.com_dica_de_foco(msg) if recusado else msg)
        tela.bip("erro" if recusado else "ok")
        if not recusado and e.perfil is not None:
            print(f"Confira: {e.conferir()}")


def principal():
    a = _args()
    e = ensino.Ensino(a.nome_arq, a.nome_jogo or a.nome, a.espera_loot,
                      osso=None if a.sem_osso else "Big bones")
    if a.de_amostras:
        de_amostras(e, a.de_amostras, a.faixa, a.max)
        return
    try:
        ao_vivo(e)
    except KeyboardInterrupt:
        print()
        print("Saindo sem gravar.")


if __name__ == "__main__":
    principal()
