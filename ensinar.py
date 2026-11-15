"""Ensina um monstro novo ao bot, sem editar código.

Uso:  python ensinar.py <nome>          (ex.: python ensinar.py ogro)
  1. Jogo maximizado e em primeiro plano, como para o bot.
  2. Ponha o mouse em cima do monstro (o texto do canto superior esquerdo deve mostrar
     "Attack <Nome> (level-N)") e aperte F9. O mouse não é movido por este programa.
  3. Confira perfis/<nome>_conferir.png: círculo verde = o que o bot acha que é o monstro,
     x vermelho = onde você ensinou. Se algum monstro ficou sem círculo (ex.: outro tom de
     pele), ponha o mouse nele e aperte F9 de novo. 2 a 4 exemplos costumam bastar.
     F7 desfaz o último exemplo (se o círculo verde apareceu no chão, por exemplo).
  4. F10 grava perfis/<nome>.json. Depois: python main.py --monstro <nome>
  Ctrl+C no terminal sai sem gravar. (Se o F9/F10 trocar a aba do painel no jogo, volte
  para a mochila antes de ligar o bot: ele lê os ossos no inventário.)

Opções:
  --espera-loot S    tempo máximo esperando o loot depois da morte (padrão 4,5 s)
  --nome-jogo "X"    nome mostrado nas mensagens (padrão: <nome>)
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
import re
import sys
import time

import cv2
import numpy as np

import ensino
import visao


def _args():
    p = argparse.ArgumentParser(description="Ensina um monstro novo ao bot.")
    p.add_argument("nome", help="nome do perfil (letras minúsculas, números, _ ou -)")
    p.add_argument("--espera-loot", type=float, default=4.5)
    p.add_argument("--nome-jogo", default=None)
    p.add_argument("--sem-osso", action="store_true")
    p.add_argument("--de-amostras", default=None, metavar="CLASSE")
    p.add_argument("--faixa", default=None)
    p.add_argument("--max", type=int, default=4)
    a = p.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+", a.nome):
        p.error("use só letras minúsculas, números, _ ou - no nome")
    if a.nome in visao.PERFIS:
        p.error(f"'{a.nome}' é um perfil embutido; escolha outro nome")
    if a.de_amostras and not a.faixa:
        p.error("--de-amostras precisa de --faixa (uma faixa do texto de ataque salva)")
    return a


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
        molde, motivo = ensino.gerar_molde_texto(faixa)
        if molde is None:
            return f"exemplo recusado: {motivo}"
        motivo = ensino.checar_exemplo(img, pos)
        if motivo is not None:
            return f"exemplo recusado: {motivo}"
        if self.molde is None:
            self.molde = molde
        elif ensino.score_molde(faixa, self.molde) < ensino.LIMIAR_TEXTO_APRENDIDO:
            return "exemplo recusado: o texto não é o do mesmo monstro do 1º exemplo"
        self.exemplos.append((img, (int(pos[0]), int(pos[1]))))
        perfil = ensino.montar_perfil(self.nome_jogo, self.molde, self.exemplos, self.espera_loot,
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
        self.perfil = ensino.montar_perfil(self.nome_jogo, self.molde, self.exemplos, self.espera_loot,
                                           osso=self.osso)
        return "exemplo desfeito; " + self.resumo()

    def perfil_visao(self):
        """O perfil no formato de visao.PERFIS, para detectar com ele."""
        p = self.perfil
        return {"cor": (np.array(p["cor"][0]), np.array(p["cor"][1])),
                "area": tuple(p["area"]), "fechar": p["fechar"], "desloc_matiz": p.get("desloc_matiz", 0)}

    def resumo(self):
        pv = self.perfil_visao()
        partes = []
        for img, pos in self.exemplos:
            c = visao.detectar_sapos(img, pv)
            perto = any(abs(s["pos"][0] - pos[0]) <= 30 and abs(s["pos"][1] - pos[1]) <= 30 for s in c)
            partes.append(f"{len(c)}{'' if perto else ' (este exemplo não foi achado!)'}")
        p = self.perfil
        return (f"{len(self.exemplos)} exemplo(s); candidatos por quadro: {', '.join(partes)}. "
                f"Cor HSV {p['cor'][0]}..{p['cor'][1]}, área {p['area'][0]}-{p['area'][1]} px, "
                f"barra {p['barra']}, mira {p['raio_mira']} px")

    def conferir(self):
        """Grava perfis/<nome>_conferir.png com os candidatos do último quadro marcados."""
        os.makedirs(visao.PASTA_PERFIS, exist_ok=True)
        img = self.exemplos[-1][0]
        out = ensino.marcar_candidatos(img, self.perfil_visao(), [e for e in self.exemplos if e[0] is img])
        caminho = os.path.join(visao.PASTA_PERFIS, f"{self.nome}_conferir.png")
        cv2.imwrite(caminho, out)
        return caminho

    def salvar(self):
        return ensino.salvar_perfil(self.nome, self.perfil, self.molde, self.exemplos, self.faixas)


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
    import pyautogui

    import tela

    print(__doc__.split("Opções:")[0])
    print("Aguardando F9 (exemplo), F7 (desfaz o último) ou F10 (gravar); Ctrl+C sai sem gravar...")

    def solta(tecla):
        while keyboard.is_pressed(tecla):
            time.sleep(0.02)

    while True:
        time.sleep(0.02)
        if keyboard.is_pressed("F7"):
            solta("F7")
            print(e.desfazer())
            if e.perfil is not None:
                print(f"Confira: {e.conferir()}")
            continue
        if keyboard.is_pressed("F10"):
            solta("F10")
            if e.perfil is None:
                print("Nada para gravar ainda: ensine pelo menos um exemplo com F9.")
                continue
            print(f"Gravado: {e.salvar()}")
            print(f"Para usar: python main.py --monstro {e.nome}")
            return
        if not keyboard.is_pressed("F9"):
            continue
        problema = tela.atualizar_janela()
        # lê na hora do aperto: o mesmo quadro tem o monstro e o texto de ação
        mx, my = pyautogui.position()
        img = tela.capturar()
        solta("F9")
        if problema:
            print(f"Exemplo recusado: {problema}")
            continue
        pos = (mx - tela.REGIAO["left"], my - tela.REGIAO["top"])
        try:
            msg = e.adicionar(img, pos, visao.recortar(img, visao.FAIXA_TEXTO_ACAO))
        except Exception as erro:  # um exemplo ruim não pode derrubar o ensino inteiro
            msg = f"exemplo recusado: erro ao aprender ({erro!r})"
        print(msg)
        if e.perfil is not None and not msg.startswith("exemplo recusado"):
            print(f"Confira: {e.conferir()}")


def principal():
    a = _args()
    e = Ensino(a.nome, a.nome_jogo or a.nome, a.espera_loot,
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
