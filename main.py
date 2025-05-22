import os
import cv2
import time
import threading
import numpy as np
import pyautogui
import keyboard
from mss import mss
import pygetwindow as gw

# ========== Configurações ==========
TEMPLATE_DIR = "templates"

TEMPLATES = {
    "frog_texts": ["giant_frog_text.png", "giant_frog_text_2.png"],
    "hover_items": ["big_bones_hover_new.png", "adamant_arrow_hover.png"],
    "bury": "bury_big_bones.png",
    "big_bones_text": "big_bones_text.png",
    "inventory_bone": "bone_icon.png",
    "block": "big_frog_text.png",
    "life_bar": "barra_vermelha_estendida.png",
}

TEMPLATES_LOADED = {
    key: (
        [cv2.imread(os.path.join(TEMPLATE_DIR, f)) for f in value]
        if isinstance(value, list)
        else cv2.imread(os.path.join(TEMPLATE_DIR, value))
    )
    for key, value in TEMPLATES.items()
}

HUNTER_LOWER = np.array([14, 55, 45])
HUNTER_UPPER = np.array([32, 205, 172])
GRASS_LOWER = np.array([35, 40, 40])
GRASS_UPPER = np.array([95, 255, 255])

DELAY_CICLO = 0.01
TEMPO_MOUSE_PARADO = 1.0
OFFSET_X_OSSO = 20
TEMPO_ATAQUE = 5.0

burying = threading.Event()
ossos_ativos = threading.Event()
ossos_ativos.set()
acao_lock = threading.Lock()
mouse_parado_desde = time.time()
pos_anterior = pyautogui.position()

# Detectar janela do RuneScape
rs_window = gw.getWindowsWithTitle("Old School RuneScape")
if not rs_window:
    raise Exception(
        "Janela do RuneScape não encontrada. Certifique-se que o jogo está aberto."
    )
rs_win = rs_window[0]

CAPTURE_REGION = {
    "left": rs_win.left,
    "top": rs_win.top,
    "width": rs_win.width,
    "height": rs_win.height,
}

CENTRO_JANELA = (rs_win.left + rs_win.width // 2, rs_win.top + rs_win.height // 2)


# ========== Funções ==========
def tempo_mouse_parado():
    global mouse_parado_desde, pos_anterior
    atual = pyautogui.position()
    if np.linalg.norm(np.subtract(atual, pos_anterior)) > 2:
        mouse_parado_desde = time.time()
        pos_anterior = atual
    return time.time() - mouse_parado_desde >= TEMPO_MOUSE_PARADO


def distancia(p1, p2):
    return np.linalg.norm(np.subtract(p1, p2))


def capturar_tela():
    with mss() as sct:
        img = np.array(sct.grab(CAPTURE_REGION))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def detectar_template(img, templates, threshold=0.5):
    if not isinstance(templates, list):
        templates = [templates]
    for template in templates:
        if template is None:
            continue
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            h, w = template.shape[:2]
            return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return None


def detectar_multiplos(img, template, threshold=0.5):
    if template is None:
        return []
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    h, w = template.shape[:2]
    return [(pt[0] + w // 2, pt[1] + h // 2) for pt in zip(*loc[::-1])]


def detectar_hunters(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_kebbit = cv2.inRange(hsv, HUNTER_LOWER, HUNTER_UPPER)
    mask_grass = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
    mask = cv2.bitwise_and(mask_kebbit, cv2.bitwise_not(mask_grass))
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hunters = []
    for c in contours:
        area = cv2.contourArea(c)
        if 80 < area < 8000:
            M = cv2.moments(c)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                hunters.append((x, y))
    return hunters


def barra_vida_visivel():
    img = capturar_tela()
    pos = detectar_template(img, TEMPLATES_LOADED["life_bar"], threshold=0.8)
    return pos is not None


def esperar_barra_sumir(timeout=10):
    inicio = time.time()
    while time.time() - inicio < timeout:
        if not barra_vida_visivel():
            return True
        time.sleep(0.5)
    return False


# ========== Loops ==========
def loop_detectar_bury():
    tempo_sem_bury = 0
    while True:
        if not ossos_ativos.is_set():
            time.sleep(0.1)
            continue
        img = capturar_tela()
        pos = detectar_template(img, TEMPLATES_LOADED["bury"])
        if pos:
            with acao_lock:
                if not burying.is_set():
                    print("🩴 Enterrando ossos...")
                    burying.set()
                pyautogui.click()
                tempo_sem_bury = 0
        elif burying.is_set():
            tempo_sem_bury += 0.1
            if tempo_sem_bury >= 1.5:
                burying.clear()
                tempo_sem_bury = 0
        time.sleep(0.05)


def loop_principal():
    sapos_confirmados = 0
    bury_confirmados = 0
    alvo_atual = None
    while True:
        if not tempo_mouse_parado() or burying.is_set():
            time.sleep(DELAY_CICLO)
            continue

        img = capturar_tela()
        centro = (CAPTURE_REGION["width"] // 2, CAPTURE_REGION["height"] // 2)
        sapos = detectar_hunters(img)
        pos_osso = detectar_template(img, TEMPLATES_LOADED["big_bones_text"])
        ossos = []
        if pos_osso:
            ossos.append(("ground", (pos_osso[0] + OFFSET_X_OSSO, pos_osso[1])))

        pos_inv = detectar_multiplos(
            img, TEMPLATES_LOADED["inventory_bone"], threshold=0.8
        )

        if len(pos_inv) >= 6:
            ossos.extend(("inventory", pos) for pos in pos_inv)

        candidatos = [("frog", p, distancia(centro, p)) for p in sapos] + [
            (tipo, p, distancia(centro, p)) for tipo, p in ossos
        ]

        if alvo_atual:
            candidatos.insert(0, alvo_atual)

        while candidatos:
            candidatos.sort(key=lambda x: x[2])
            tipo, pos, _ = candidatos.pop(0)

            with acao_lock:
                pyautogui.moveTo(pos)
                time.sleep(0.2)
                nova_img = capturar_tela()

                if tipo == "frog":
                    if detectar_template(
                        nova_img, TEMPLATES_LOADED["frog_texts"], threshold=0.6
                    ):
                        pyautogui.click()
                        sapos_confirmados += 1
                        print(
                            f"🐸 Total de sapos confirmados e atacados: {sapos_confirmados}"
                        )
                        alvo_atual = ("frog", pos, 0)
                        esperar_barra_sumir(timeout=10)
                        alvo_atual = None
                        break
                    elif detectar_template(
                        nova_img, TEMPLATES_LOADED["block"], threshold=0.5
                    ):
                        alvo_atual = None
                        break

                elif tipo == "ground":
                    if detectar_template(nova_img, TEMPLATES_LOADED["hover_items"]):
                        pyautogui.click()
                        alvo_atual = ("ground", pos, 0)
                        break

                elif tipo == "inventory":
                    for pos_osso in pos_inv:
                        pyautogui.moveTo(pos_osso)
                        time.sleep(0.1)
                        pyautogui.click()
                        bury_confirmados += 1
                        print(f"🦴 Total de ossos enterrados: {bury_confirmados}")
                    alvo_atual = None
                    break
        time.sleep(DELAY_CICLO)


def loop_atalhos():
    while True:
        if keyboard.is_pressed("F8"):
            if ossos_ativos.is_set():
                ossos_ativos.clear()
                print("🩴 Coleta de ossos DESATIVADA!")
            else:
                ossos_ativos.set()
                print("🩴 Coleta de ossos ATIVADA!")
            time.sleep(0.5)
        time.sleep(0.05)


# ========== Execução ==========
print("🔁 Bot iniciado... Pressione CTRL+C para parar.")
try:
    threading.Thread(target=loop_detectar_bury, daemon=True).start()
    threading.Thread(target=loop_principal, daemon=True).start()
    threading.Thread(target=loop_atalhos, daemon=True).start()
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Execução finalizada pelo usuário.")
