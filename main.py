import pyautogui
import cv2
import numpy as np
import time
import os
import threading

TEMPLATE_DIR = "templates"

SPO_TEMPLATES = [
    cv2.imread(os.path.join(TEMPLATE_DIR, "giant_frog_text.png")),
    cv2.imread(os.path.join(TEMPLATE_DIR, "giant_frog_text_2.png")),
]

HOVER_TEMPLATES = [
    cv2.imread(os.path.join(TEMPLATE_DIR, "big_bones_hover_new.png")),
    cv2.imread(os.path.join(TEMPLATE_DIR, "adamant_arrow_hover.png")),
]

BURY_TEMPLATE = cv2.imread(os.path.join(TEMPLATE_DIR, "bury_big_bones.png"))
BIG_BONES_TEXT_TEMPLATE = cv2.imread(os.path.join(TEMPLATE_DIR, "big_bones_text.png"))

FROG_LOWER = np.array([60, 200, 51])
FROG_UPPER = np.array([61, 214, 69])

burying = threading.Event()
acao_em_execucao = threading.Lock()


def detectar_template(img_bgr, templates, threshold=0.6):
    for template in templates:
        if template is None:
            continue
        res = cv2.matchTemplate(img_bgr, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            h, w = template.shape[:2]
            return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return None


def detectar_cores_sapos(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FROG_LOWER, FROG_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    locais = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 400 < area < 5000:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                locais.append((cx, cy))
    return locais


def loop_detectar_bury():
    tempo_sem_bury = 0
    while True:
        img_bgr = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        pos = detectar_template(img_bgr, [BURY_TEMPLATE])

        if pos:
            with acao_em_execucao:
                if not burying.is_set():
                    print("🦴 Enterrar os ossos! Clicando...")
                    burying.set()
                pyautogui.click()
                tempo_sem_bury = 0
        else:
            if burying.is_set():
                tempo_sem_bury += 0.2
                if tempo_sem_bury >= 2.0:
                    burying.clear()
                    tempo_sem_bury = 0
        time.sleep(0.2)


def loop_sapo_hsv():
    ultimo_click = 0
    cooldown = 8

    while True:
        if not burying.is_set():
            tempo_atual = time.time()
            if tempo_atual - ultimo_click < cooldown:
                time.sleep(0.2)
                continue

            img_bgr = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
            sapos = detectar_cores_sapos(img_bgr)
            for x, y in sapos:
                with acao_em_execucao:
                    pyautogui.moveTo(x, y)
                    time.sleep(0.2)
                    nova_img = cv2.cvtColor(
                        np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR
                    )
                    if detectar_template(nova_img, SPO_TEMPLATES):
                        print("🐸 Sapo confirmado por texto! Clicando...")
                        pyautogui.click()
                        ultimo_click = time.time()
                        break
                    else:
                        print("❌ Texto de sapo não confirmado. Pulando.")
        time.sleep(0.2)


def loop_ossos():
    while True:
        if not burying.is_set():
            img_bgr = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
            pos = detectar_template(img_bgr, [BIG_BONES_TEXT_TEMPLATE])
            if pos:
                with acao_em_execucao:
                    print("📍 Ossos no chão. Movendo cursor...")
                    pyautogui.moveTo(pos[0] + 20, pos[1])
                    time.sleep(0.3)
                    for _ in range(5):
                        confirm_img = cv2.cvtColor(
                            np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR
                        )
                        if detectar_template(confirm_img, HOVER_TEMPLATES):
                            print("✅ Hover confirmado! Clicando duas vezes...")
                            pyautogui.click()
                            time.sleep(0.1)
                            pyautogui.click()
                            print("⏳ Aguardando antes de voltar a detectar sapos...")
                            time.sleep(1.2)
                            break
                        time.sleep(0.1)
                    else:
                        print("❌ Hover não apareceu.")
        time.sleep(0.1)


# Início
print("🔁 Bot iniciado... Pressione CTRL+C para parar.")
try:
    threading.Thread(target=loop_detectar_bury, daemon=True).start()
    threading.Thread(target=loop_sapo_hsv, daemon=True).start()
    threading.Thread(target=loop_ossos, daemon=True).start()

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Execução finalizada pelo usuário.")
