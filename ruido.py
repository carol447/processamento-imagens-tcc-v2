import numpy as np


# =========================
# RUIDOS CONTROLADOS
# =========================
def ruido_gaussiano(img, sigma=22):
    ruido = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + ruido
    return np.clip(noisy, 0, 255).astype(np.uint8)


def ruido_sal_pimenta(img, prob=0.02):
    noisy = img.copy()
    rnd = np.random.rand(*img.shape[:2])
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy


def ruido_speckle(img, intensidade=0.12):
    ruido = np.random.randn(*img.shape)
    noisy = img.astype(np.float32) + img.astype(np.float32) * ruido * intensidade
    return np.clip(noisy, 0, 255).astype(np.uint8)


def ruido_gaussiano_forte(img):
    return ruido_gaussiano(img, sigma=32)


# =========================
# DEGRADACOES ESTRUTURAIS
# =========================
def ruido_listras(img):
    noisy = img.copy()
    h = img.shape[0]

    for i in range(0, h, 18):
        noisy[i:i+1, :] = 255

    return noisy


def ruido_regiao(img):
    noisy = img.copy()
    h, w = img.shape[:2]
    noisy[h//4:h//2, w//4:w//2] = 0
    return noisy


def ruido_regiao_com_mascara(img):
    noisy = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w = img.shape[:2]

    y0, y1 = h // 4, h // 2
    x0, x1 = w // 4, w // 2

    noisy[y0:y1, x0:x1] = 0
    mask[y0:y1, x0:x1] = 255

    return noisy, mask


def piorar_imagem(img):
    noisy = img.astype(np.float32)
    h, w = img.shape[:2]

    noisy += np.random.normal(0, 18, img.shape)
    noisy += noisy * np.random.randn(*img.shape) * 0.08

    for i in range(0, h, 25):
        noisy[i:i+1, :] = 255

    for _ in range(2):
        x = np.random.randint(0, max(1, w - 30))
        y = np.random.randint(0, max(1, h - 30))
        noisy[y:y+25, x:x+25] = 0

    return np.clip(noisy, 0, 255).astype(np.uint8)


def degradacao_inpainting(img):
    noisy = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w = img.shape[:2]

    x = w // 3
    noisy[h // 5:4 * h // 5, x:x + 4] = 0
    mask[h // 5:4 * h // 5, x:x + 4] = 255

    y0, y1 = h // 3, h // 3 + max(12, h // 8)
    x0, x1 = w // 2, w // 2 + max(12, w // 8)
    noisy[y0:y1, x0:x1] = 0
    mask[y0:y1, x0:x1] = 255

    return noisy, mask


# =========================
# FUNCAO PRINCIPAL
# =========================
def aplicar_degradacao(img, nivel):
    img = img.copy().astype(np.float32)

    if nivel == "1":
        img += np.random.normal(0, 28, img.shape)

    elif nivel == "2":
        img += np.random.normal(0, 18, img.shape)
        img += img * np.random.randn(*img.shape) * 0.06

    elif nivel == "3":
        img += np.random.normal(0, 22, img.shape)

        for i in range(0, img.shape[0], 22):
            img[i:i+1, :] = 255

        h, w = img.shape[:2]
        img[h//3:h//3+30, w//3:w//3+30] = 0

    elif nivel == "4":
        img += np.random.normal(0, 22, img.shape)
        img += img * np.random.randn(*img.shape) * 0.04

    return np.clip(img, 0, 255).astype(np.uint8)


def aplicar_degradacao_com_mascara(img, nivel):
    if nivel == "5":
        return degradacao_inpainting(img)

    if nivel == "3":
        img_ruido = img.copy().astype(np.float32)
        img_ruido += np.random.normal(0, 22, img_ruido.shape)

        for i in range(0, img_ruido.shape[0], 22):
            img_ruido[i:i+1, :] = 255

        img_ruido = np.clip(img_ruido, 0, 255).astype(np.uint8)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        h, w = img.shape[:2]
        mask[h//3:h//3+30, w//3:w//3+30] = 255
        img_ruido[mask > 0] = 0
        return img_ruido, mask

    return aplicar_degradacao(img, nivel), None

