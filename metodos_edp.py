import cv2
import numpy as np


# =========================
# DIFUSÃO ISOTROPICA (equação do calor)
# =========================
def difusao_isotropica(img, iteracoes=40, alpha=0.2):

    img = img.astype(np.float32)

    for _ in range(iteracoes):
        if img.ndim == 3:
            p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
            norte = p[:-2, 1:-1, :]
            sul = p[2:, 1:-1, :]
            leste = p[1:-1, 2:, :]
            oeste = p[1:-1, :-2, :]
        else:
            p = np.pad(img, 1, mode="edge")
            norte = p[:-2, 1:-1]
            sul = p[2:, 1:-1]
            leste = p[1:-1, 2:]
            oeste = p[1:-1, :-2]

        lap = (
            norte +
            sul +
            leste +
            oeste -
            4 * img
        )
        img = img + alpha * lap

    return np.clip(img, 0, 255).astype(np.uint8)


# =========================
# DIFUSÃO ANISOTROPICA (Perona-Malik)
# =========================
def difusao_anisotropica(img, iteracoes=60, kappa=25, gamma=0.18):
    img = img.astype(np.float32)

    for _ in range(iteracoes):
        if img.ndim == 3:
            p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
            norte = p[:-2, 1:-1, :] - img
            sul = p[2:, 1:-1, :] - img
            leste = p[1:-1, 2:, :] - img
            oeste = p[1:-1, :-2, :] - img
        else:
            p = np.pad(img, 1, mode="edge")
            norte = p[:-2, 1:-1] - img
            sul = p[2:, 1:-1] - img
            leste = p[1:-1, 2:] - img
            oeste = p[1:-1, :-2] - img

        cN = np.exp(-(norte / kappa) ** 2)
        cS = np.exp(-(sul / kappa) ** 2)
        cL = np.exp(-(leste / kappa) ** 2)
        cO = np.exp(-(oeste / kappa) ** 2)

        img = img + gamma * (
            cN * norte +
            cS * sul +
            cL * leste +
            cO * oeste
        )

    return np.clip(img, 0, 255).astype(np.uint8)


# =========================
# INPAINTING NAVIER-STOKES
# Implementacao propria por diferencas finitas:
# I_t = grad_perp(I) . grad(Delta I)
# A atualizacao acontece iterativamente apenas dentro da mascara.
# =========================
def _normalizar_mascara(mask, tamanho):
    if mask is None:
        return None

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[:2] != tamanho:
        mask = cv2.resize(mask, (tamanho[1], tamanho[0]), interpolation=cv2.INTER_NEAREST)

    return mask > 0


def _laplaciano(canal):
    p = np.pad(canal, 1, mode="edge")
    return (
        p[:-2, 1:-1] +
        p[2:, 1:-1] +
        p[1:-1, :-2] +
        p[1:-1, 2:] -
        4 * canal
    )


def _gradiente_central(canal):
    p = np.pad(canal, 1, mode="edge")
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
    return gx, gy


def _inicializar_mascara_por_media(canal, mask_bin, iteracoes=80):
    saida = canal.astype(np.float32).copy()
    conhecida = ~mask_bin

    if np.any(conhecida):
        saida[mask_bin] = np.mean(saida[conhecida])

    for _ in range(iteracoes):
        p = np.pad(saida, 1, mode="edge")
        media_vizinhos = (
            p[:-2, 1:-1] +
            p[2:, 1:-1] +
            p[1:-1, :-2] +
            p[1:-1, 2:]
        ) * 0.25

        saida[mask_bin] = media_vizinhos[mask_bin]

    saida[conhecida] = canal[conhecida]
    return saida


def _navier_stokes_canal(canal, mask_bin, iteracoes, dt, suavizacao):
    original = canal.astype(np.float32)
    saida = _inicializar_mascara_por_media(original, mask_bin)

    for _ in range(iteracoes):
        anterior = saida.copy()

        lap = _laplaciano(anterior)
        ix, iy = _gradiente_central(anterior)
        lap_x, lap_y = _gradiente_central(lap)

        # Equacao: I_t = grad_perp(I) . grad(Delta I)
        transporte = (-iy * lap_x) + (ix * lap_y)

        max_transporte = np.max(np.abs(transporte[mask_bin]))
        if max_transporte > 1.0e-8:
            transporte = transporte / max_transporte

        difusao = _laplaciano(anterior)
        atualizacao = dt * (transporte + suavizacao * difusao)
        saida[mask_bin] = anterior[mask_bin] + atualizacao[mask_bin]
        saida[~mask_bin] = original[~mask_bin]
        saida = np.clip(saida, 0, 255)

    return saida


def inpainting_navier_stokes(img, mask, iteracoes=250, dt=0.15, suavizacao=0.08):
    mask_bin = _normalizar_mascara(mask, img.shape[:2])
    if mask_bin is None or not np.any(mask_bin):
        return img

    if img.ndim == 2:
        saida = _navier_stokes_canal(img, mask_bin, iteracoes, dt, suavizacao)
        return np.clip(saida, 0, 255).astype(np.uint8)

    saida = img.astype(np.float32).copy()

    for canal in range(img.shape[2]):
        saida[:, :, canal] = _navier_stokes_canal(
            img[:, :, canal],
            mask_bin,
            iteracoes,
            dt,
            suavizacao
        )

    return np.clip(saida, 0, 255).astype(np.uint8)


def inpainting_navier_stokes_opencv(img, mask, raio=3):
    mask_bin = _normalizar_mascara(mask, img.shape[:2])
    if mask_bin is None or not np.any(mask_bin):
        return img

    mask_uint8 = (mask_bin.astype(np.uint8)) * 255
    return cv2.inpaint(img, mask_uint8, raio, cv2.INPAINT_NS)


# =========================
# INTERFACE (OBRIGATORIA)
# =========================
def aplicar_metodo_edp(img, tipo, mask=None, parametros=None):
    if parametros is None:
        parametros = {}

    if tipo == "1":
        print("Difusão isotrópica")
        return difusao_isotropica(
            img,
            iteracoes=parametros.get("isotropica_iteracoes", 40),
            alpha=parametros.get("isotropica_alpha", 0.2)
        )

    elif tipo == "2":
        print("Perona-Malik")
        return difusao_anisotropica(
            img,
            iteracoes=parametros.get("perona_iteracoes", 60),
            kappa=parametros.get("perona_kappa", 25),
            gamma=parametros.get("perona_gamma", 0.18)
        )

    elif tipo == "3":
        print("Inpainting Navier-Stokes")
        return inpainting_navier_stokes(
            img,
            mask,
            iteracoes=parametros.get("navier_iteracoes", 250),
            dt=parametros.get("navier_dt", 0.15),
            suavizacao=parametros.get("navier_suavizacao", 0.08)
        )

    elif tipo == "4":
        print("Inpainting Navier-Stokes OpenCV")
        return inpainting_navier_stokes_opencv(
            img,
            mask,
            raio=parametros.get("opencv_raio", 3)
        )

    return img
