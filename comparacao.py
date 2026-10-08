import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from filtros_classicos import aplicar_filtro_classico
from metodos_edp import aplicar_metodo_edp
from ruido import aplicar_degradacao_com_mascara


# =========================
# METRICAS
# =========================

def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(m))

def calcular_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)


# =========================
# MÉTODOS
# =========================

def get_metodos(parametros_edp=None):
    if parametros_edp is None:
        parametros_edp = {}

    return {
        "Media": lambda img, mask=None: aplicar_filtro_classico(img, "1"),
        "Gaussiano": lambda img, mask=None: aplicar_filtro_classico(img, "2"),
        "Mediana": lambda img, mask=None: aplicar_filtro_classico(img, "3"),
        "Laplaciano": lambda img, mask=None: aplicar_filtro_classico(img, "4"),
        "Difusao": lambda img, mask=None: aplicar_metodo_edp(img, "1", parametros=parametros_edp),
        "Perona-Malik": lambda img, mask=None: aplicar_metodo_edp(img, "2", parametros=parametros_edp),
        "Navier-Stokes": lambda img, mask=None: aplicar_metodo_edp(img, "3", mask, parametros_edp),
        "Navier OpenCV": lambda img, mask=None: aplicar_metodo_edp(img, "4", mask, parametros_edp),
    }


# =========================
# FUNÇÃO PRINCIPAL
# =========================

def executar_comparacao(img_original, metodos_selecionados, nivel_degradacao, parametros_edp=None):

    # 🔥 agora usa o ruído escolhido pelo usuário
    img_ruido, mask = aplicar_degradacao_com_mascara(img_original, nivel_degradacao)

    resultados = {}
    resultados["Ruido"] = img_ruido

    if mask is not None:
        resultados["Mascara"] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # aplicar métodos no ruído
    for nome in metodos_selecionados:
        metodo = get_metodos(parametros_edp)[nome]
        resultados[nome] = metodo(img_ruido, mask)

    # =========================
    # GERAR GRID
    # =========================
    imagens = []

    # original
    base = cv2.resize(img_original, (250, 200))
    cv2.rectangle(base, (0, 0), (250, 30), (0, 0, 0), -1)
    cv2.putText(base, "Original", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    imagens.append(base)

    for nome, img in resultados.items():
        img_r = cv2.resize(img, (250, 200))

        if len(img_r.shape) == 2:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(img_r, (0, 0), (250, 30), (0, 0, 0), -1)
        cv2.putText(img_r, nome, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        imagens.append(img_r)

    # montar grid sem quadrado preto
    linhas = []
    for i in range(0, len(imagens), 3):
        linha = imagens[i:i+3]

        # 🔥 completa a linha com imagem preta
        while len(linha) < 3:
            linha.append(np.zeros_like(imagens[0]))

        linhas.append(cv2.hconcat(linha))

    grid = cv2.vconcat(linhas)

# =========================
# MÉTRICAS
# =========================

    dados_tabela = []

    for nome, img in resultados.items():
        if nome in ["Ruido", "Mascara"]:
            continue

        m = mse(img_original, img)
        p = psnr(img_original, img)
        s = calcular_ssim(img_original, img)

        dados_tabela.append((
            nome,
            f"{m:.2f}",
            f"{p:.2f}",
            f"{s:.4f}"
        ))

    # agora retorna dados estruturados (não string!)
    return grid, dados_tabela
