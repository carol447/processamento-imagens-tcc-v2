import cv2


def aplicar_filtro_classico(img, tipo):

    # =========================
    # 1 - MÉDIA (Box Filter)
    # =========================
    if tipo == "1":
        return cv2.blur(img, (5, 5))

    # =========================
    # 2 - GAUSSIANO
    # =========================
    elif tipo == "2":
        return cv2.GaussianBlur(img, (3, 3), 0.8)

    # =========================
    # 3 - MEDIANA (não linear)
    # =========================
    elif tipo == "3":
        return cv2.medianBlur(img, 3)

    # =========================
    # 4 - LAPLACIANO (segunda derivada)
    # =========================
    elif tipo == "4":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

    return img