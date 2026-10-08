import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from comparacao import calcular_ssim, mse, psnr
from filtros_classicos import aplicar_filtro_classico
from metodos_edp import aplicar_metodo_edp
from ruido import degradacao_inpainting, ruido_gaussiano, ruido_sal_pimenta, ruido_speckle


EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


PARAMETROS_EDP_PADRAO = {
    "isotropica_iteracoes": 40,
    "isotropica_alpha": 0.2,
    "perona_iteracoes": 60,
    "perona_kappa": 25,
    "perona_gamma": 0.18,
    "navier_iteracoes": 250,
    "navier_dt": 0.15,
    "navier_suavizacao": 0.08,
    "opencv_raio": 3,
}


def normalizar_parametros_edp(parametros=None):
    resultado = PARAMETROS_EDP_PADRAO.copy()
    if parametros:
        resultado.update(parametros)
    return resultado


def descrever_parametros_edp(parametros):
    return {
        "isotropica": (
            f"iteracoes={parametros['isotropica_iteracoes']} "
            f"alpha={parametros['isotropica_alpha']}"
        ),
        "perona": (
            f"iteracoes={parametros['perona_iteracoes']} "
            f"kappa={parametros['perona_kappa']} "
            f"gamma={parametros['perona_gamma']}"
        ),
        "navier": (
            f"iteracoes={parametros['navier_iteracoes']} "
            f"dt={parametros['navier_dt']} "
            f"suavizacao={parametros['navier_suavizacao']}"
        ),
        "opencv": f"raio={parametros['opencv_raio']} cv2.INPAINT_NS",
    }


def listar_imagens(pasta_img="img"):
    pasta = Path(pasta_img)
    if not pasta.exists():
        return []

    return sorted(
        caminho for caminho in pasta.iterdir()
        if caminho.suffix.lower() in EXTENSOES_IMAGEM
    )


def _salvar_imagem(caminho, img):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite falha em silencio com acentos no caminho (Windows)
    ok, buffer = cv2.imencode(caminho.suffix or ".png", img)
    if not ok:
        raise ValueError(f"Nao foi possivel codificar a imagem {caminho.name}.")
    buffer.tofile(str(caminho))


def _avaliar(img_original, img_resultado):
    return {
        "mse": mse(img_original, img_resultado),
        "psnr": psnr(img_original, img_resultado),
        "ssim": calcular_ssim(img_original, img_resultado),
    }


def _metodos_ruido(parametros_edp):
    descricoes = descrever_parametros_edp(parametros_edp)
    return [
        ("Media", "filtro_classico", "kernel=5x5", lambda img, mask: aplicar_filtro_classico(img, "1")),
        ("Gaussiano", "filtro_classico", "kernel=3x3 sigma=0.8", lambda img, mask: aplicar_filtro_classico(img, "2")),
        ("Mediana", "filtro_classico", "kernel=3x3", lambda img, mask: aplicar_filtro_classico(img, "3")),
        ("Laplaciano", "filtro_classico", "realce de bordas", lambda img, mask: aplicar_filtro_classico(img, "4")),
        ("Difusao isotropica", "edp", descricoes["isotropica"], lambda img, mask: aplicar_metodo_edp(img, "1", parametros=parametros_edp)),
        ("Perona-Malik", "edp", descricoes["perona"], lambda img, mask: aplicar_metodo_edp(img, "2", parametros=parametros_edp)),
    ]


def _metodos_inpainting(parametros_edp):
    descricoes = descrever_parametros_edp(parametros_edp)
    return [
        ("Media", "filtro_classico", "baseline kernel=5x5", lambda img, mask: aplicar_filtro_classico(img, "1")),
        ("Mediana", "filtro_classico", "baseline kernel=3x3", lambda img, mask: aplicar_filtro_classico(img, "3")),
        ("Navier-Stokes", "edp", descricoes["navier"], lambda img, mask: aplicar_metodo_edp(img, "3", mask, parametros_edp)),
        ("Navier OpenCV", "referencia", descricoes["opencv"], lambda img, mask: aplicar_metodo_edp(img, "4", mask, parametros_edp)),
    ]


def _criar_grid(imagens_com_titulos, tamanho=(260, 210), colunas=3):
    quadros = []

    for titulo, img in imagens_com_titulos:
        quadro = cv2.resize(img, tamanho)
        if len(quadro.shape) == 2:
            quadro = cv2.cvtColor(quadro, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(quadro, (0, 0), (tamanho[0], 30), (0, 0, 0), -1)
        cv2.putText(
            quadro,
            titulo[:24],
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        quadros.append(quadro)

    while len(quadros) % colunas != 0:
        quadros.append(np.zeros_like(quadros[0]))

    linhas = []
    for i in range(0, len(quadros), colunas):
        linhas.append(cv2.hconcat(quadros[i:i + colunas]))

    return cv2.vconcat(linhas)


def _registrar_resultado(linhas_csv, imagem_nome, experimento, metodo, categoria, parametros, metricas, caminho_resultado):
    linhas_csv.append({
        "imagem": imagem_nome,
        "experimento": experimento,
        "metodo": metodo,
        "categoria": categoria,
        "parametros": parametros,
        "mse": metricas["mse"],
        "psnr": metricas["psnr"],
        "ssim": metricas["ssim"],
        "arquivo_resultado": str(caminho_resultado),
    })


def _linha_csv_formatada(linha):
    return {
        **linha,
        "mse": f"{linha['mse']:.4f}",
        "psnr": f"{linha['psnr']:.4f}",
        "ssim": f"{linha['ssim']:.6f}",
    }


def _salvar_csv(caminho, linhas, campos):
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


def _tipo_experimento(nome_experimento):
    if nome_experimento.startswith("inpainting"):
        return "inpainting"
    return "ruido"


def _media(valores):
    if not valores:
        return 0.0
    return sum(valores) / len(valores)


def _agrupar_linhas(linhas, chaves):
    grupos = {}
    for linha in linhas:
        chave = tuple(linha[campo] for campo in chaves)
        grupos.setdefault(chave, []).append(linha)
    return grupos


def _gerar_medias_por_metodo(linhas):
    medias = []
    grupos = _agrupar_linhas(linhas, ["experimento", "metodo"])

    for (experimento, metodo), itens in sorted(grupos.items()):
        medias.append({
            "tipo_experimento": _tipo_experimento(experimento),
            "experimento": experimento,
            "metodo": metodo,
            "quantidade": len(itens),
            "mse_medio": _media([item["mse"] for item in itens]),
            "psnr_medio": _media([item["psnr"] for item in itens]),
            "ssim_medio": _media([item["ssim"] for item in itens]),
        })

    return medias


def _gerar_melhores_por_imagem(linhas):
    melhores = []
    grupos = _agrupar_linhas(linhas, ["imagem", "experimento"])

    for (imagem, experimento), itens in sorted(grupos.items()):
        melhor_mse = min(itens, key=lambda item: item["mse"])
        melhor_psnr = max(itens, key=lambda item: item["psnr"])
        melhor_ssim = max(itens, key=lambda item: item["ssim"])

        melhores.append({
            "tipo_experimento": _tipo_experimento(experimento),
            "imagem": imagem,
            "experimento": experimento,
            "melhor_mse": melhor_mse["metodo"],
            "valor_mse": melhor_mse["mse"],
            "melhor_psnr": melhor_psnr["metodo"],
            "valor_psnr": melhor_psnr["psnr"],
            "melhor_ssim": melhor_ssim["metodo"],
            "valor_ssim": melhor_ssim["ssim"],
        })

    return melhores


def _gerar_ranking_por_experimento(linhas):
    ranking = []
    medias = _gerar_medias_por_metodo(linhas)
    grupos = _agrupar_linhas(medias, ["experimento"])

    for (experimento,), itens in sorted(grupos.items()):
        ordenados = sorted(
            itens,
            key=lambda item: (item["ssim_medio"], item["psnr_medio"], -item["mse_medio"]),
            reverse=True,
        )

        for posicao, item in enumerate(ordenados, start=1):
            ranking.append({
                "tipo_experimento": item["tipo_experimento"],
                "experimento": experimento,
                "posicao": posicao,
                "metodo": item["metodo"],
                "mse_medio": item["mse_medio"],
                "psnr_medio": item["psnr_medio"],
                "ssim_medio": item["ssim_medio"],
            })

    return ranking


def _formatar_linha_medias(linha):
    return {
        **linha,
        "mse_medio": f"{linha['mse_medio']:.4f}",
        "psnr_medio": f"{linha['psnr_medio']:.4f}",
        "ssim_medio": f"{linha['ssim_medio']:.6f}",
    }


def _formatar_linha_melhores(linha):
    return {
        **linha,
        "valor_mse": f"{linha['valor_mse']:.4f}",
        "valor_psnr": f"{linha['valor_psnr']:.4f}",
        "valor_ssim": f"{linha['valor_ssim']:.6f}",
    }


def _formatar_linha_ranking(linha):
    return {
        **linha,
        "mse_medio": f"{linha['mse_medio']:.4f}",
        "psnr_medio": f"{linha['psnr_medio']:.4f}",
        "ssim_medio": f"{linha['ssim_medio']:.6f}",
    }


def _salvar_resumos(saida, linhas_csv):
    campos_metricas = [
        "imagem",
        "experimento",
        "metodo",
        "categoria",
        "parametros",
        "mse",
        "psnr",
        "ssim",
        "arquivo_resultado",
    ]

    linhas_ruido = [linha for linha in linhas_csv if _tipo_experimento(linha["experimento"]) == "ruido"]
    linhas_inpainting = [linha for linha in linhas_csv if _tipo_experimento(linha["experimento"]) == "inpainting"]

    _salvar_csv(saida / "metricas.csv", [_linha_csv_formatada(linha) for linha in linhas_csv], campos_metricas)
    _salvar_csv(saida / "metricas_ruido.csv", [_linha_csv_formatada(linha) for linha in linhas_ruido], campos_metricas)
    _salvar_csv(saida / "metricas_inpainting.csv", [_linha_csv_formatada(linha) for linha in linhas_inpainting], campos_metricas)

    medias = _gerar_medias_por_metodo(linhas_csv)
    melhores = _gerar_melhores_por_imagem(linhas_csv)
    ranking = _gerar_ranking_por_experimento(linhas_csv)

    campos_medias = [
        "tipo_experimento",
        "experimento",
        "metodo",
        "quantidade",
        "mse_medio",
        "psnr_medio",
        "ssim_medio",
    ]
    campos_melhores = [
        "tipo_experimento",
        "imagem",
        "experimento",
        "melhor_mse",
        "valor_mse",
        "melhor_psnr",
        "valor_psnr",
        "melhor_ssim",
        "valor_ssim",
    ]
    campos_ranking = [
        "tipo_experimento",
        "experimento",
        "posicao",
        "metodo",
        "mse_medio",
        "psnr_medio",
        "ssim_medio",
    ]

    _salvar_csv(saida / "medias_por_metodo.csv", [_formatar_linha_medias(linha) for linha in medias], campos_medias)
    _salvar_csv(saida / "melhores_por_imagem.csv", [_formatar_linha_melhores(linha) for linha in melhores], campos_melhores)
    _salvar_csv(saida / "ranking_por_experimento.csv", [_formatar_linha_ranking(linha) for linha in ranking], campos_ranking)

    _salvar_relatorio_texto(saida / "resumo_resultados.txt", medias, melhores, ranking)


def _salvar_relatorio_texto(caminho, medias, melhores, ranking):
    linhas = [
        "Resumo dos resultados",
        "",
        "Arquivos gerados:",
        "- metricas.csv: todos os resultados individuais.",
        "- metricas_ruido.csv: apenas experimentos com ruido.",
        "- metricas_inpainting.csv: apenas experimentos de inpainting.",
        "- medias_por_metodo.csv: media de MSE, PSNR e SSIM por metodo.",
        "- melhores_por_imagem.csv: melhores metodos por imagem e experimento.",
        "- ranking_por_experimento.csv: ranking medio por experimento.",
        "",
        "Ranking medio por experimento:",
    ]

    experimento_atual = None
    for item in ranking:
        if item["experimento"] != experimento_atual:
            experimento_atual = item["experimento"]
            linhas.append("")
            linhas.append(f"{experimento_atual} ({item['tipo_experimento']}):")

        linhas.append(
            f"{item['posicao']}. {item['metodo']} - "
            f"MSE medio={item['mse_medio']:.4f}, "
            f"PSNR medio={item['psnr_medio']:.4f}, "
            f"SSIM medio={item['ssim_medio']:.6f}"
        )

    linhas.append("")
    linhas.append("Vencedores gerais por experimento:")
    experimento_atual = None
    for item in ranking:
        if item["posicao"] != 1:
            continue

        experimento_atual = item["experimento"]
        linhas.append(
            f"- {experimento_atual}: {item['metodo']} "
            f"(MSE medio={item['mse_medio']:.4f}, "
            f"PSNR medio={item['psnr_medio']:.4f}, "
            f"SSIM medio={item['ssim_medio']:.6f})"
        )

    linhas.append("")
    linhas.append("Melhores resultados por imagem:")
    linhas.append("Criterio principal: maior SSIM; MSE e PSNR ficam registrados para conferencia.")

    imagem_atual = None
    for item in melhores:
        if item["imagem"] != imagem_atual:
            imagem_atual = item["imagem"]
            linhas.append("")
            linhas.append(f"Imagem: {imagem_atual}")

        linhas.append(
            f"- {item['experimento']}: melhor SSIM={item['melhor_ssim']} "
            f"({item['valor_ssim']:.6f}); "
            f"menor MSE={item['melhor_mse']} ({item['valor_mse']:.4f}); "
            f"maior PSNR={item['melhor_psnr']} ({item['valor_psnr']:.4f})"
        )

    caminho.write_text("\n".join(linhas), encoding="utf-8")


def executar_experimentos(pasta_img="img", pasta_saida="resultados_experimentos", parametros_edp=None, imagens_entrada=None, descricao_imagens=None):
    parametros_edp = normalizar_parametros_edp(parametros_edp)
    if imagens_entrada is None:
        imagens = [(caminho.name, cv2.imread(str(caminho))) for caminho in listar_imagens(pasta_img)]
        descricao_imagens = descricao_imagens or f"pasta {pasta_img}"
    else:
        imagens = imagens_entrada
        descricao_imagens = descricao_imagens or "imagens selecionadas pela interface"

    if not imagens:
        raise FileNotFoundError("Nenhuma imagem foi encontrada para executar os experimentos.")

    raiz_saida = Path(pasta_saida)
    rodada = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida = raiz_saida / rodada
    saida.mkdir(parents=True, exist_ok=True)

    linhas_csv = []
    total_resultados = 0

    imagens_validas = 0

    for nome_arquivo, img_original in imagens:
        if img_original is None:
            continue

        imagens_validas += 1
        nome_img = Path(nome_arquivo).stem
        pasta_imagem = saida / nome_img
        _salvar_imagem(pasta_imagem / "original.png", img_original)

        experimentos = [
            ("ruido_gaussiano", ruido_gaussiano(img_original), None, _metodos_ruido(parametros_edp)),
            ("ruido_impulsivo", ruido_sal_pimenta(img_original), None, _metodos_ruido(parametros_edp)),
            ("ruido_speckle", ruido_speckle(img_original), None, _metodos_ruido(parametros_edp)),
        ]

        img_inpaint, mask = degradacao_inpainting(img_original)
        experimentos.append(("inpainting_regiao_removida", img_inpaint, mask, _metodos_inpainting(parametros_edp)))

        for nome_exp, img_degradada, mask_exp, metodos in experimentos:
            pasta_exp = pasta_imagem / nome_exp
            pasta_exp.mkdir(parents=True, exist_ok=True)

            _salvar_imagem(pasta_exp / "degradada.png", img_degradada)
            grid_itens = [("Original", img_original), ("Degradada", img_degradada)]

            if mask_exp is not None:
                _salvar_imagem(pasta_exp / "mascara.png", mask_exp)
                grid_itens.append(("Mascara", cv2.cvtColor(mask_exp, cv2.COLOR_GRAY2BGR)))

            for metodo, categoria, parametros, funcao in metodos:
                resultado = funcao(img_degradada, mask_exp)
                nome_saida = metodo.lower().replace(" ", "_").replace("-", "_") + ".png"
                caminho_resultado = pasta_exp / nome_saida
                _salvar_imagem(caminho_resultado, resultado)

                metricas = _avaliar(img_original, resultado)
                _registrar_resultado(
                    linhas_csv,
                    nome_img,
                    nome_exp,
                    metodo,
                    categoria,
                    parametros,
                    metricas,
                    caminho_resultado,
                )

                grid_itens.append((metodo, resultado))
                total_resultados += 1

            grid = _criar_grid(grid_itens)
            _salvar_imagem(pasta_exp / "comparacao_grid.png", grid)

    if imagens_validas == 0:
        raise FileNotFoundError("Nenhuma imagem valida foi encontrada para executar os experimentos.")

    _salvar_resumos(saida, linhas_csv)
    caminho_csv = saida / "metricas.csv"

    caminho_parametros = saida / "parametros_experimento.txt"
    caminho_parametros.write_text(
        "\n".join([
            "Experimentos do TCC",
            f"Imagens: {descricao_imagens}",
            "Degradacoes: ruido_gaussiano, ruido_impulsivo, ruido_speckle, inpainting_regiao_removida",
            "Filtros classicos: Media, Gaussiano, Laplaciano, Mediana",
            f"Difusao isotropica: {descrever_parametros_edp(parametros_edp)['isotropica']}",
            f"Perona-Malik: {descrever_parametros_edp(parametros_edp)['perona']}",
            f"Navier-Stokes: {descrever_parametros_edp(parametros_edp)['navier']}",
            "Referencia adicional: Navier OpenCV",
            f"Navier OpenCV: {descrever_parametros_edp(parametros_edp)['opencv']}",
            "Metricas: MSE, PSNR, SSIM",
        ]),
        encoding="utf-8",
    )

    return {
        "pasta_saida": str(saida),
        "csv": str(caminho_csv),
        "resumo": str(saida / "resumo_resultados.txt"),
        "imagens": imagens_validas,
        "resultados": total_resultados,
    }
