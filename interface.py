import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
import cv2
from PIL import Image, ImageTk
from tkinter import ttk

from filtros_classicos import aplicar_filtro_classico
from metodos_edp import aplicar_metodo_edp
from comparacao import executar_comparacao
from experimentos import PARAMETROS_EDP_PADRAO, executar_experimentos


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Processamento de Imagens")
        self.root.geometry("1100x650")
        self.root.configure(bg="#f0f0f0")

        self.imagem_original = None
        self.imagem_atual = None
        self.caminho_imagem_original = None

        # =========================
        # FUNCOES DE ESTILO (FALTAVAM!)
        # =========================

        def criar_botao(texto, comando):
            return tk.Button(
                self.frame_botoes,
                text=texto,
                width=22,
                bg="#34495e",
                fg="white",
                bd=0,
                pady=6,
                command=comando,
                activebackground="#1abc9c"
            )

        def titulo(texto):
            return tk.Label(
                self.frame_botoes,
                text=texto,
                bg="#2c3e50",
                fg="#ecf0f1",
                font=("Arial", 10, "bold")
            )

        # =========================
        # FRAME ESQUERDA (MENU)
        # =========================
        self.frame_botoes = tk.Frame(root, bg="#2c3e50", width=250)
        self.frame_botoes.pack(side=tk.LEFT, fill=tk.Y)

        # =========================
        # SEPARADOR
        # =========================
        self.separador = tk.Frame(root, width=2, bg="#bdc3c7")
        self.separador.pack(side=tk.LEFT, fill=tk.Y)

        # =========================
        # FRAME DIREITA (IMAGEM)
        # =========================
        self.frame_imagem = tk.Frame(root, bg="#ecf0f1")
        self.frame_imagem.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # =========================
        # BOTÃ•ES
        # =========================
        criar_botao("Carregar Imagem", self.carregar_imagem).pack(pady=8)
        criar_botao("Salvar Imagem", self.salvar_imagem).pack(pady=5)

        titulo("FILTROS CLASSICOS").pack(pady=(15, 5))
        criar_botao("Media", lambda: self.aplicar_filtro("1")).pack()
        criar_botao("Gaussiano", lambda: self.aplicar_filtro("2")).pack()
        criar_botao("Mediana", lambda: self.aplicar_filtro("3")).pack()
        criar_botao("Laplaciano", lambda: self.aplicar_filtro("4")).pack()

        titulo("METODOS EDP").pack(pady=(15, 5))
        criar_botao("Difusao (Calor)", lambda: self.aplicar_edp("1")).pack()
        criar_botao("Perona-Malik", lambda: self.aplicar_edp("2")).pack()
        criar_botao("Navier-Stokes", lambda: self.aplicar_edp("3")).pack()

        titulo("COMPARACAO").pack(pady=(15, 5))
        criar_botao("Comparar Metodos", self.comparar_metodos).pack()
        criar_botao("Executar Experimentos", self.executar_experimentos_tcc).pack(pady=5)

        criar_botao("Resetar Imagem", self.resetar_imagem).pack(pady=20)

        # =========================
        # IMAGEM
        # =========================
        self.label_imagem = tk.Label(self.frame_imagem, bg="#ecf0f1")
        self.label_imagem.pack(expand=True)

    # =========================
    # FUNCOES
    # =========================
    def mostrar_tabela_metricas(self, dados):
        janela = tk.Toplevel(self.root)
        janela.title("Resultados das Metricas")
        janela.geometry("520x320")
        janela.configure(bg="#ecf0f1")

        colunas = ("Metodo", "MSE", "PSNR", "SSIM")

        tabela = ttk.Treeview(janela, columns=colunas, show="headings")

        # Estilo
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.configure("Treeview", font=("Arial", 10))

        # CabeÃ§alhos
        for col in colunas:
            tabela.heading(col, text=col)
            tabela.column(col, anchor="center", width=110)

        tabela.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Inserir dados
        for linha in dados:
            tabela.insert("", tk.END, values=linha)

        # =========================
        # LEGENDA
        # =========================
        legenda = tk.Label(
            janela,
            text=(
                "MSE: erro medio (menor e melhor)\n"
                "PSNR: qualidade do sinal (>30 bom, >40 excelente)\n"
                "SSIM: similaridade estrutural (0 a 1, ideal proximo de 1)"
            ),
            justify="left",
            bg="#ecf0f1",
            font=("Arial", 9)
        )
        legenda.pack(pady=5)

    def carregar_imagem(self):
        caminho = filedialog.askopenfilename()
        if not caminho:
            return

        img = cv2.imread(caminho)
        if img is None:
            messagebox.showerror("Erro", "Nao foi possivel carregar a imagem.")
            return

        self.imagem_original = img.copy()
        self.imagem_atual = img.copy()
        self.caminho_imagem_original = caminho
        self.mostrar_imagem(self.imagem_atual)

    def salvar_imagem(self):
        if self.imagem_atual is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada.")
            return

        caminho = filedialog.asksaveasfilename(defaultextension=".png")
        if caminho:
            cv2.imwrite(caminho, self.imagem_atual)
            messagebox.showinfo("Sucesso", "Imagem salva!")

    def resetar_imagem(self):
        if self.imagem_original is not None:
            self.imagem_atual = self.imagem_original.copy()
            self.mostrar_imagem(self.imagem_atual)

    def aplicar_filtro(self, tipo):
        if self.imagem_atual is None:
            return
        self.imagem_atual = aplicar_filtro_classico(self.imagem_atual, tipo)
        self.mostrar_imagem(self.imagem_atual)

    def configurar_parametros_edp(self, tipos):
        campos_por_tipo = {
            "1": [
                ("isotropica_iteracoes", "Iteracoes", PARAMETROS_EDP_PADRAO["isotropica_iteracoes"], int),
                ("isotropica_alpha", "Alpha", PARAMETROS_EDP_PADRAO["isotropica_alpha"], float),
            ],
            "2": [
                ("perona_iteracoes", "Iteracoes", PARAMETROS_EDP_PADRAO["perona_iteracoes"], int),
                ("perona_kappa", "Kappa", PARAMETROS_EDP_PADRAO["perona_kappa"], float),
                ("perona_gamma", "Gamma", PARAMETROS_EDP_PADRAO["perona_gamma"], float),
            ],
            "3": [
                ("navier_iteracoes", "Iteracoes", PARAMETROS_EDP_PADRAO["navier_iteracoes"], int),
                ("navier_dt", "Passo temporal dt", PARAMETROS_EDP_PADRAO["navier_dt"], float),
                ("navier_suavizacao", "Suavizacao", PARAMETROS_EDP_PADRAO["navier_suavizacao"], float),
            ],
            "4": [
                ("opencv_raio", "Raio OpenCV", PARAMETROS_EDP_PADRAO["opencv_raio"], int),
            ],
        }

        nomes_metodos = {
            "1": "Difusao isotropica",
            "2": "Perona-Malik",
            "3": "Navier-Stokes",
            "4": "Navier OpenCV",
        }

        janela = tk.Toplevel(self.root)
        janela.title("Parametros EDP")
        janela.geometry("430x420")
        janela.configure(bg="#ecf0f1")
        janela.transient(self.root)
        janela.grab_set()

        resultado = {"parametros": None}
        entradas = {}

        titulo = tk.Label(
            janela,
            text="Configurar parametros dos metodos EDP",
            bg="#ecf0f1",
            fg="#2c3e50",
            font=("Arial", 12, "bold")
        )
        titulo.pack(pady=(12, 4))

        aviso = tk.Label(
            janela,
            text="Edite todos os valores nesta janela ou mantenha os padroes.",
            bg="#ecf0f1",
            fg="#555555",
            font=("Arial", 9)
        )
        aviso.pack(pady=(0, 8))

        conteudo = tk.Frame(janela, bg="#ecf0f1")
        conteudo.pack(fill=tk.BOTH, expand=True, padx=18)

        linha = 0
        for tipo in tipos:
            if tipo not in campos_por_tipo:
                continue

            subtitulo = tk.Label(
                conteudo,
                text=nomes_metodos[tipo],
                bg="#ecf0f1",
                fg="#34495e",
                font=("Arial", 10, "bold")
            )
            subtitulo.grid(row=linha, column=0, columnspan=2, sticky="w", pady=(10, 4))
            linha += 1

            for chave, rotulo, padrao, conversor in campos_por_tipo[tipo]:
                label = tk.Label(conteudo, text=rotulo, bg="#ecf0f1", font=("Arial", 9))
                label.grid(row=linha, column=0, sticky="w", padx=(8, 8), pady=3)

                entrada = tk.Entry(conteudo, width=16)
                entrada.insert(0, str(padrao))
                entrada.grid(row=linha, column=1, sticky="w", pady=3)
                entradas[chave] = (entrada, conversor, padrao, rotulo)
                linha += 1

        def usar_padrao():
            resultado["parametros"] = {}
            janela.destroy()

        def cancelar():
            resultado["parametros"] = None
            janela.destroy()

        def confirmar():
            parametros = {}
            for chave, (entrada, conversor, padrao, rotulo) in entradas.items():
                texto = entrada.get().strip().replace(",", ".")
                if texto == "":
                    texto = str(padrao)

                try:
                    valor = conversor(texto)
                except ValueError:
                    messagebox.showerror("Erro", f"Valor invalido para {rotulo}.", parent=janela)
                    return

                if valor < 0:
                    messagebox.showerror("Erro", f"{rotulo} nao pode ser negativo.", parent=janela)
                    return

                parametros[chave] = valor

            resultado["parametros"] = parametros
            janela.destroy()

        botoes = tk.Frame(janela, bg="#ecf0f1")
        botoes.pack(fill=tk.X, padx=18, pady=12)

        tk.Button(botoes, text="Usar padroes", command=usar_padrao, width=13).pack(side=tk.LEFT)
        tk.Button(botoes, text="Cancelar", command=cancelar, width=10).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(botoes, text="Aplicar", command=confirmar, width=10, bg="#34495e", fg="white").pack(side=tk.RIGHT)

        janela.protocol("WM_DELETE_WINDOW", cancelar)
        self.root.wait_window(janela)

        return resultado["parametros"]
    def aplicar_edp(self, tipo):
        if self.imagem_atual is None:
            return

        mask = None
        if tipo == "3":
            caminho_mask = filedialog.askopenfilename(
                title="Selecione a mascara do inpainting"
            )
            if not caminho_mask:
                return

            mask = cv2.imread(caminho_mask, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                messagebox.showerror("Erro", "Nao foi possivel carregar a mascara.")
                return

        parametros = self.configurar_parametros_edp([tipo])
        if parametros is None:
            return
        self.imagem_atual = aplicar_metodo_edp(self.imagem_atual, tipo, mask, parametros)
        self.mostrar_imagem(self.imagem_atual)

    def mostrar_imagem(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((700, 500))

        img_tk = ImageTk.PhotoImage(img_pil)

        self.label_imagem.configure(image=img_tk)
        self.label_imagem.image = img_tk

    from comparacao import executar_comparacao

    from tkinter import simpledialog

    def comparar_metodos(self):
        if self.imagem_original is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return

        nivel = simpledialog.askstring(
            "Degradacao",
            "Escolha o nÃ­vel:\n"
            "1 - Gaussiano forte (EDP)\n"
            "2 - Medio\n"
            "3 - Complexo\n"
            "4 - EDP ideal\n"
            "5 - Inpainting Navier-Stokes"
        )

        if nivel not in ["1", "2", "3", "4", "5"]:
            return

        if nivel == "5":
            metodos = ["Media", "Mediana", "Navier-Stokes", "Navier OpenCV"]
            parametros = self.configurar_parametros_edp(["3", "4"])
        else:
            metodos = ["Media", "Gaussiano", "Mediana", "Difusao", "Perona-Malik"]
            parametros = self.configurar_parametros_edp(["1", "2"])

        if parametros is None:
            return

        grid, dados = executar_comparacao(self.imagem_original, metodos, nivel, parametros)

        self.mostrar_imagem(grid)
        self.mostrar_tabela_metricas(dados)

    def executar_experimentos_tcc(self):
        try:
            origem = simpledialog.askstring(
                "Origem das imagens",
                "Escolha a origem dos experimentos:\n"
                "1 - Todas as imagens da pasta img\n"
                "2 - Apenas a imagem carregada\n"
                "3 - Escolher uma pasta de imagens"
            )

            if origem not in ["1", "2", "3"]:
                return

            pasta_img = "img"
            imagens_entrada = None
            descricao_imagens = "pasta img"

            if origem == "2":
                if self.imagem_original is None:
                    messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
                    return

                nome_imagem = "imagem_carregada.png"
                if self.caminho_imagem_original:
                    nome_imagem = Path(self.caminho_imagem_original).name

                imagens_entrada = [(nome_imagem, self.imagem_original.copy())]
                descricao_imagens = f"imagem carregada ({nome_imagem})"

            elif origem == "3":
                pasta_escolhida = filedialog.askdirectory(title="Escolha a pasta de imagens")
                if not pasta_escolhida:
                    return

                pasta_img = pasta_escolhida
                descricao_imagens = f"pasta escolhida ({pasta_escolhida})"

            parametros = self.configurar_parametros_edp(["1", "2", "3", "4"])
            if parametros is None:
                return
            resumo = executar_experimentos(
                pasta_img=pasta_img,
                parametros_edp=parametros,
                imagens_entrada=imagens_entrada,
                descricao_imagens=descricao_imagens
            )
        except Exception as erro:
            messagebox.showerror("Erro", f"Nao foi possivel executar os experimentos:\n{erro}")
            return

        messagebox.showinfo(
            "Experimentos concluidos",
            "Experimentos salvos com sucesso.\n\n"
            f"Imagens processadas: {resumo['imagens']}\n"
            f"Resultados gerados: {resumo['resultados']}\n"
            f"Pasta: {resumo['pasta_saida']}\n"
            f"CSV: {resumo['csv']}\n"
            f"Resumo: {resumo['resumo']}"
        )






