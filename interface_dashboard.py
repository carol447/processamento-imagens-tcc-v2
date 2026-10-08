"""
Dashboard (IHM) do sistema de restauracao de imagens do TCC.

Reaproveita os modulos existentes (filtros_classicos, metodos_edp,
comparacao, experimentos) e organiza tudo em tres abas:
  1. Imagem: original x resultado atual, com historico para desfazer.
  2. Comparacao de metodos: grade visual, KPIs, tabela e graficos.
  3. Relatorio de experimentos: le as rodadas salvas em
     resultados_experimentos/ e mostra tabelas e graficos no lugar do .txt.

Execute com:  python main_dashboard.py
"""

import csv
import ctypes
import queue
import sys
import threading
import warnings
import tkinter as tk
from datetime import datetime
from pathlib import Path, PurePosixPath
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import cv2
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from comparacao import executar_comparacao
from experimentos import (
    PARAMETROS_EDP_PADRAO,
    _gerar_medias_por_metodo,
    _gerar_melhores_por_imagem,
    _gerar_ranking_por_experimento,
    executar_experimentos,
    listar_imagens,
)
from filtros_classicos import aplicar_filtro_classico
from metodos_edp import aplicar_metodo_edp

warnings.filterwarnings("ignore", message="constrained_layout not applied")


# =====================================================================
# PALETA DA INTERFACE
# =====================================================================
CREME = "#fff7bc"       # fundo
AMARELO = "#fee78a"     # destaques, status, abas nao selecionadas
LARANJA = "#f8a348"     # acao principal
VERMELHO = "#e15244"    # erros
VERDE = "#3a7b50"       # menu lateral, aba selecionada, cabecalhos

VERDE_ESCURO = "#2c5e3d"   # hover
CARD = "#fffdf0"
BORDA = "#efdf8f"
TEXTO = "#2b2a22"
TEXTO_SEC = "#6b6750"
BRANCO = "#ffffff"
ZEBRA = "#fffbe2"
VERDE_APAGADO = "#8fb89c"  # texto desativado no menu

# =====================================================================
# CORES DOS GRAFICOS (codificam a CATEGORIA do metodo)
# =====================================================================
CORES_CATEGORIA = {
    "edp": "#2f7d4f",
    "filtro_classico": "#2b62b8",
    "referencia": "#c2410c",
}
HACHURA_CATEGORIA = {"referencia": "////"}
NOMES_CATEGORIA = {
    "edp": "Método EDP",
    "filtro_classico": "Filtro clássico",
    "referencia": "Referência OpenCV",
}
ORDEM_CATEGORIAS = ("edp", "filtro_classico", "referencia")

CATEGORIA_METODO = {
    "Media": "filtro_classico",
    "Gaussiano": "filtro_classico",
    "Mediana": "filtro_classico",
    "Laplaciano": "filtro_classico",
    "Difusao": "edp",
    "Difusao isotropica": "edp",
    "Perona-Malik": "edp",
    "Navier-Stokes": "edp",
    "Navier OpenCV": "referencia",
}

NOMES_EXPERIMENTO = {
    "ruido_gaussiano": "Ruído gaussiano",
    "ruido_impulsivo": "Ruído impulsivo",
    "ruido_speckle": "Ruído speckle",
    "inpainting_regiao_removida": "Inpainting",
}

# metrica -> (chave, maior_e_melhor)
METRICAS = {
    "SSIM": ("ssim", True),
    "PSNR": ("psnr", True),
    "MSE": ("mse", False),
}

NIVEIS_DEGRADACAO = {
    "Gaussiano forte": "1",
    "Gaussiano + speckle (medio)": "2",
    "Complexo": "3",
    "Ideal para EDP": "4",
    "Inpainting (Navier-Stokes)": "5",
}

EXTENSOES_TIPOS = [("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*")]
PASTA_RESULTADOS = Path("resultados_experimentos")

FONTE = "Segoe UI"
ESCALA = 1.0   # pixels fisicos por pixel logico (1.25 numa tela com escala de 125%)


def fnt(tamanho=10, peso="normal"):
    return (FONTE, tamanho, peso)


def px(valor):
    """Converte uma medida pensada para 100% de escala em pixels reais."""
    return int(round(valor * ESCALA))


def ativar_dpi_nitido():
    """Pede ao Windows para nao ampliar a janela como bitmap (texto borrado).

    Deve ser chamada ANTES de criar o tk.Tk(). As fontes do Tk ja acompanham
    a escala; as medidas em pixels usam px().
    """
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


# =====================================================================
# FUNCOES AUXILIARES
# =====================================================================
def ler_imagem(caminho, flags=cv2.IMREAD_COLOR):
    """cv2.imread falha com acentos no caminho no Windows; imdecode nao."""
    try:
        dados = np.fromfile(str(caminho), dtype=np.uint8)
    except OSError:
        return None
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, flags)


def salvar_imagem_arquivo(caminho, img):
    ext = Path(caminho).suffix or ".png"
    ok, buffer = cv2.imencode(ext, img)
    if ok:
        buffer.tofile(str(caminho))
    return ok


def categoria_de(metodo):
    return CATEGORIA_METODO.get(metodo, "filtro_classico")


def nome_experimento(chave):
    return NOMES_EXPERIMENTO.get(chave, chave.replace("_", " ").capitalize())


def ordem_experimento(chave):
    lista = list(NOMES_EXPERIMENTO)
    return lista.index(chave) if chave in lista else len(lista)


def formatar_metrica(chave, valor):
    return f"{valor:.4f}" if chave.startswith("ssim") else f"{valor:.2f}"


def rotulo_rodada(pasta):
    try:
        return datetime.strptime(Path(pasta).name, "%Y%m%d_%H%M%S").strftime("%d/%m/%Y %H:%M:%S")
    except ValueError:
        return Path(pasta).name


# =====================================================================
# ESTILO
# =====================================================================
def configurar_estilos(root):
    global FONTE, ESCALA
    # tk scaling e relativo a 72 dpi; o Windows usa 96 dpi como 100%
    ESCALA = max(1.0, float(root.tk.call("tk", "scaling")) / (96 / 72))
    familias = set(tkfont.families(root))
    for candidata in ("Segoe UI", "Helvetica Neue", "Helvetica", "DejaVu Sans", "Arial"):
        if candidata in familias:
            FONTE = candidata
            break
    else:
        FONTE = "TkDefaultFont"

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [FONTE, "DejaVu Sans", "Arial"]

    root.option_add("*TCombobox*Listbox.font", fnt(10))
    root.option_add("*TCombobox*Listbox.selectBackground", AMARELO)
    root.option_add("*TCombobox*Listbox.selectForeground", TEXTO)

    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", font=fnt(10), background=CREME, foreground=TEXTO)
    style.configure("TFrame", background=CREME)

    style.configure("TNotebook", background=CREME, borderwidth=0, tabmargins=(0, 4, 0, 0))
    style.configure("TNotebook.Tab", background=AMARELO, foreground=TEXTO,
                    padding=(16, 7), borderwidth=0, font=fnt(10, "bold"))
    style.map("TNotebook.Tab",
              background=[("selected", VERDE), ("active", "#fbd95a")],
              foreground=[("selected", BRANCO)])

    style.configure("Sub.TNotebook", background=CARD, borderwidth=0)
    style.configure("Sub.TNotebook.Tab", background=CREME, foreground=TEXTO_SEC,
                    padding=(12, 5), font=fnt(9, "bold"))
    style.map("Sub.TNotebook.Tab",
              background=[("selected", AMARELO), ("active", "#fff0a8")],
              foreground=[("selected", TEXTO)])

    style.configure("Treeview", background=BRANCO, fieldbackground=BRANCO,
                    foreground=TEXTO, rowheight=px(26), borderwidth=0, font=fnt(9))
    style.configure("Treeview.Heading", background=VERDE, foreground=BRANCO,
                    font=fnt(9, "bold"), relief="flat", padding=(6, 6))
    style.map("Treeview.Heading", background=[("active", VERDE_ESCURO)])
    style.map("Treeview", background=[("selected", LARANJA)],
              foreground=[("selected", TEXTO)])

    style.configure("TCombobox", fieldbackground=BRANCO, background=AMARELO,
                    arrowcolor=VERDE, bordercolor=BORDA, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", BRANCO)],
              selectbackground=[("readonly", BRANCO)],
              selectforeground=[("readonly", TEXTO)])

    style.configure("Laranja.Horizontal.TProgressbar", troughcolor=CREME,
                    background=LARANJA, bordercolor=BORDA,
                    lightcolor=LARANJA, darkcolor=LARANJA)

    style.configure("TPanedwindow", background=CREME)
    style.configure("Sash", sashthickness=8, gripcount=0, background=CREME)

    style.configure("TRadiobutton", background=CARD, foreground=TEXTO, font=fnt(10))
    style.map("TRadiobutton", background=[("active", CARD)],
              foreground=[("disabled", "#b3ab88")])

    for orient in ("Vertical", "Horizontal"):
        style.configure(f"{orient}.TScrollbar", background=AMARELO, troughcolor=CARD,
                        bordercolor=CARD, arrowcolor=VERDE, lightcolor=AMARELO,
                        darkcolor=AMARELO)


# =====================================================================
# COMPONENTES
# =====================================================================
class BotaoPlano(tk.Button):
    """Botao plano com hover e aparencia propria quando desativado."""

    def __init__(self, master, texto, comando, bg, fg, hover, bg_off, fg_off, **kw):
        kw.setdefault("font", fnt(10, "bold"))
        kw.setdefault("padx", 14)
        kw.setdefault("pady", 6)
        super().__init__(master, text=texto, command=comando, bg=bg, fg=fg,
                         activebackground=hover, activeforeground=fg,
                         disabledforeground=fg_off, relief="flat", bd=0,
                         highlightthickness=0, cursor="hand2", **kw)
        self._bg, self._hover, self._bg_off = bg, hover, bg_off
        self.bind("<Enter>", self._entrar, add="+")
        self.bind("<Leave>", self._sair, add="+")

    def ativo(self):
        return str(self["state"]) != "disabled"

    def _entrar(self, _evento=None):
        if self.ativo():
            self.configure(bg=self._hover)

    def _sair(self, _evento=None):
        self.configure(bg=self._bg if self.ativo() else self._bg_off)

    def habilitar(self, ativo):
        self.configure(state="normal" if ativo else "disabled",
                       bg=self._bg if ativo else self._bg_off,
                       cursor="hand2" if ativo else "arrow")


ESTILOS_BOTAO = {
    #              bg        fg      hover      bg_off     fg_off
    "primario":   (LARANJA, TEXTO, "#f39028", "#f7d9b3", "#9a8f6c"),
    "secundario": (AMARELO, TEXTO, "#fbd95a", "#fbf0c4", "#a8a07c"),
}


def botao(master, texto, comando, tipo="primario", **kw):
    return BotaoPlano(master, texto, comando, *ESTILOS_BOTAO[tipo], **kw)


def card(master, **kw):
    return tk.Frame(master, bg=CARD, highlightbackground=BORDA,
                    highlightcolor=BORDA, highlightthickness=1, **kw)


class PainelImagem(tk.Frame):
    """Mostra uma imagem OpenCV ajustada ao espaco disponivel."""

    def __init__(self, master, titulo, texto_vazio, borda=True):
        super().__init__(master, bg=CARD, highlightbackground=BORDA,
                         highlightcolor=BORDA, highlightthickness=1 if borda else 0)
        self.titulo = tk.Label(self, text=titulo, bg=CARD, fg=VERDE,
                               font=fnt(9, "bold"), anchor="w")
        if titulo:
            self.titulo.pack(fill=tk.X, padx=12, pady=(10, 4))

        self.area = tk.Frame(self, bg=CARD)
        self.area.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.label = tk.Label(self.area, bg=CARD, fg=TEXTO_SEC, text=texto_vazio,
                              font=fnt(10), justify="center")
        self.label.place(relx=0.5, rely=0.5, anchor="center")

        self.texto_vazio = texto_vazio
        self.img = None
        self._agendado = None
        self._tamanho = None
        self.area.bind("<Configure>", self._agendar)

    def definir_titulo(self, texto):
        self.titulo.configure(text=texto)

    def ao_duplo_clique(self, callback):
        for w in (self.area, self.label):
            w.bind("<Double-Button-1>", lambda _e: self.img is not None and callback(), add="+")

    def mostrar(self, img, texto_vazio=None):
        if texto_vazio is not None:
            self.texto_vazio = texto_vazio
        self.img = img
        self._tamanho = None
        self._render()

    def _agendar(self, _evento=None):
        # debounce: so redesenha quando o redimensionamento "assenta"
        if self._agendado:
            self.after_cancel(self._agendado)
        self._agendado = self.after(90, self._render)

    def _render(self):
        self._agendado = None
        if self.img is None:
            self.label.configure(image="", text=self.texto_vazio)
            self.label.image = None
            return

        w = max(self.area.winfo_width() - 6, 40)
        h = max(self.area.winfo_height() - 6, 40)
        if self._tamanho == (w, h):
            return
        self._tamanho = (w, h)

        img = self.img
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        escala = min(w / pil.width, h / pil.height)
        novo = (max(1, int(pil.width * escala)), max(1, int(pil.height * escala)))
        pil = pil.resize(novo, Image.Resampling.LANCZOS)
        foto = ImageTk.PhotoImage(pil)
        self.label.configure(image=foto, text="")
        self.label.image = foto


class Tabela(tk.Frame):
    """Treeview com scrollbars, linhas zebradas, destaque e ordenacao por coluna.

    colunas: lista de (titulo, largura, ancora).
    """

    def __init__(self, master, colunas, altura=8):
        super().__init__(master, bg=CARD)
        ids = [f"c{i}" for i in range(len(colunas))]
        self.tree = ttk.Treeview(self, columns=ids, show="headings", height=altura)
        for cid, (titulo, largura, ancora) in zip(ids, colunas):
            self.tree.heading(cid, text=titulo, anchor=ancora,
                              command=lambda c=cid: self._ordenar(c))
            self.tree.column(cid, anchor=ancora, width=px(largura), minwidth=px(50))

        sy = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        sx = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tree.tag_configure("par", background=BRANCO)
        self.tree.tag_configure("impar", background=ZEBRA)
        self.tree.tag_configure("destaque", background=AMARELO, font=fnt(9, "bold"))
        self._reverso = {}

    def preencher(self, linhas, destacar=None):
        """linhas: lista de tuplas; destacar: funcao(linha) -> bool."""
        self.tree.delete(*self.tree.get_children())
        self._reverso = {}
        for i, linha in enumerate(linhas):
            tag = "destaque" if destacar and destacar(linha) else ("par" if i % 2 == 0 else "impar")
            self.tree.insert("", tk.END, values=linha, tags=(tag,))

    def _ordenar(self, coluna):
        itens = [(self.tree.set(k, coluna), k) for k in self.tree.get_children("")]

        def chave(par):
            texto = par[0]
            try:
                return (0, float(texto.replace(",", ".")), "")
            except ValueError:
                return (1, 0.0, texto.lower())

        reverso = self._reverso.get(coluna, False)
        itens.sort(key=chave, reverse=reverso)
        self._reverso = {coluna: not reverso}
        for i, (_valor, k) in enumerate(itens):
            self.tree.move(k, "", i)
            if "destaque" not in self.tree.item(k, "tags"):
                self.tree.item(k, tags=("par" if i % 2 == 0 else "impar",))


class Grafico(tk.Frame):
    """Figura matplotlib embutida no Tk."""

    def __init__(self, master, texto_vazio):
        super().__init__(master, bg=CARD)
        # figura pequena: o canvas a expande ate o tamanho do widget sem cortar
        self.fig = Figure(figsize=(4, 2), dpi=100, facecolor=CARD, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=CARD, highlightthickness=0)
        widget.pack(fill=tk.BOTH, expand=True)
        widget.bind("<Map>", self._sincronizar_tamanho, add="+")
        self.tem_dados = False
        self.vazio(texto_vazio)

    def _sincronizar_tamanho(self, _evento=None):
        # Em telas com escala > 100%, o backend Tk do matplotlib aumenta o DPI
        # da figura quando o widget aparece, mas mantem o tamanho antigo em
        # polegadas: a figura ficaria maior que o widget (cortada a direita).
        # Aqui a figura volta a ter exatamente o tamanho real do widget.
        widget = self.canvas.get_tk_widget()
        largura, altura = widget.winfo_width(), widget.winfo_height()
        if largura > 1 and altura > 1:
            self.fig.set_size_inches(largura / self.fig.dpi, altura / self.fig.dpi, forward=False)
        # o backend tambem aumenta o tamanho pedido do widget; volta ao minimo
        # para que o layout continue sendo decidido pelos pesos dos paineis
        widget.configure(width=px(400), height=px(200))
        self.canvas.draw_idle()

    def vazio(self, texto):
        self.tem_dados = False
        self.fig.clear()
        self.fig.text(0.5, 0.5, texto, ha="center", va="center", color=TEXTO_SEC, fontsize=10)
        self.canvas.draw_idle()

    def desenhado(self):
        self.tem_dados = True
        self.canvas.draw_idle()

    def exportar(self, parent, nome_sugerido):
        caminho = filedialog.asksaveasfilename(
            parent=parent, title="Exportar gráfico", initialfile=nome_sugerido,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not caminho:
            return None
        self.fig.savefig(caminho, dpi=200, facecolor=BRANCO)
        return caminho


def desenhar_barras(ax, nomes, valores, metrica, titulo=None):
    """Barras horizontais ordenadas (melhor no topo); cor = categoria do metodo."""
    chave, maior_melhor = METRICAS[metrica]
    pares = sorted(zip(nomes, valores), key=lambda p: p[1], reverse=maior_melhor)
    pares.reverse()  # barh desenha de baixo para cima: o melhor fica no topo
    nomes_ord = [p[0] for p in pares]
    vals = [p[1] for p in pares]
    categorias = [categoria_de(n) for n in nomes_ord]
    posicoes = list(range(len(vals)))

    barras = ax.barh(posicoes, vals, height=0.62,
                     color=[CORES_CATEGORIA[c] for c in categorias],
                     edgecolor=CARD, linewidth=2)
    for barra, cat in zip(barras, categorias):
        if cat in HACHURA_CATEGORIA:
            barra.set_hatch(HACHURA_CATEGORIA[cat])

    ax.set_yticks(posicoes)
    ax.set_yticklabels(nomes_ord, fontsize=8.5, color=TEXTO)

    maximo = max(vals) if vals else 1
    if chave == "ssim":
        ax.set_xlim(0, 1.25)
    else:
        ax.set_xlim(0, maximo * 1.32 if maximo > 0 else 1)

    deslocamento = ax.get_xlim()[1] * 0.012
    for i, v in enumerate(vals):
        melhor = i == len(vals) - 1
        ax.text(v + deslocamento, i, formatar_metrica(chave, v), va="center",
                fontsize=8, color=TEXTO, fontweight="bold" if melhor else "normal")

    ax.set_facecolor("none")
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(BORDA)
    ax.tick_params(axis="x", colors=TEXTO_SEC, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=BORDA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(titulo or metrica, fontsize=10, color=VERDE, fontweight="bold", loc="left")
    sentido = "maior é melhor" if maior_melhor else "menor é melhor"
    ax.set_xlabel(f"{metrica} ({sentido})", fontsize=8, color=TEXTO_SEC)


def legenda_categorias(fig, categorias):
    handles = [
        Patch(facecolor=CORES_CATEGORIA[c], edgecolor=CARD,
              hatch=HACHURA_CATEGORIA.get(c), label=NOMES_CATEGORIA[c])
        for c in ORDEM_CATEGORIAS if c in categorias
    ]
    fig.legend(handles=handles, loc="outside upper left", ncol=len(handles),
               frameon=False, fontsize=8.5, labelcolor=TEXTO)


# =====================================================================
# DIALOGOS
# =====================================================================
class DialogoBase(tk.Toplevel):
    """Toplevel modal, centralizado na janela principal; Esc cancela."""

    def __init__(self, root, titulo, subtitulo):
        super().__init__(root)
        self.root_app = root
        self.withdraw()
        self.title(titulo)
        self.configure(bg=CARD)
        self.transient(root)
        self.resizable(False, False)

        topo = tk.Frame(self, bg=VERDE)
        topo.pack(fill=tk.X)
        tk.Label(topo, text=titulo, bg=VERDE, fg=BRANCO,
                 font=fnt(12, "bold")).pack(anchor="w", padx=18, pady=(12, 0))
        tk.Label(topo, text=subtitulo, bg=VERDE, fg=CREME,
                 font=fnt(9)).pack(anchor="w", padx=18, pady=(0, 12))

        self.corpo = tk.Frame(self, bg=CARD)
        self.corpo.pack(fill=tk.BOTH, expand=True, padx=18, pady=10)
        self.rodape = tk.Frame(self, bg=CARD)
        self.rodape.pack(fill=tk.X, padx=18, pady=(0, 14))

        self.resultado = None
        self.protocol("WM_DELETE_WINDOW", self.cancelar)
        self.bind("<Escape>", lambda _e: self.cancelar())

    def _centralizar(self):
        self.update_idletasks()
        root = self.root_app
        x = root.winfo_rootx() + (root.winfo_width() - self.winfo_reqwidth()) // 2
        y = root.winfo_rooty() + (root.winfo_height() - self.winfo_reqheight()) // 3
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def cancelar(self):
        self.resultado = None
        self.destroy()

    def rodar(self, foco=None):
        self._centralizar()
        self.deiconify()
        self.grab_set()
        (foco or self).focus_set()
        self.root_app.wait_window(self)
        return self.resultado


class DialogoParametros(DialogoBase):
    """Substitui configurar_parametros_edp: mesmos campos e padroes."""

    # tipo -> [(chave, rotulo, conversor, deve_ser_positivo)]
    CAMPOS = {
        "1": [("isotropica_iteracoes", "Iterações", int, False),
              ("isotropica_alpha", "Alpha (passo)", float, False)],
        "2": [("perona_iteracoes", "Iterações", int, False),
              ("perona_kappa", "Kappa (limiar de borda)", float, True),
              ("perona_gamma", "Gamma (passo)", float, False)],
        "3": [("navier_iteracoes", "Iterações", int, False),
              ("navier_dt", "Passo temporal dt", float, False),
              ("navier_suavizacao", "Suavização", float, False)],
        "4": [("opencv_raio", "Raio OpenCV", int, False)],
    }
    NOMES = {"1": "Difusão isotrópica", "2": "Perona-Malik",
             "3": "Navier-Stokes", "4": "Navier OpenCV"}

    def __init__(self, root, tipos):
        super().__init__(root, "Parâmetros EDP",
                         "Ajuste os valores ou mantenha os padrões do TCC.")
        self.entradas = {}
        linha = 0
        for tipo in tipos:
            if tipo not in self.CAMPOS:
                continue
            tk.Label(self.corpo, text=self.NOMES[tipo], bg=CARD, fg=VERDE,
                     font=fnt(10, "bold")).grid(row=linha, column=0, columnspan=3,
                                                sticky="w", pady=(8, 2))
            linha += 1
            for chave, rotulo, conversor, positivo in self.CAMPOS[tipo]:
                padrao = PARAMETROS_EDP_PADRAO[chave]
                tk.Label(self.corpo, text=rotulo, bg=CARD, fg=TEXTO,
                         font=fnt(9)).grid(row=linha, column=0, sticky="w",
                                           padx=(6, 16), pady=3)
                entrada = tk.Entry(self.corpo, width=12, relief="flat", bg=BRANCO,
                                   fg=TEXTO, insertbackground=TEXTO,
                                   highlightthickness=1, highlightbackground=BORDA,
                                   highlightcolor=LARANJA, font=fnt(10))
                entrada.insert(0, str(padrao))
                entrada.grid(row=linha, column=1, sticky="w", pady=3)
                tk.Label(self.corpo, text=f"padrão: {padrao}", bg=CARD, fg=TEXTO_SEC,
                         font=fnt(8)).grid(row=linha, column=2, sticky="w", padx=(10, 0))
                self.entradas[chave] = (entrada, conversor, rotulo, positivo)
                linha += 1

        self.lbl_erro = tk.Label(self.corpo, text="", bg=CARD, fg=VERMELHO,
                                 font=fnt(9, "bold"), anchor="w", justify="left")
        self.lbl_erro.grid(row=linha, column=0, columnspan=3, sticky="w", pady=(8, 0))

        botao(self.rodape, "Usar padrões", self.usar_padroes, "secundario").pack(side=tk.LEFT)
        botao(self.rodape, "Aplicar", self.confirmar, "primario").pack(side=tk.RIGHT)
        botao(self.rodape, "Cancelar", self.cancelar, "secundario").pack(side=tk.RIGHT, padx=6)
        self.bind("<Return>", lambda _e: self.confirmar())
        self.bind("<KP_Enter>", lambda _e: self.confirmar())

    def rodar(self, foco=None):
        primeira = next(iter(self.entradas.values()))[0] if self.entradas else None
        if primeira is not None:
            primeira.select_range(0, tk.END)
        return super().rodar(primeira)

    def usar_padroes(self):
        # mesmo comportamento do dialogo original: {} -> padroes de cada metodo
        self.resultado = {}
        self.destroy()

    def confirmar(self):
        for entrada, *_ in self.entradas.values():
            entrada.configure(highlightbackground=BORDA, highlightcolor=LARANJA)

        parametros = {}
        for chave, (entrada, conversor, rotulo, positivo) in self.entradas.items():
            texto = entrada.get().strip().replace(",", ".") or str(PARAMETROS_EDP_PADRAO[chave])
            try:
                numero = float(texto)
            except ValueError:
                return self._marcar(entrada, f"Valor inválido em \"{rotulo}\": use um número.")
            if conversor is int and not numero.is_integer():
                return self._marcar(entrada, f"\"{rotulo}\" deve ser um número inteiro.")
            if numero < 0:
                return self._marcar(entrada, f"\"{rotulo}\" não pode ser negativo.")
            if positivo and numero == 0:
                return self._marcar(entrada, f"\"{rotulo}\" deve ser maior que zero.")
            parametros[chave] = conversor(numero)

        self.resultado = parametros
        self.destroy()

    def _marcar(self, entrada, mensagem):
        entrada.configure(highlightbackground=VERMELHO, highlightcolor=VERMELHO)
        entrada.focus_set()
        entrada.select_range(0, tk.END)
        self.lbl_erro.configure(text=mensagem)
        self.bell()


class DialogoOrigem(DialogoBase):
    """Escolha das imagens do experimento (substitui o simpledialog 1-3)."""

    def __init__(self, root, nome_imagem=None):
        super().__init__(root, "Novo experimento", "Escolha quais imagens entram na rodada.")
        self.opcao = tk.StringVar(value="pasta_img")
        n_img = len(listar_imagens("img"))
        texto_carregada = (f"Apenas a imagem carregada ({nome_imagem})" if nome_imagem
                           else "Apenas a imagem carregada (nenhuma imagem carregada)")
        opcoes = [
            ("pasta_img", f"Todas as imagens da pasta img ({n_img} arquivo(s))", True),
            ("carregada", texto_carregada, nome_imagem is not None),
            ("escolher", "Escolher outra pasta...", True),
        ]
        for valor, texto, habilitada in opcoes:
            rb = ttk.Radiobutton(self.corpo, text=texto, value=valor, variable=self.opcao)
            rb.pack(anchor="w", pady=3)
            if not habilitada:
                rb.state(["disabled"])

        tk.Label(self.corpo,
                 text="Cada imagem passa por 4 degradações e por todos os métodos.\n"
                      "Com Navier-Stokes a rodada pode levar alguns minutos;\n"
                      "a janela continua respondendo durante o processamento.",
                 bg=CARD, fg=TEXTO_SEC, font=fnt(9), justify="left").pack(anchor="w", pady=(10, 0))

        botao(self.rodape, "Continuar", self.confirmar, "primario").pack(side=tk.RIGHT)
        botao(self.rodape, "Cancelar", self.cancelar, "secundario").pack(side=tk.RIGHT, padx=6)
        self.bind("<Return>", lambda _e: self.confirmar())

    def confirmar(self):
        self.resultado = self.opcao.get()
        self.destroy()


# =====================================================================
# APLICACAO
# =====================================================================
class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("Restauração de Imagens por EDP - Dashboard")
        configurar_estilos(root)   # define FONTE e ESCALA, usados em todo o layout
        largura = min(px(1360), root.winfo_screenwidth() - 40)
        altura = min(px(860), root.winfo_screenheight() - 90)
        x = max((root.winfo_screenwidth() - largura) // 2, 0)
        y = max((root.winfo_screenheight() - altura) // 2 - 20, 0)
        root.geometry(f"{largura}x{altura}+{x}+{y}")
        root.minsize(min(px(1100), largura), min(px(660), altura))
        root.configure(bg=CREME)

        self.imagem_original = None
        self.imagem_atual = None
        self.caminho_imagem = None
        self.historico = []            # [(nome_etapa, imagem_antes_da_etapa)]
        self.ocupado = False
        self.acoes = []                # [(BotaoPlano, requisito() -> bool | None)]

        self.rodadas = {}              # rotulo -> Path
        self.rel_pasta = None
        self.rel_medias = []
        self.rel_ranking = []
        self.mapa_exp = {}

        self._montar_barra_status()
        self._montar_menu_lateral()
        self._montar_area_principal()
        self._atalhos()
        self._atualizar_estado_botoes()
        self.status("Pronto. Ctrl+O abre uma imagem; Ctrl+Z desfaz a última etapa.")
        self.root.after(150, self._carregar_lista_rodadas)

    # -----------------------------------------------------------------
    # ESTRUTURA
    # -----------------------------------------------------------------
    def _montar_barra_status(self):
        barra = tk.Frame(self.root, bg=AMARELO, height=30)
        barra.pack(side=tk.BOTTOM, fill=tk.X)
        self.ponto_status = tk.Canvas(barra, width=px(10), height=px(10), bg=AMARELO,
                                      highlightthickness=0)
        self.ponto_status.pack(side=tk.LEFT, padx=(14, 8), pady=8)
        self._oval = self.ponto_status.create_oval(1, 1, px(10) - 1, px(10) - 1, fill=VERDE, outline="")
        self.lbl_status = tk.Label(barra, text="", bg=AMARELO, fg=TEXTO,
                                   font=fnt(9), anchor="w")
        self.lbl_status.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=5)
        self.progresso = ttk.Progressbar(barra, mode="indeterminate", length=px(200),
                                         style="Laranja.Horizontal.TProgressbar")

    def _montar_menu_lateral(self):
        menu = tk.Frame(self.root, bg=VERDE, width=px(230))
        menu.pack(side=tk.LEFT, fill=tk.Y)
        menu.pack_propagate(False)

        marca = tk.Frame(menu, bg=VERDE)
        marca.pack(fill=tk.X, padx=18, pady=(20, 8))
        tk.Label(marca, text="EDP Lab", bg=VERDE, fg=AMARELO,
                 font=fnt(15, "bold")).pack(anchor="w")
        tk.Label(marca, text="Restauração de imagens", bg=VERDE, fg=CREME,
                 font=fnt(9)).pack(anchor="w")

        tem_imagem = lambda: self.imagem_original is not None  # noqa: E731
        tem_historico = lambda: bool(self.historico)           # noqa: E731

        def secao(texto):
            tk.Label(menu, text=texto, bg=VERDE, fg=VERDE_APAGADO, font=fnt(8, "bold"),
                     anchor="w").pack(fill=tk.X, padx=18, pady=(14, 3))

        def item(texto, comando, requisito=tem_imagem, atalho=""):
            linha = tk.Frame(menu, bg=VERDE)
            linha.pack(fill=tk.X)
            b = BotaoPlano(linha, texto, comando, VERDE, BRANCO, VERDE_ESCURO,
                           VERDE, VERDE_APAGADO, anchor="w", font=fnt(10),
                           padx=18, pady=5)
            b.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if atalho:
                dica = tk.Label(linha, text=atalho, bg=VERDE, fg=VERDE_APAGADO, font=fnt(8))
                dica.pack(side=tk.RIGHT, padx=(0, 14))
                # o rotulo do atalho acompanha o hover do botao
                b.bind("<Enter>", lambda _e: b.ativo() and dica.configure(bg=VERDE_ESCURO), add="+")
                b.bind("<Leave>", lambda _e: dica.configure(bg=VERDE), add="+")
            self.acoes.append((b, requisito))
            return b

        secao("IMAGEM")
        item("Carregar", self.carregar_imagem, None, "Ctrl+O")
        item("Salvar", self.salvar_imagem, tem_imagem, "Ctrl+S")
        item("Desfazer", self.desfazer, tem_historico, "Ctrl+Z")
        item("Restaurar original", self.resetar_imagem, tem_historico)

        secao("FILTROS CLÁSSICOS")
        item("Média", lambda: self.aplicar_filtro("1", "Média"))
        item("Gaussiano", lambda: self.aplicar_filtro("2", "Gaussiano"))
        item("Mediana", lambda: self.aplicar_filtro("3", "Mediana"))
        item("Laplaciano", lambda: self.aplicar_filtro("4", "Laplaciano"))

        secao("MÉTODOS EDP")
        item("Difusão isotrópica", lambda: self.aplicar_edp("1", "Difusão isotrópica"))
        item("Perona-Malik", lambda: self.aplicar_edp("2", "Perona-Malik"))
        item("Navier-Stokes (inpainting)", lambda: self.aplicar_edp("3", "Navier-Stokes"))

        secao("ANÁLISE")
        item("Comparar métodos", self.ir_para_comparacao)
        item("Novo experimento", self.executar_experimentos_tcc, None)

    def _montar_area_principal(self):
        principal = tk.Frame(self.root, bg=CREME)
        principal.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        topo = tk.Frame(principal, bg=CREME)
        topo.pack(fill=tk.X, padx=22, pady=(16, 6))
        tk.Label(topo, text="Restauração de imagens: filtros clássicos x EDP",
                 bg=CREME, fg=VERDE_ESCURO, font=fnt(16, "bold")).pack(side=tk.LEFT)
        self.chip_imagem = tk.Label(topo, text="Nenhuma imagem carregada", bg=AMARELO,
                                    fg=TEXTO, font=fnt(9, "bold"), padx=12, pady=4)
        self.chip_imagem.pack(side=tk.RIGHT)

        self.abas = ttk.Notebook(principal)
        self.abas.pack(fill=tk.BOTH, expand=True, padx=22, pady=(4, 12))
        self.aba_img = tk.Frame(self.abas, bg=CREME)
        self.aba_comp = tk.Frame(self.abas, bg=CREME)
        self.aba_rel = tk.Frame(self.abas, bg=CREME)
        self.abas.add(self.aba_img, text="Imagem")
        self.abas.add(self.aba_comp, text="Comparação de métodos")
        self.abas.add(self.aba_rel, text="Relatório de experimentos")

        self._montar_aba_imagem()
        self._montar_aba_comparacao()
        self._montar_aba_relatorio()

    # ---------------- Aba 1: Imagem ----------------
    def _montar_aba_imagem(self):
        corpo = tk.Frame(self.aba_img, bg=CREME)
        corpo.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        corpo.grid_columnconfigure(0, weight=1, uniform="lado")
        corpo.grid_columnconfigure(1, weight=1, uniform="lado")
        corpo.grid_rowconfigure(0, weight=1)

        self.painel_original = PainelImagem(
            corpo, "ORIGINAL", "Carregue uma imagem pelo menu\n\"Carregar\" (Ctrl+O).")
        self.painel_original.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.painel_atual = PainelImagem(
            corpo, "RESULTADO ATUAL", "Aplique um filtro ou um método EDP\npara ver o resultado aqui.")
        self.painel_atual.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.painel_original.ao_duplo_clique(
            lambda: self._ampliar(self.imagem_original, "Original"))
        self.painel_atual.ao_duplo_clique(
            lambda: self._ampliar(self.imagem_atual, "Resultado atual · " + (
                self.historico[-1][0] if self.historico else "Original")))

        trilha = card(self.aba_img)
        trilha.pack(fill=tk.X, pady=(10, 0))
        tk.Label(trilha, text="Etapas aplicadas:", bg=CARD, fg=VERDE,
                 font=fnt(10, "bold")).pack(side=tk.LEFT, padx=(12, 8), pady=8)
        self.lbl_etapas = tk.Label(trilha, text="nenhuma imagem carregada", bg=CARD,
                                   fg=TEXTO, font=fnt(10), anchor="w")
        self.lbl_etapas.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8)

    # ---------------- Aba 2: Comparacao ----------------
    def _montar_aba_comparacao(self):
        barra = card(self.aba_comp)
        barra.pack(fill=tk.X, pady=(12, 10))
        tk.Label(barra, text="Degradação", bg=CARD, fg=TEXTO_SEC,
                 font=fnt(9)).pack(side=tk.LEFT, padx=(12, 6), pady=10)
        self.cb_nivel = ttk.Combobox(barra, values=list(NIVEIS_DEGRADACAO),
                                     state="readonly", width=30)
        self.cb_nivel.current(0)
        self.cb_nivel.pack(side=tk.LEFT, pady=10)
        self.btn_comparar = botao(barra, "Executar comparação", self.comparar_metodos)
        self.btn_comparar.pack(side=tk.LEFT, padx=10)
        self.acoes.append((self.btn_comparar, lambda: self.imagem_original is not None))
        self.btn_exp_comp = botao(barra, "Exportar gráfico",
                                  lambda: self._exportar(self.graf_comp, "comparacao_metodos"),
                                  "secundario")
        self.btn_exp_comp.pack(side=tk.RIGHT, padx=12)
        self.acoes.append((self.btn_exp_comp, lambda: self.graf_comp.tem_dados))

        paned = ttk.PanedWindow(self.aba_comp, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        topo = tk.Frame(paned, bg=CREME)
        topo.grid_columnconfigure(0, weight=5, uniform="t")
        topo.grid_columnconfigure(1, weight=4, uniform="t")
        topo.grid_rowconfigure(0, weight=1)

        self.painel_grid = PainelImagem(
            topo, "GRADE VISUAL",
            "Escolha a degradação e clique em\n\"Executar comparação\".")
        self.painel_grid.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.painel_grid.ao_duplo_clique(
            lambda: self._ampliar(self.painel_grid.img, f"Grade visual · {self.rotulo_comparacao}"))

        direita = tk.Frame(topo, bg=CREME)
        direita.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        kpis = tk.Frame(direita, bg=CREME)
        kpis.pack(fill=tk.X)
        self.kpis_comp = {}
        for i, (rotulo, metrica) in enumerate([("MAIOR SSIM", "SSIM"),
                                               ("MAIOR PSNR", "PSNR"),
                                               ("MENOR MSE", "MSE")]):
            kpis.grid_columnconfigure(i, weight=1, uniform="k")
            c = card(kpis)
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            tk.Label(c, text=rotulo, bg=CARD, fg=TEXTO_SEC,
                     font=fnt(8, "bold")).pack(anchor="w", padx=10, pady=(8, 0))
            valor = tk.Label(c, text="-", bg=CARD, fg=TEXTO, font=fnt(16, "bold"))
            valor.pack(anchor="w", padx=10)
            metodo = tk.Label(c, text=" ", bg=CARD, fg=TEXTO, font=fnt(9, "bold"))
            metodo.pack(anchor="w", padx=10)
            categoria = tk.Label(c, text=" ", bg=CARD, fg=TEXTO_SEC, font=fnt(8))
            categoria.pack(anchor="w", padx=10, pady=(0, 8))
            self.kpis_comp[metrica] = (valor, metodo, categoria)

        caixa = card(direita)
        caixa.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        tk.Label(caixa, text="MÉTRICAS POR MÉTODO", bg=CARD, fg=VERDE,
                 font=fnt(9, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.tab_comp = Tabela(caixa, [("Método", 105, "w"), ("Categoria", 105, "w"),
                                       ("MSE", 64, "e"), ("PSNR", 60, "e"),
                                       ("SSIM", 64, "e")], altura=5)
        self.tab_comp.pack(fill=tk.BOTH, expand=True, padx=12)
        tk.Label(caixa,
                 text="Destaque: maior SSIM. Clique no cabeçalho para ordenar.\n"
                      "MSE: erro médio (menor é melhor)  ·  SSIM: 0 a 1 (ideal 1)\n"
                      "PSNR: qualidade em dB (>30 bom, >40 excelente)",
                 bg=CARD, fg=TEXTO_SEC, font=fnt(8), justify="left"
                 ).pack(anchor="w", padx=12, pady=(4, 8))
        self.rotulo_comparacao = ""

        graf_card = card(paned)
        self.graf_comp = Grafico(graf_card, "Os gráficos aparecem aqui depois da comparação.")
        self.graf_comp.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        paned.add(topo, weight=3)
        paned.add(graf_card, weight=2)

    # ---------------- Aba 3: Relatorio ----------------
    def _montar_aba_relatorio(self):
        barra = card(self.aba_rel)
        barra.pack(fill=tk.X, pady=(12, 8))
        tk.Label(barra, text="Rodada", bg=CARD, fg=TEXTO_SEC,
                 font=fnt(9)).pack(side=tk.LEFT, padx=(12, 6), pady=10)
        self.cb_rodada = ttk.Combobox(barra, state="readonly", width=21)
        self.cb_rodada.pack(side=tk.LEFT)
        self.cb_rodada.bind("<<ComboboxSelected>>", self._rodada_escolhida)
        botao(barra, "Abrir pasta...", self.abrir_pasta_rodada,
              "secundario").pack(side=tk.LEFT, padx=6)

        tk.Label(barra, text="Métrica do gráfico", bg=CARD, fg=TEXTO_SEC,
                 font=fnt(9)).pack(side=tk.LEFT, padx=(16, 6))
        self.cb_metrica = ttk.Combobox(barra, values=list(METRICAS), state="readonly", width=6)
        self.cb_metrica.current(0)
        self.cb_metrica.pack(side=tk.LEFT)
        self.cb_metrica.bind("<<ComboboxSelected>>", lambda _e: self._desenhar_grafico_relatorio())

        self.btn_experimento = botao(barra, "Novo experimento", self.executar_experimentos_tcc)
        self.btn_experimento.pack(side=tk.RIGHT, padx=12)
        self.acoes.append((self.btn_experimento, None))
        self.btn_exp_rel = botao(barra, "Exportar gráfico",
                                 lambda: self._exportar(self.graf_rel, self._nome_export_relatorio()),
                                 "secundario")
        self.btn_exp_rel.pack(side=tk.RIGHT)
        self.acoes.append((self.btn_exp_rel, lambda: self.graf_rel.tem_dados))

        self.lbl_resumo_rel = tk.Label(self.aba_rel, text="Nenhuma rodada carregada.",
                                       bg=CREME, fg=TEXTO_SEC, font=fnt(9), anchor="w")
        self.lbl_resumo_rel.pack(fill=tk.X, pady=(0, 6))

        self.frame_vencedores = tk.Frame(self.aba_rel, bg=CREME)
        self.frame_vencedores.pack(fill=tk.X)

        paned = ttk.PanedWindow(self.aba_rel, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        graf_card = card(paned)
        self.graf_rel = Grafico(graf_card, "Carregando rodadas...")
        self.graf_rel.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        tabelas_card = card(paned)
        self.sub = ttk.Notebook(tabelas_card, style="Sub.TNotebook")
        self.sub.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab_ranking = Tabela(self.sub, [
            ("Experimento", 150, "w"), ("Posição", 70, "center"), ("Método", 140, "w"),
            ("Categoria", 130, "w"), ("MSE médio", 90, "e"), ("PSNR médio", 90, "e"),
            ("SSIM médio", 90, "e")], altura=5)
        self.tab_medias = Tabela(self.sub, [
            ("Experimento", 150, "w"), ("Método", 140, "w"), ("Categoria", 130, "w"),
            ("Qtd", 50, "center"), ("MSE médio", 90, "e"), ("PSNR médio", 90, "e"),
            ("SSIM médio", 90, "e")], altura=5)
        self.tab_melhores = Tabela(self.sub, [
            ("Imagem", 190, "w"), ("Experimento", 130, "w"),
            ("Melhor SSIM", 130, "w"), ("SSIM", 70, "e"),
            ("Maior PSNR", 130, "w"), ("PSNR", 70, "e"),
            ("Menor MSE", 130, "w"), ("MSE", 70, "e")], altura=5)
        self.tab_metricas = Tabela(self.sub, [
            ("Imagem", 190, "w"), ("Experimento", 130, "w"), ("Método", 130, "w"),
            ("Categoria", 120, "w"), ("Parâmetros", 240, "w"),
            ("MSE", 70, "e"), ("PSNR", 70, "e"), ("SSIM", 70, "e")], altura=5)

        grade = tk.Frame(self.sub, bg=CARD)
        filtros = tk.Frame(grade, bg=CARD)
        filtros.pack(fill=tk.X, pady=(6, 0))
        tk.Label(filtros, text="Imagem", bg=CARD, fg=TEXTO_SEC,
                 font=fnt(9)).pack(side=tk.LEFT, padx=(6, 4))
        self.cb_grade_img = ttk.Combobox(filtros, state="readonly", width=32)
        self.cb_grade_img.pack(side=tk.LEFT)
        tk.Label(filtros, text="Experimento", bg=CARD, fg=TEXTO_SEC,
                 font=fnt(9)).pack(side=tk.LEFT, padx=(14, 4))
        self.cb_grade_exp = ttk.Combobox(filtros, state="readonly", width=20)
        self.cb_grade_exp.pack(side=tk.LEFT)
        for cb in (self.cb_grade_img, self.cb_grade_exp):
            cb.bind("<<ComboboxSelected>>", lambda _e: self._mostrar_grade_rodada())
        self.btn_ampliar_grade = botao(filtros, "Abrir em janela", self._ampliar_grade_rodada,
                                       "secundario", font=fnt(9, "bold"), pady=3)
        self.btn_ampliar_grade.pack(side=tk.LEFT, padx=(14, 0))
        self.acoes.append((self.btn_ampliar_grade, lambda: self.painel_grade_rel.img is not None))
        tk.Label(filtros, text="(ou duplo clique na grade)", bg=CARD, fg=TEXTO_SEC,
                 font=fnt(8)).pack(side=tk.LEFT, padx=(6, 0))
        self.painel_grade_rel = PainelImagem(grade, "", "Sem grade para exibir.", borda=False)
        self.painel_grade_rel.pack(fill=tk.BOTH, expand=True)
        self.painel_grade_rel.ao_duplo_clique(self._ampliar_grade_rodada)
        self.aba_grade = grade

        param = tk.Frame(self.sub, bg=BRANCO)
        self.txt_param = tk.Text(param, bg=BRANCO, fg=TEXTO, font=("Consolas", 10),
                                 relief="flat", padx=12, pady=10, height=8, wrap="word")
        sy = ttk.Scrollbar(param, orient="vertical", command=self.txt_param.yview)
        self.txt_param.configure(yscrollcommand=sy.set, state="disabled")
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_param.pack(fill=tk.BOTH, expand=True)

        self.sub.add(self.tab_ranking, text="Ranking")
        self.sub.add(self.tab_medias, text="Médias por método")
        self.sub.add(self.tab_melhores, text="Melhores por imagem")
        self.sub.add(self.tab_metricas, text="Métricas completas")
        self.sub.add(grade, text="Grade visual")
        self.sub.add(param, text="Parâmetros")

        paned.add(graf_card, weight=4)
        paned.add(tabelas_card, weight=2)
        self.paned_rel = paned
        self.graf_rel_card = graf_card
        self._sash_antes_da_grade = None
        self.sub.bind("<<NotebookTabChanged>>", self._sub_aba_trocada)

    # -----------------------------------------------------------------
    # INFRAESTRUTURA: atalhos, status, estado dos botoes, segundo plano
    # -----------------------------------------------------------------
    def _atalhos(self):
        for teclas, acao in ((("<Control-o>", "<Control-O>"), self.carregar_imagem),
                             (("<Control-s>", "<Control-S>"), self.salvar_imagem),
                             (("<Control-z>", "<Control-Z>"), self.desfazer)):
            for tecla in teclas:
                self.root.bind(tecla, lambda _e, a=acao: (a(), "break")[1])

    def status(self, texto, tipo="info"):
        cor = {"info": TEXTO_SEC, "ok": VERDE, "erro": VERMELHO}[tipo]
        self.ponto_status.itemconfigure(self._oval, fill=cor)
        self.lbl_status.configure(text=texto, fg=VERMELHO if tipo == "erro" else TEXTO)

    def _atualizar_estado_botoes(self):
        for b, requisito in self.acoes:
            b.habilitar(not self.ocupado and (requisito is None or requisito()))
        self.cb_nivel.configure(state="disabled" if self.ocupado else "readonly")

    def _set_ocupado(self, ocupado, mensagem=""):
        self.ocupado = ocupado
        if ocupado:
            self.status(mensagem)
            self.progresso.pack(side=tk.RIGHT, padx=14)
            self.progresso.start(12)
            self.root.configure(cursor="watch")
        else:
            self.progresso.stop()
            self.progresso.pack_forget()
            self.root.configure(cursor="")
        self._atualizar_estado_botoes()

    def _em_segundo_plano(self, mensagem, tarefa, ao_concluir):
        """Roda a tarefa em uma thread; o resultado volta por uma fila
        verificada com root.after, entao a janela nunca trava."""
        self._set_ocupado(True, mensagem)
        fila = queue.Queue()
        inicio = datetime.now()

        def trabalhador():
            try:
                fila.put(("ok", tarefa()))
            except Exception as erro:  # noqa: BLE001 - mostrado ao usuario
                fila.put(("erro", erro))

        threading.Thread(target=trabalhador, daemon=True).start()

        def verificar():
            try:
                estado, valor = fila.get_nowait()
            except queue.Empty:
                segundos = int((datetime.now() - inicio).total_seconds())
                self.lbl_status.configure(text=f"{mensagem}  ({segundos} s)")
                self.root.after(150, verificar)
                return
            self._set_ocupado(False)
            if estado == "erro":
                self.status(f"Falha: {valor}", "erro")
                messagebox.showerror("Erro", str(valor), parent=self.root)
            else:
                ao_concluir(valor)

        self.root.after(150, verificar)

    def _exportar(self, grafico, nome_sugerido):
        if not grafico.tem_dados:
            return
        try:
            caminho = grafico.exportar(self.root, nome_sugerido)
        except Exception as erro:  # noqa: BLE001
            self.status(f"Não foi possível exportar o gráfico: {erro}", "erro")
            return
        if caminho:
            self.status(f"Gráfico exportado: {caminho}", "ok")

    # -----------------------------------------------------------------
    # ABA IMAGEM
    # -----------------------------------------------------------------
    def carregar_imagem(self):
        if self.ocupado:
            return
        caminho = filedialog.askopenfilename(parent=self.root, title="Carregar imagem",
                                             initialdir="img", filetypes=EXTENSOES_TIPOS)
        if caminho:
            self.abrir_imagem(caminho)

    def abrir_imagem(self, caminho):
        img = ler_imagem(caminho)
        if img is None:
            self.status(f"Não foi possível abrir \"{Path(caminho).name}\" como imagem.", "erro")
            return False
        self.imagem_original = img
        self.imagem_atual = img.copy()
        self.caminho_imagem = caminho
        self.historico = []
        h, w = img.shape[:2]
        self.chip_imagem.configure(text=f"{Path(caminho).name}  ·  {w} x {h} px")
        self.painel_original.mostrar(img)
        self._atualizar_resultado()
        self.abas.select(self.aba_img)
        self.status(f"Imagem carregada: {Path(caminho).name}", "ok")
        return True

    def salvar_imagem(self):
        if self.imagem_atual is None or self.ocupado:
            return
        sugestao = Path(self.caminho_imagem).stem + "_resultado.png" if self.caminho_imagem else "resultado.png"
        caminho = filedialog.asksaveasfilename(
            parent=self.root, title="Salvar resultado", initialfile=sugestao,
            defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if not caminho:
            return
        if salvar_imagem_arquivo(caminho, self.imagem_atual):
            self.status(f"Resultado salvo em {caminho}", "ok")
        else:
            self.status("Não foi possível salvar a imagem nesse formato.", "erro")

    def _registrar_etapa(self, nome, nova):
        self.historico.append((nome, self.imagem_atual))
        self.imagem_atual = nova
        self._atualizar_resultado()

    def _atualizar_resultado(self):
        self.painel_atual.mostrar(self.imagem_atual)
        if self.historico:
            etapas = " → ".join(nome for nome, _ in self.historico)
            self.lbl_etapas.configure(text=f"Original → {etapas}")
            self.painel_atual.definir_titulo(f"RESULTADO ATUAL · {self.historico[-1][0]}")
        else:
            self.lbl_etapas.configure(text="Original (nenhuma etapa aplicada)")
            self.painel_atual.definir_titulo("RESULTADO ATUAL · Original")
        self._atualizar_estado_botoes()

    def desfazer(self):
        if not self.historico or self.ocupado:
            return
        nome, anterior = self.historico.pop()
        self.imagem_atual = anterior
        self._atualizar_resultado()
        self.abas.select(self.aba_img)
        self.status(f"Etapa desfeita: {nome}")

    def resetar_imagem(self):
        if self.imagem_original is None or self.ocupado:
            return
        self.imagem_atual = self.imagem_original.copy()
        self.historico = []
        self._atualizar_resultado()
        self.abas.select(self.aba_img)
        self.status("Imagem restaurada para a original.")

    def aplicar_filtro(self, tipo, nome):
        if self.imagem_atual is None or self.ocupado:
            return
        self._registrar_etapa(nome, aplicar_filtro_classico(self.imagem_atual, tipo))
        self.abas.select(self.aba_img)
        self.status(f"Filtro aplicado: {nome}", "ok")

    def aplicar_edp(self, tipo, nome):
        if self.imagem_atual is None or self.ocupado:
            return
        mask = None
        if tipo == "3":
            caminho = filedialog.askopenfilename(parent=self.root,
                                                 title="Selecione a máscara do inpainting",
                                                 filetypes=EXTENSOES_TIPOS)
            if not caminho:
                return
            mask = ler_imagem(caminho, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                self.status("Não foi possível carregar a máscara.", "erro")
                return
        parametros = DialogoParametros(self.root, [tipo]).rodar()
        if parametros is None:
            self.status("Operação cancelada.")
            return

        entrada = self.imagem_atual.copy()
        self.abas.select(self.aba_img)

        def concluir(resultado):
            self._registrar_etapa(nome, resultado)
            self.status(f"Método aplicado: {nome}", "ok")

        self._em_segundo_plano(f"Aplicando {nome}...",
                               lambda: aplicar_metodo_edp(entrada, tipo, mask, parametros),
                               concluir)

    # -----------------------------------------------------------------
    # ABA COMPARACAO
    # -----------------------------------------------------------------
    def ir_para_comparacao(self):
        self.abas.select(self.aba_comp)
        self.cb_nivel.focus_set()
        self.status("Escolha a degradação e clique em \"Executar comparação\".")

    def comparar_metodos(self):
        if self.imagem_original is None or self.ocupado:
            return
        rotulo = self.cb_nivel.get()
        nivel = NIVEIS_DEGRADACAO[rotulo]
        if nivel == "5":
            metodos = ["Media", "Mediana", "Navier-Stokes", "Navier OpenCV"]
            parametros = DialogoParametros(self.root, ["3", "4"]).rodar()
        else:
            metodos = ["Media", "Gaussiano", "Mediana", "Difusao", "Perona-Malik"]
            parametros = DialogoParametros(self.root, ["1", "2"]).rodar()
        if parametros is None:
            self.status("Comparação cancelada.")
            return

        img = self.imagem_original.copy()
        self._em_segundo_plano(
            f"Comparando {len(metodos)} métodos ({rotulo})...",
            lambda: executar_comparacao(img, metodos, nivel, parametros),
            lambda resultado: self._mostrar_comparacao(resultado, rotulo))

    def _mostrar_comparacao(self, resultado, rotulo):
        grid, dados = resultado
        self.rotulo_comparacao = rotulo
        self.painel_grid.mostrar(grid)
        self.painel_grid.definir_titulo(f"GRADE VISUAL · {rotulo}")

        linhas = [{"metodo": nome, "mse": float(m), "psnr": float(p), "ssim": float(s)}
                  for nome, m, p, s in dados]
        melhor = max(linhas, key=lambda d: d["ssim"])["metodo"]
        self.tab_comp.preencher(
            [(d["metodo"], NOMES_CATEGORIA[categoria_de(d["metodo"])],
              f"{d['mse']:.2f}", f"{d['psnr']:.2f}", f"{d['ssim']:.4f}") for d in linhas],
            destacar=lambda linha: linha[0] == melhor)

        for metrica, (lbl_valor, lbl_metodo, lbl_categoria) in self.kpis_comp.items():
            chave, maior_melhor = METRICAS[metrica]
            d = (max if maior_melhor else min)(linhas, key=lambda x: x[chave])
            cat = categoria_de(d["metodo"])
            lbl_valor.configure(text=formatar_metrica(chave, d[chave]))
            lbl_metodo.configure(text=d["metodo"])
            lbl_categoria.configure(text=NOMES_CATEGORIA[cat], fg=CORES_CATEGORIA[cat])

        fig = self.graf_comp.fig
        fig.clear()
        for ax, metrica in zip(fig.subplots(1, 3), ("SSIM", "PSNR", "MSE")):
            chave = METRICAS[metrica][0]
            desenhar_barras(ax, [d["metodo"] for d in linhas], [d[chave] for d in linhas], metrica)
        legenda_categorias(fig, {categoria_de(d["metodo"]) for d in linhas})
        self.graf_comp.desenhado()
        self._atualizar_estado_botoes()
        self.status(f"Comparação concluída ({rotulo}). Maior SSIM: {melhor}.", "ok")

    # -----------------------------------------------------------------
    # EXPERIMENTOS
    # -----------------------------------------------------------------
    def executar_experimentos_tcc(self):
        if self.ocupado:
            return
        nome_carregada = Path(self.caminho_imagem).name if self.caminho_imagem else None
        if self.imagem_original is not None and nome_carregada is None:
            nome_carregada = "imagem_carregada.png"
        origem = DialogoOrigem(self.root, nome_carregada).rodar()
        if origem is None:
            return

        pasta_img, descricao, imagens_entrada = "img", "pasta img", None
        if origem == "carregada":
            imagens_entrada = [(nome_carregada, self.imagem_original.copy())]
            descricao = f"imagem carregada ({nome_carregada})"
        elif origem == "escolher":
            pasta = filedialog.askdirectory(parent=self.root, title="Escolha a pasta de imagens")
            if not pasta:
                return
            pasta_img, descricao = pasta, f"pasta escolhida ({pasta})"

        parametros = DialogoParametros(self.root, ["1", "2", "3", "4"]).rodar()
        if parametros is None:
            self.status("Experimento cancelado.")
            return

        def tarefa():
            entradas = imagens_entrada
            if entradas is None:
                # le aqui (e nao via cv2.imread) para aceitar acentos no caminho
                entradas = [(c.name, ler_imagem(c)) for c in listar_imagens(pasta_img)]
            return executar_experimentos(pasta_img=pasta_img, parametros_edp=parametros,
                                         imagens_entrada=entradas, descricao_imagens=descricao)

        def concluir(resumo):
            self._carregar_lista_rodadas(selecionar=Path(resumo["pasta_saida"]))
            self.abas.select(self.aba_rel)
            self.status(f"Experimento concluído: {resumo['imagens']} imagem(ns), "
                        f"{resumo['resultados']} resultados em {resumo['pasta_saida']}.", "ok")

        self._em_segundo_plano("Executando experimentos (pode levar alguns minutos)...",
                               tarefa, concluir)

    # -----------------------------------------------------------------
    # ABA RELATORIO
    # -----------------------------------------------------------------
    def _carregar_lista_rodadas(self, selecionar=None):
        pastas = []
        if PASTA_RESULTADOS.is_dir():
            pastas = sorted((p for p in PASTA_RESULTADOS.iterdir()
                             if p.is_dir() and (p / "metricas.csv").is_file()),
                            key=lambda p: p.name, reverse=True)
        self.rodadas = {rotulo_rodada(p): p for p in pastas}
        self.cb_rodada.configure(values=list(self.rodadas))
        if not pastas:
            self.graf_rel.vazio("Nenhuma rodada encontrada. Clique em \"Novo experimento\".")
            self._atualizar_estado_botoes()
            return
        alvo = rotulo_rodada(selecionar) if selecionar is not None else rotulo_rodada(pastas[0])
        if alvo not in self.rodadas:
            alvo = rotulo_rodada(pastas[0])
        self.cb_rodada.set(alvo)
        self.carregar_rodada(self.rodadas[alvo])

    def _rodada_escolhida(self, _evento=None):
        pasta = self.rodadas.get(self.cb_rodada.get())
        if pasta is not None:
            self.carregar_rodada(pasta)

    def abrir_pasta_rodada(self):
        pasta = filedialog.askdirectory(parent=self.root, initialdir=str(PASTA_RESULTADOS),
                                        title="Pasta de uma rodada (com metricas.csv)")
        if not pasta:
            return
        pasta = Path(pasta)
        if not (pasta / "metricas.csv").is_file():
            self.status(f"A pasta \"{pasta.name}\" não contém metricas.csv.", "erro")
            return
        rotulo = rotulo_rodada(pasta)
        self.rodadas[rotulo] = pasta
        self.cb_rodada.configure(values=list(self.rodadas))
        self.cb_rodada.set(rotulo)
        self.carregar_rodada(pasta)

    @staticmethod
    def _ler_metricas(pasta):
        linhas = []
        with open(pasta / "metricas.csv", encoding="utf-8", newline="") as arquivo:
            for row in csv.DictReader(arquivo):
                # A coluna "imagem" das rodadas antigas esta errada; o nome real
                # vem do caminho: .../<rodada>/<imagem>/<experimento>/<metodo>.png
                partes = PurePosixPath(row["arquivo_resultado"].replace("\\", "/")).parts
                imagem = partes[-3] if len(partes) >= 3 else row["imagem"]
                linhas.append({
                    "imagem": imagem,
                    "experimento": row["experimento"],
                    "metodo": row["metodo"],
                    "categoria": row.get("categoria") or categoria_de(row["metodo"]),
                    "parametros": row.get("parametros", ""),
                    "mse": float(row["mse"]),
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                })
        return linhas

    def carregar_rodada(self, pasta):
        pasta = Path(pasta)
        try:
            linhas = self._ler_metricas(pasta)
        except (OSError, KeyError, ValueError) as erro:
            self.status(f"Não foi possível ler a rodada {pasta.name}: {erro}", "erro")
            return
        if not linhas:
            self.status(f"A rodada {pasta.name} não tem resultados.", "erro")
            return

        self.rel_pasta = pasta
        self.rel_medias = _gerar_medias_por_metodo(linhas)
        self.rel_ranking = _gerar_ranking_por_experimento(linhas)
        melhores = _gerar_melhores_por_imagem(linhas)

        imagens = sorted({l["imagem"] for l in linhas})
        experimentos = sorted({l["experimento"] for l in linhas}, key=ordem_experimento)
        self.lbl_resumo_rel.configure(
            text=f"Rodada {rotulo_rodada(pasta)}   ·   {len(imagens)} imagem(ns)   ·   "
                 f"{len(experimentos)} experimentos   ·   {len(linhas)} resultados   ·   "
                 f"os gráficos mostram a média entre as imagens")

        self._desenhar_vencedores()
        self._desenhar_grafico_relatorio()

        cat = lambda metodo: NOMES_CATEGORIA[categoria_de(metodo)]  # noqa: E731
        self.tab_ranking.preencher(
            [(nome_experimento(r["experimento"]), r["posicao"], r["metodo"], cat(r["metodo"]),
              f"{r['mse_medio']:.2f}", f"{r['psnr_medio']:.2f}", f"{r['ssim_medio']:.4f}")
             for r in sorted(self.rel_ranking,
                             key=lambda r: (ordem_experimento(r["experimento"]), r["posicao"]))],
            destacar=lambda linha: linha[1] == 1)
        self.tab_medias.preencher(
            [(nome_experimento(m["experimento"]), m["metodo"], cat(m["metodo"]), m["quantidade"],
              f"{m['mse_medio']:.2f}", f"{m['psnr_medio']:.2f}", f"{m['ssim_medio']:.4f}")
             for m in sorted(self.rel_medias,
                             key=lambda m: (ordem_experimento(m["experimento"]), m["metodo"]))])
        self.tab_melhores.preencher(
            [(m["imagem"], nome_experimento(m["experimento"]),
              m["melhor_ssim"], f"{m['valor_ssim']:.4f}",
              m["melhor_psnr"], f"{m['valor_psnr']:.2f}",
              m["melhor_mse"], f"{m['valor_mse']:.2f}") for m in melhores])
        self.tab_metricas.preencher(
            [(l["imagem"], nome_experimento(l["experimento"]), l["metodo"], cat(l["metodo"]),
              l["parametros"], f"{l['mse']:.2f}", f"{l['psnr']:.2f}", f"{l['ssim']:.4f}")
             for l in linhas])

        arquivo_param = pasta / "parametros_experimento.txt"
        texto = (arquivo_param.read_text(encoding="utf-8") if arquivo_param.is_file()
                 else "Arquivo parametros_experimento.txt não encontrado nesta rodada.")
        self.txt_param.configure(state="normal")
        self.txt_param.delete("1.0", tk.END)
        self.txt_param.insert("1.0", texto)
        self.txt_param.configure(state="disabled")

        self.mapa_exp = {nome_experimento(e): e for e in experimentos}
        self.cb_grade_img.configure(values=imagens)
        self.cb_grade_exp.configure(values=list(self.mapa_exp))
        self.cb_grade_img.current(0)
        self.cb_grade_exp.current(0)
        self._mostrar_grade_rodada()

        self._atualizar_estado_botoes()
        self.status(f"Rodada {rotulo_rodada(pasta)} carregada.", "ok")

    def _desenhar_vencedores(self):
        for w in self.frame_vencedores.winfo_children():
            w.destroy()
        primeiros = sorted((r for r in self.rel_ranking if r["posicao"] == 1),
                           key=lambda r: ordem_experimento(r["experimento"]))
        for i in range(8):
            self.frame_vencedores.grid_columnconfigure(i, weight=0, uniform="")
        for i, r in enumerate(primeiros):
            self.frame_vencedores.grid_columnconfigure(i, weight=1, uniform="v")
            cat = categoria_de(r["metodo"])
            c = card(self.frame_vencedores)
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            tk.Frame(c, bg=CORES_CATEGORIA[cat], width=px(6)).pack(side=tk.LEFT, fill=tk.Y)
            corpo = tk.Frame(c, bg=CARD)
            corpo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=8)
            tk.Label(corpo, text=f"1º LUGAR · {nome_experimento(r['experimento']).upper()}",
                     bg=CARD, fg=TEXTO_SEC, font=fnt(8, "bold")).pack(anchor="w")
            tk.Label(corpo, text=r["metodo"], bg=CARD, fg=TEXTO,
                     font=fnt(13, "bold")).pack(anchor="w")
            tk.Label(corpo, text=f"SSIM {r['ssim_medio']:.4f}  ·  PSNR {r['psnr_medio']:.2f} dB",
                     bg=CARD, fg=TEXTO, font=fnt(9)).pack(anchor="w")
            tk.Label(corpo, text=NOMES_CATEGORIA[cat], bg=CARD, fg=CORES_CATEGORIA[cat],
                     font=fnt(8, "bold")).pack(anchor="w")

    def _desenhar_grafico_relatorio(self):
        if not self.rel_medias:
            return
        metrica = self.cb_metrica.get()
        chave = METRICAS[metrica][0] + "_medio"
        experimentos = sorted({m["experimento"] for m in self.rel_medias}, key=ordem_experimento)

        fig = self.graf_rel.fig
        fig.clear()
        axes = fig.subplots(1, len(experimentos), squeeze=False).ravel()
        for ax, exp in zip(axes, experimentos):
            itens = [m for m in self.rel_medias if m["experimento"] == exp]
            desenhar_barras(ax, [m["metodo"] for m in itens], [m[chave] for m in itens],
                            metrica, titulo=nome_experimento(exp))
        legenda_categorias(fig, {categoria_de(m["metodo"]) for m in self.rel_medias})
        self.graf_rel.desenhado()

    def _nome_export_relatorio(self):
        nome = self.rel_pasta.name if self.rel_pasta else "rodada"
        return f"relatorio_{nome}_{self.cb_metrica.get().lower()}"

    def _mostrar_grade_rodada(self):
        imagem = self.cb_grade_img.get()
        experimento = self.mapa_exp.get(self.cb_grade_exp.get())
        if not imagem or not experimento or self.rel_pasta is None:
            self.painel_grade_rel.mostrar(None)
            return
        caminho = self.rel_pasta / imagem / experimento / "comparacao_grid.png"
        img = ler_imagem(caminho) if caminho.is_file() else None
        self.painel_grade_rel.mostrar(img, texto_vazio=f"Grade não encontrada:\n{caminho}")
        self._atualizar_estado_botoes()

    def _ampliar_grade_rodada(self):
        self._ampliar(self.painel_grade_rel.img,
                      f"Grade visual · {self.cb_grade_img.get()} · {self.cb_grade_exp.get()}")

    def _sub_aba_trocada(self, _evento=None):
        """Na sub-aba Grade visual o grafico e recolhido para a imagem ganhar
        todo o espaco (espremido, ele ficaria ilegivel); ao sair da sub-aba,
        o grafico volta com a divisoria na posicao anterior."""
        na_grade = self.sub.select() == str(self.aba_grade)
        try:
            if na_grade and self._sash_antes_da_grade is None:
                self._sash_antes_da_grade = self.paned_rel.sashpos(0)
                self.paned_rel.forget(self.graf_rel_card)
            elif not na_grade and self._sash_antes_da_grade is not None:
                posicao, self._sash_antes_da_grade = self._sash_antes_da_grade, None
                self.paned_rel.insert(0, self.graf_rel_card, weight=4)
                self.root.after_idle(lambda: self.paned_rel.sashpos(0, posicao))
        except tk.TclError:
            pass

    # -----------------------------------------------------------------
    # JANELA DE AMPLIACAO
    # -----------------------------------------------------------------
    def _ampliar(self, img, titulo):
        """Abre a imagem numa janela propria, redimensionavel (boa para
        examinar detalhes ou fazer capturas de tela)."""
        if img is None:
            return
        janela = tk.Toplevel(self.root)
        janela.title(titulo)
        janela.configure(bg=CREME)
        h, w = img.shape[:2]
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        escala = min((sw * 0.85) / w, (sh * 0.80) / h)
        largura, altura = int(w * escala) + px(40), int(h * escala) + px(90)
        janela.geometry(f"{largura}x{altura}+{(sw - largura) // 2}+{max((sh - altura) // 2 - 30, 0)}")
        janela.minsize(px(300), px(250))

        painel = PainelImagem(janela, titulo.upper(), "")
        painel.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 4))
        tk.Label(janela, text=f"Tamanho original: {w} x {h} px.  Redimensione a janela para "
                              f"ampliar; Esc fecha.",
                 bg=CREME, fg=TEXTO_SEC, font=fnt(9)).pack(anchor="w", padx=14, pady=(0, 8))
        janela.bind("<Escape>", lambda _e: janela.destroy())
        janela.after(60, lambda: painel.mostrar(img))
        janela.focus_set()
