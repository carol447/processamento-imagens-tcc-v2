import os
import tkinter as tk
from pathlib import Path

# garante que img/ e resultados_experimentos/ sejam encontrados
os.chdir(Path(__file__).resolve().parent)

from interface_dashboard import Dashboard, ativar_dpi_nitido

if __name__ == "__main__":
    ativar_dpi_nitido()  # texto nitido em telas com escala (125%, 150%...)
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()
