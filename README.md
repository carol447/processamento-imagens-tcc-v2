# Processamento de imagens — TCC, versão 2

Protótipo de computação científica para comparar filtros clássicos e métodos baseados em equações diferenciais parciais na restauração de imagens.

## Recursos

- Ruídos gaussiano, sal e pimenta e speckle; degradações para inpainting.
- Filtros de média, gaussiano, mediana e laplaciano.
- Difusão isotrópica, Perona–Malik e implementações de inpainting identificadas no código.
- Comparação por MSE, PSNR e SSIM.
- Interface Tkinter, dashboard e exportação de experimentos.

## Execução

```bash
python -m pip install -r requirements.txt
python main_dashboard.py
```

Para a interface original, execute `python main.py`. Selecione uma imagem própria ou com autorização de uso na interface. Imagens e resultados locais não acompanham o repositório.

## Tecnologias

Python, NumPy, OpenCV, scikit-image, Matplotlib, Pillow e Tkinter. Tkinter precisa estar disponível na instalação do Python; os experimentos são implementações de pesquisa, sujeitos à validação dos parâmetros e condições numéricas.
