
import cv2
import numpy as np
import os
import imghdr
from PIL import Image, ImageTk
import tkinter as tk
from screeninfo import get_monitors

def count_images_in_folder(folder_path):
    image_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and imghdr.what(file_path):
            image_count += 1
    return image_count

def contrasteBrilho(image,alpha,beta):
    return cv2.convertScaleAbs(image, alpha, beta)

def rgb_to_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def threshold(image, threshold_value):
    _, binarized_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binarized_image

def adapt_threshold(image):
    binarized_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    return binarized_image

def OTSU_threshold(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def negative(image):
    negative_image = cv2.bitwise_not(image)
    return negative_image

def fill_holes(image):
    # Imagem para o preenchimento é uma cópia
    filled_image = image.copy()
    
    # Outra imagem com fundo branco, para modificação
    h, w = image.shape
    white_image = np.ones((h, w), np.uint8) * 255
    
    # Flood fill a partir de um ponto fora da imagem (por exemplo, (0, 0))
    flood_filled = image.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Máscara sendo 2 pixels maior que a imagem
    cv2.floodFill(flood_filled, mask, (0, 0), 255)
    
    # Inverter a imagem de flood fill
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    
    # Obter somente os buracos
    holes = cv2.bitwise_and(white_image, white_image, mask=flood_filled_inv)
    
    # Adicionar os buracos de volta à imagem original
    filled_image = cv2.bitwise_or(image, holes)
    
    return filled_image

def detect_circles(image):
    # Parametros da Transformada de Hough
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=27,
        param2=26,
        minRadius=17,
        maxRadius=100
    )
    return circles

# Função para desenhar círculos detectados na imagem
def draw_circles(image, circles, save_path):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Desenhar o contorno do círculo com borda grossa e vermelha
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 5)
            # Desenhar o centro do círculo com borda grossa e vermelha
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imwrite(save_path, image)

#Converte a lista devolvida por 'detect_circles' para um dicionario para atrelar a imagem aos circulos encontrados
def converter_para_dicionario(listas):
    dicionario = {}
    for i, arr in enumerate(listas):
        identificador = f"{i + 1}"
        coordenadas = arr[0].tolist()  # Converter o array para lista
        dicionario[identificador] = coordenadas
    return dicionario


def encontrar_moeda_mais_proxima(raio_rel, raio_moedas):
        total_centavos_dict = {}
        for imagem, circulos in raio_rel.items():
            total_centavos = 0
            for raio in circulos:
                mais_proximo = min(raio_moedas.values(), key=lambda x: abs(x - raio))
                moeda = [k for k, v in raio_moedas.items() if v == mais_proximo][0]
                total_centavos += moeda
            total_centavos_dict[imagem] = total_centavos
        return total_centavos_dict

class JanelaImagem:
    def __init__(self, imagem_path, largura_imagem, texto_input):
        self.root = tk.Toplevel()
        self.root.title("Janela de Imagem")

        self.imagem_path = imagem_path
        self.largura_imagem = largura_imagem
        self.texto_input = texto_input

        self.imagem_original = Image.open(imagem_path)
        self.imagem_redimensionada = self.redimensionar_imagem(self.imagem_original)

        self.imagem_tk = ImageTk.PhotoImage(self.imagem_redimensionada)

        self.label_imagem = tk.Label(self.root, image=self.imagem_tk)
        self.label_imagem.pack()

        self.label_input = tk.Label(self.root, text=texto_input)
        self.label_input.pack()

        self.entry_input = tk.Entry(self.root)
        self.entry_input.pack()

        self.entry_input.bind("<Return>", self.confirmar)  # Ligando o evento 'Enter' à função confirmar

        self.botao_confirmar = tk.Button(self.root, text="Confirmar", command=self.confirmar)
        self.botao_confirmar.pack()

        self.resposta_usuario = None

    def redimensionar_imagem(self, imagem):
        largura_original, altura_original = imagem.size
        proporcao = self.largura_imagem / largura_original
        nova_largura = int(largura_original * proporcao)
        nova_altura = int(altura_original * proporcao)
        return imagem.resize((nova_largura, nova_altura))

    def confirmar(self, event=None):  # O argumento event é necessário para a ligação com o evento 'Enter'
        resposta = self.entry_input.get()
        self.resposta_usuario = resposta
        self.root.destroy()  # Fechar a janela atual

#Função que recebe uma lista de imagens, uma string, e a largura da image.
#Para cada imagem recebe o input do usuario, guarda elas e devolve como 'respostas'
def interface_usuario(
        lista_de_imagens, 
        texto_input="Qual a moeda de maior tamanho na foto? 100 = 1 Real; 50 = 50 centavos",
        largura_imagem=550
):
    respostas = []  # Dicionário para armazenar as respostas do usuário
    for i, imagem_path in enumerate(lista_de_imagens):
        janela_imagem = JanelaImagem(imagem_path, largura_imagem, texto_input)
        janela_imagem.root.wait_window()  # Esperar até que a janela seja fechada
        respostas.append(int(janela_imagem.resposta_usuario))
    return respostas

#Janela em TKinter para apresentar os resultado da soma das imagens
class JanelaTexto:
    def __init__(self, texto):
        self.root = tk.Toplevel()
        self.root.title("Resultado")
        
        # Calculando as dimensões do monitor
        monitor = get_monitors()[0]  # Assumindo que há apenas um monitor
        largura_monitor, altura_monitor = monitor.width, monitor.height
        
        # Definindo o tamanho da janela
        largura_janela = largura_monitor // 2
        altura_janela = altura_monitor // 2
        pos_x = (largura_monitor - largura_janela) // 2
        pos_y = (altura_monitor - altura_janela) // 2
        self.root.geometry(f"{largura_janela}x{altura_janela}+{pos_x}+{pos_y}")
        
        # Exibindo o texto
        self.label_texto = tk.Label(self.root, text=texto)
        self.label_texto.pack()
        
        # Botão para fechar a janela
        self.botao_fechar = tk.Button(self.root, text="Fechar", command=self.root.destroy)
        self.botao_fechar.pack()

'''


'''