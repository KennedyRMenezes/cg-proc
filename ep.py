import cv2
import numpy as np

import pathlib
from process_functions import *  

def main():

    master_path = pathlib.Path(__file__).parent.resolve()

    # Dicionário para facilitar a refêrencia das imagens antes do threshold
    image_base_path = "{}/img/".format(master_path)

    total_images = count_images_in_folder(image_base_path)

    images_dict = {}

    for i in range(1, total_images + 1):
        image_name = f"{i}.jpeg"  # Nome da imagem
        image_path = image_base_path + image_name
        images_dict[f"{i}"] = image_path  # 1, 2 ..., 3

    # Primeira etapa: threshold com o método de OTSU
    # Método de OTSU: um dos mais populares algoritmos de threshold que determina o limiar ótimo
    for name, img_path in images_dict.items():
        image = cv2.imread(img_path)
        brilho = 90
        contraste = 1.40
        image = contrasteBrilho(image, brilho, contraste)
        '''cv2.imwrite(
            "{}/img/edited/ContrasteBrilho_{}.jpeg".format(master_path, name),
            image,
        )'''
        
        # Aplicar filtro Gaussiano
        gaussian_blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        '''cv2.imwrite(
            "{}/img/edited/FiltroGaussiano_{}.jpeg".format(master_path, name),
            gaussian_blurred,
        )'''
        gray_image = rgb_to_gray(gaussian_blurred)
        '''cv2.imwrite(
            "{}/img/edited/grayImage_{}.jpeg".format(master_path, name),
            gray_image,
        )'''
        binarized_image = OTSU_threshold(gray_image)
        cv2.imwrite(
            "{}/img/edited/t_hold_OTSU_{}.jpeg".format(master_path, name),
            binarized_image,
        )

    # Dicionário para facilitar a refêrencia das imagens após o threshold

    edited_image_base_path = "{}/img/edited/t_hold_OTSU_".format(master_path)

    images_dict_edited = {}

    for i in range(1, (total_images + 1)):
        image_name = f"{i}.jpeg"  # Nome da imagem editada
        image_path = edited_image_base_path + image_name
        images_dict_edited[f"{i}_"] = image_path

    list_circles = []
    list_images = []

    for name, img_path in images_dict_edited.items():
        list_images.append(img_path)

        # TESTE: Ler a imagem em tons de cinza e aplicar algumas correções
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        brilho = 100
        contraste = 1.4
        image = contrasteBrilho(image, brilho, contraste)

        # Inverter as cores da imagem
        negative_image = negative(image)
        '''cv2.imwrite(
            "{}/img/edited/{}_NEGATIVE.jpeg".format(master_path, name), negative_image
        )'''

        # Preencher buracos na imagem e adicionar a imagem na pasta EDITED
        filled_image = fill_holes(negative_image)
        '''cv2.imwrite(
            "{}/img/edited/{}_FILLED.jpeg".format(master_path, name), filled_image
        )'''

        # Remover ruído usando abertura
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(filled_image, cv2.MORPH_OPEN, kernel, iterations=2)

        # Área de fundo seguro
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Área de primeiro plano seguro
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        cv2.imwrite("{}/img/edited/{}_DIST.jpeg".format(master_path, name), dist_transform)
        sure_fg = threshold(dist_transform, 0.6 * dist_transform.max())

        # Como descrito no relatório, lógica utilizada das seguintes fontes:
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
        # https://medium.com/turing-talks/transforma%C3%A7%C3%A3o-watershed-com-opencv-68a5bd8196a0
        # Área da região desconhecida
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Rotulagem de marcadores
        ret, markers = cv2.connectedComponents(sure_fg)

        # Adicionar um a todos os rótulos para que o fundo seguro não seja 0, mas 1
        markers = markers + 1

        # Marcar a região desconhecida com zero
        markers[unknown == 255] = 0

        # Aplicar o algoritmo de watershed
        original_image = cv2.imread(img_path)
        markers = cv2.watershed(original_image, markers)
        original_image[markers == -1] = [255, 0, 0] 

        # Imagens Intermediárias
        cv2.imwrite("{}/img/edited/{}_opening.jpeg".format(master_path, name), opening)
        cv2.imwrite("{}/img/edited/{}_sure_bg.jpeg".format(master_path, name), sure_bg)
        cv2.imwrite("{}/img/edited/{}_sure_fg.jpeg".format(master_path, name), sure_fg)
        cv2.imwrite("{}/img/edited/{}_unknown.jpeg".format(master_path, name), unknown)
        cv2.imwrite(
            "{}/img/edited/{}_markers.jpeg".format(master_path, name),
            markers.astype(np.uint8),
        ) 

        # Função de seleção de círculo (HOUGH)
        circles = detect_circles(markers.astype(np.uint8))
        #circles = detect_circles(filled_image) # UTILIZADO PARA TESTES --> SE SAIU PIOR NOS TESTES

        list_circles.append(circles)

        # Coordenadas utilizadas para o processamento geradas pelo Algoritmo de Hough

        teste = cv2.imread(
            "{}/img/{}.jpeg".format(master_path, name.rstrip("_"))
        )  # O _ porque não temos isso na imagem inicial

        # Desenha uma linha vermelha ao redor das moedas e adiciona na pasta circles
        draw_circles(
            teste,
            circles,
            "{}/img/edited/final/circles/{}.jpeg".format(master_path, name),
        )

    dicionario_de_circulos_encontrados = converter_para_dicionario(list_circles)

    #Maior circulo de cada imagem sera definida pelo usuario
    maiores_circulos_de_cada_image = {}


    #Raio das moedas em centiemtros
    raio_moedas = {5: 11, 10: 10, 25: 12.5, 50: 11.5, 100: 13.5}

    
    for imagem, circulo in dicionario_de_circulos_encontrados.items():
        maior_circulo = float("-inf")
        for coordenada in circulo:
            raio = coordenada[2]
            if raio > maior_circulo:
                maior_circulo = raio
        maiores_circulos_de_cada_image[imagem] = maior_circulo

    list_path = []
    for i in range(total_images):   
        i = i + 1
        img_path = "{}/img/{}.jpeg".format(master_path, i)
        list_path.append(img_path)

    #Lista da maior moeda para cada imagem devolvida pelo usuario
    maior_moeda = []

    # Exibir janela para todas as imagens
    maior_moeda = interface_usuario(list_path)

    #Valores reais contido nas imagens
    moedas = {
        "1": 3.55,
        "2": 6.10,
        "3": 8.00,
        "4": 6.45,
        "5": 6.20,
        "6": 2.80,
        "7": 2.50,
        "8": 2.50,
        "9": 1.60,
        "10": 1.60,
    }

    #Dicionario para os raios relativos das imagens.
    #Fará a proporção de cada raio levando em consideração
    # o maior raio devolvido pelo usuário
    raio_rel = {}
    
    for imagem in dicionario_de_circulos_encontrados:

        #Esse será o valor(value) do dicionário
        rel = []

        #Maior circulo da imagen n, n+1, n+2...
        maior = maiores_circulos_de_cada_image[imagem]

        #Pega a maior moeda de cada imagem indicada pelo usuario - maior_moeda
        #Relaciona com a mesma moeda no dicionario raio_moeda em que tem o raio das moedas em centímetros
        #Por fim faz a relação (pela divisão) com a maior moeda
        #maior == pixel
        #maior_moeda == pixel
        #raio_moedas == centrímetros
        #ratio == cent/pixel
        ratio = (raio_moedas[maior_moeda[int(imagem) - 1]]) / maior

        #O for passa por cada círculo da imagem e faz a relação multiplicando pelo raio de cada moeda
        for coordenada in dicionario_de_circulos_encontrados[imagem]:

            #rel == (cent/pixel) * pixel == centimetros
            rel.append(coordenada[2] * ratio)

        raio_rel[imagem] = rel

    #Devolve a soma da imagem com relação nos raios da imagem calculados.
    total_centavos_dict = encontrar_moeda_mais_proxima(raio_rel, raio_moedas)

    #Exibição de quantos reais há na imagem
    erro = {}
    texto_resultado = ""
    erro_texto = ""
    for chave, valor in total_centavos_dict.items():

        valor_em_real = round((valor / 100), 2)
        num_format = "{:.2f}".format(valor_em_real)

        erro[chave] = "{:.2f}".format(abs(moedas[chave] - float(num_format)))

        texto_resultado += f"A imagem {chave} tem R$ {num_format}\n"
        erro_texto += f"o erro da imagem {chave} é {erro[chave]}\n"

    # Exibir janela com o resultado
    janela_resultado = JanelaTexto(texto_resultado)
    janela_resultado.root.mainloop()

    janela_resultado = JanelaTexto(erro_texto)
    janela_resultado.root.mainloop()

main()
