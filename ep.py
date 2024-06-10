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

    for i in range(1,total_images+1):
        image_name = f"{i}.jpeg"  # Nome da imagem
        image_path = image_base_path + image_name  
        images_dict[f"{i}"] = image_path  # 1, 2 ..., 3

    #Ajuste de brilho e contraste (TESTE)
    for name, img_path in images_dict.items():
        image = cv2.imread(img_path)
        brilho = 100
        contraste = 1.5
        image = contrasteBrilho(image,brilho,contraste)

    # Primeira etapa: threshold com o método de OTSU
    # Método de OTSU: um dos mais populares algoritmos de threshold que determina o limiar ótimo
    for name, img_path in images_dict.items():
        image = cv2.imread(img_path)
        gray_image = rgb_to_gray(image)
        binarized_image = OTSU_threshold(gray_image)
        cv2.imwrite("{}/img/edited/t_hold_OTSU_{}.jpeg".format(master_path, name), binarized_image)

    # Dicionário para facilitar a refêrencia das imagens após o threshold

    edited_image_base_path = "{}/img/edited/t_hold_OTSU_".format(master_path)

    images_dict_edited = {}

    for i in range(1, (total_images+1)):
        image_name = f"{i}.jpeg"  # Nome da imagem editada
        image_path = edited_image_base_path + image_name  
        images_dict_edited[f"{i}_"] = image_path  

    list_circles = []
    list_images = []

    for name, img_path in images_dict_edited.items():
        list_images.append(img_path)

        # Ler a imagem em tons de cinza e aplicar algumas correções
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        brilho = 100
        contraste = 1.2
        image = contrasteBrilho(image, brilho, contraste)

        # Aplicar filtro de mediana
        median_blurred = cv2.medianBlur(image, 7)

        # Inverter as cores da imagem
        negative_image = negative(median_blurred)

        # Preencher buracos na imagem e adicionar a imagem na pasta EDITED
        filled_image = fill_holes(negative_image)
        cv2.imwrite("{}/img/edited/{}_FILLED.jpeg".format(master_path, name), filled_image)
        
        # Remover ruído usando abertura 
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(filled_image, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Área de fundo seguro
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Área de primeiro plano seguro
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        #cv2.imwrite("{}/img/edited/{}_DIST.jpeg".format(master_path, name), dist_transform)
        ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
        
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
        
    
        # Intermediate images
        cv2.imwrite("{}/img/edited/{}_opening.jpeg".format(master_path, name), opening)
        cv2.imwrite("{}/img/edited/{}_sure_bg.jpeg".format(master_path, name), sure_bg)
        cv2.imwrite("{}/img/edited/{}_sure_fg.jpeg".format(master_path, name), sure_fg)
        cv2.imwrite("{}/img/edited/{}_unknown.jpeg".format(master_path, name), unknown)
        cv2.imwrite("{}/img/edited/{}_markers.jpeg".format(master_path, name), markers.astype(np.uint8))
        cv2.imwrite("{}/img/edited/{}_final.jpeg".format(master_path, name), original_image)
        

        # Função de seleção de círculo (HOUGH)
        circles = detect_circles(markers.astype(np.uint8))
        #circles = detect_circles(filled_image) # UTILIZADO PARA TESTES
        # print(circles)

        list_circles.append(circles)

        # Coordenadas utilizadas para o processamento geradas pelo Algoritmo de Hough
        #print("\nCircles: \n",circles,"\nFrom image: ",name)

        teste = cv2.imread("{}/img/{}.jpeg".format(master_path, name.rstrip('_'))) # O _ porque não temos isso na imagem inicial
        
        # Desenha uma linha vermelha ao redor das moedas e adiciona na pasta circles
        draw_circles(teste, circles, "{}/img/edited/final/circles/{}.jpeg".format(master_path, name))

    new_dict = converter_para_dicionario(list_circles)
        
    maiores_valores = {}
    maior_moeda = []
    raio_moedas = {5:11, 10:10, 25:12.5, 50:11.5, 100:13.5}
    for chave, valor in new_dict.items():
        maior_valor = float('-inf')
        for coordenada in valor:
            raio = coordenada[2]
            if raio > maior_valor:
                maior_valor = raio
        maiores_valores[chave] = maior_valor


    list_path = []
    for i in range(total_images):
        i = i+1
        img_path = "{}/img/{}.jpeg".format(master_path, i)
        list_path.append(img_path)

    # Exibir janela para todas as imagens
    maior_moeda = exibir_janela_imagens(list_path)

    raio_rel = {}

    for chave in new_dict:
        rel = []
        maior = maiores_valores[chave]
        ratio = (raio_moedas[maior_moeda[int(chave) - 1]])/maior
        for coordenada in new_dict[chave]:
            rel.append(coordenada[2] * ratio)
        raio_rel[chave] = rel

    total_centavos_dict = encontrar_moeda(raio_rel, raio_moedas)

    # for chave, valor in total_centavos_dict.items():
    #     valor_em_real = round((valor / 100), 2)
    #     num_format = "{:.2f}".format(valor_em_real)
    #     print("A imagem {} tem R$ {}".format(chave, num_format))

    # Preparando texto para exibição na janela
    texto_resultado = ""
    for chave, valor in total_centavos_dict.items():
        valor_em_real = round((valor / 100), 2)
        num_format = "{:.2f}".format(valor_em_real)
        texto_resultado += f"A imagem {chave} tem R$ {num_format}\n"

    # Exibir janela com o resultado
    janela_resultado = JanelaTexto(texto_resultado)
    janela_resultado.root.mainloop()
    
main()