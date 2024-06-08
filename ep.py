import cv2
import matplotlib.pyplot as plt
import pathlib
from process_functions import *

def main():

    master_path = pathlib.Path(__file__).parent.resolve()

    # Paths das imagens
    one_each = "{}/img/one_each.jpeg".format(master_path)
    glued = "{}/img/glued.jpeg".format(master_path)
    mess = "{}/img/mess.jpeg".format(master_path)
    real_and_twentyFive = "{}/img/real_and_twentyFive.jpeg".format(master_path)

    kernel = np.ones((2,2),np.uint8)
    kernel_erosion = np.ones((20,20),np.uint8)

    # Dict para usar o nome da variável na hora de salvar o arquivo alterado.
    images_dict = {
        'one_each': one_each,
        'glued': glued,
        'mess': mess,
        'real_and_twentyFive': real_and_twentyFive
    }


    #Ajuste de brilho e contraste
    for name, img_path in images_dict.items():
        image = cv2.imread(img_path)
        brilho = 100
        contraste = 1.5
        #image = contrasteBrilho(image,brilho,contraste)

    # Segunda etapa: um novo threshold para deixar a imagem mais "flat": sem as sombras das imagens
    for name, img_path in images_dict.items():
        image = cv2.imread(img_path)
        original = image

        image = contrasteBrilho(image,5)
        cv2.imwrite("{}/img/edited/final/0_{}_brilho_contraste.jpeg".format(master_path, name), image)

        image = cv2.medianBlur(image,11)
        cv2.imwrite("{}/img/edited/final/1_{}_blur.jpeg".format(master_path, name), image)

        image = rgb_to_gray(image)
        cv2.imwrite("{}/img/edited/final/2_{}_black_white.jpeg".format(master_path, name), image)

        image = adapt_threshold(image)
        cv2.imwrite("{}/img/edited/final/3_{}_adaptative_treshold.jpeg".format(master_path, name), image)

        #####Esse blur aparentemente desmancha mais ainda o círculo
        image = cv2.medianBlur(image,7)
        # show_image(image, "blur")
        ###########################################################

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        # show_image(image, "open")
        # image = cv2.medianBlur(image,13)
        # show_image(image, "blur2")
        cv2.imwrite("{}/img/edited/final/4_{}_fechamento.jpeg".format(master_path, name), image)
        
        #Função de detecção de círculos usando Hough.
        circles = detect_circles(image)
        print("\nCircles: \n",circles,"\nFrom image: ",name)

        #Desenha na imagem os círculos encontrados.
        draw_circles(original, circles, "{}/img/edited/final/circles/{}".format(master_path, name))

        cv2.destroyAllWindows()
        # image = cv2.erode(image, kernel_erosion, iterations=6)

    # # Terceira etapa: fazendo o negativo da imagem para trabalhar com as cores certas.
    # images_dict_edited = {
    #     'one_each_': "{}/img/edited/t_hold_one_each.jpeg".format(master_path),
    #     'glued_': "{}/img/edited/t_hold_glued.jpeg".format(master_path),
    #     'mess_': "{}/img/edited/t_hold_mess.jpeg".format(master_path),
    #     'real_and_twentyFive_': "{}/img/edited/t_hold_real_and_twentyFive.jpeg".format(master_path)
    # }

    # for name, img_path in images_dict_edited.items():
    #     image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #     median_blurred = cv2.medianBlur(image,7)
    #     threshold_value = 135
    #     binarized_image = threshold(median_blurred, threshold_value)
    #     negative_image = negative(binarized_image)
    #     cv2.imwrite("{}/img/edited/{}_negative.jpeg".format(master_path, name), negative_image)

main()