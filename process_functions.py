import cv2
import numpy as np
import matplotlib.pyplot as plt

#Funções de auxilio para plicar as transformações na imagem

def contrasteBrilho(image,gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    
    return cv2.LUT(image, lookUpTable)


def rgb_to_gray(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def threshold(image, threshold_value):

    _, binarized_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binarized_image

def adapt_threshold(image):

    binarized_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    return binarized_image

def negative(image):

    negative_image = cv2.bitwise_not(image)
    return negative_image

def detect_circles(image):

    # Obtém o número de linhas (altura) da imagem
    rows = image.shape[0]

    # Detecta círculos na imagem usando a transformada de Hough
    circles = cv2.HoughCircles(
        image,                  # Imagem de entrada (deve ser em escala de cinza)
        cv2.HOUGH_GRADIENT,     # Método de detecção (HOUGH_GRADIENT é o método clássico)
        1,                      # Razão de resolução inversa do acumulador
        rows / 8,               # Distância mínima entre os centros dos círculos detectados
        param1=100,             # Parâmetro para o detector de bordas de Canny (limite superior)
        param2=30,               # Limite para o centro de detecção do círculo
    )
    return circles

def draw_circles(image, circles, name):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 0), 3)
        # cv2.imshow("circulos", image)
        cv2.waitKey(0)
        cv2.imwrite(name + ".png", image)


def show_image(image, name):
    cv2.imshow("moedas_" + name, image)
    k = cv2.waitKey(0)
    if (k == ord("s")):
        save_name = input()
        cv2.imwrite(save_name + ".png", image)
    return