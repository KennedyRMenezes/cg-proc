import cv2
import numpy as np
import matplotlib.pyplot as plt

#Funções de auxilio para plicar as transformações na imagem

def contrasteBrilho(image,alpha,beta):
    print("hello")
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

def negative(image):

    negative_image = cv2.bitwise_not(image)
    return negative_image

def show_image(image, name):
    cv2.imshow("moedas_" + name, image)
    k = cv2.waitKey(0)
    if (k == ord("s")):
        save_name = input()
        cv2.imwrite(save_name + ".png", image)
    return