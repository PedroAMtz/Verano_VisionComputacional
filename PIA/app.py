import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time 

with open("description.txt", encoding="utf-8") as f:
    text = f.read()

imagen = cv2.imread("617.tif")

def detectar_lineas(imagen_path):   
    imagen = cv2.imread(imagen_path)
    imagen_color = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # contructor de función LSD()
    lsd = cv2.createLineSegmentDetector()
    # Se utiliza el método detect() para detectar las líneas presentes en la imagen
    lineas, _, _, _ = lsd.detect(gray)
    
    line_image = np.copy(imagen_color)
    
    if lineas is not None:
        lineas = lineas.reshape((-1, 4))
        for x1, y1, x2, y2 in lineas:
                # imagen, coordenada inicio, coordenada final, color, grosor
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
    return line_image

imagen_lineas = detectar_lineas("617.tif")

st.title("Prodcuto Integrador de Aprendizaje")
st.subheader("Visión Computacional :eye:")
st.write(text)


st.image([imagen, imagen_lineas],
         width=250,
         caption=["Imagen Original",
                  "Imagen con LSD"])




