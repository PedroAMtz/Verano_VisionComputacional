import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time 
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

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

url_string = "https://img.freepik.com/free-vector/white-abstract-background-design_23-2148825582.jpg"

st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("{url_string}");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

st.markdown("<h1 style='text-align:center;'>Prodcuto Integrador de Aprendizaje</h1>",
         unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h2 style='text-align:center;'>Visión Computacional</h2>",
            unsafe_allow_html=True)
st.write(text)


st.image([imagen, imagen_lineas],
         width=250,
         caption=["Imagen Original",
                  "Imagen con LSD"])
