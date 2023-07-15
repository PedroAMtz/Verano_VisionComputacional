import numpy as np
import cv2

def detectar_bordes(img, kernel=5):
    try: 
        imagen_grises = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = kernel
        img_filtrada = cv2.GaussianBlur(imagen_grises,(kernel, kernel),0)
        canny = cv2.Canny(img_filtrada, 50, 150)
    except: 
        cap.release()
        cv2.destroyAllWindows()
        exit()
    return canny

def region_interes(img_bordes, corx=700, ctx=100):
    alto = img_bordes.shape[0]
    ancho = img_bordes.shape[1]

    x1 = ancho/2 - corx
    x2 = ancho/2 + corx

    centro_x = ancho/2 + ctx
    centro_y = alto/2

    mask = np.zeros_like(img_bordes)
    triangulo = np.array([[
        (x1, alto),
        (centro_x, centro_y),
        (x2, alto)]], np.int32)
    cv2.fillPoly(mask, triangulo, 255)
    img_mask = cv2.bitwise_and(img_bordes, mask)
    return img_mask

def detectar_lineas(img, lsd):
    lineas, _, _, _ = lsd.detect(img)
    return lineas

def _dibujar_lineas(img, lineas):
    imagen_lineas = np.copy(img)
    if lineas is not None:
        lineas = lineas.reshape((-1, 4))
        for x1, y1, x2, y2 in lineas:
                # imagen, coordenada inicio, coordenada final, color, grosor
                cv2.line(imagen_lineas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                
    return imagen_lineas

# Funciones de utilidad

def dibujar_lineas(img, lineas, lsd):
    img_lineas = np.zeros_like(img)
    try:
        img_lineas = lsd.drawSegments(img, lineas)
    except ValueError as ve:
        print("No hay lineas que dibujar.")
    return img_lineas

def escalar_dims(img, pct_escala=50):
    alto = img.shape[0] * pct_escala / 100
    ancho = img.shape[1] * pct_escala / 100

    return (int(ancho), int(alto))


def resize_imagenes(lista_img):
    resized_imgs = []
    for index, imagen in enumerate(lista_img):
        img_para_resize = cv2.resize(imagen, escalar_dims(imagen))
        resized_imgs.append(img_para_resize)
    return resized_imgs


# LineSegementDetector construnctor
line_sd = cv2.createLineSegmentDetector()


cap = cv2.VideoCapture("test_video.mp4")
frame_count = 0
while(cap.isOpened()):
    _, frame = cap.read()
    frame_count += 1

    if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    canny_image = detectar_bordes(frame)
    cropped_canny = region_interes(canny_image)
    lineas_imagen = detectar_lineas(cropped_canny, line_sd)
    imagen_lsd = _dibujar_lineas(frame, lineas_imagen)

    lista_imgs = [canny_image, cropped_canny, imagen_lsd]
    imgs_para_mostrar = resize_imagenes(lista_imgs)

    resized_canny = cv2.cvtColor(imgs_para_mostrar[0], cv2.COLOR_GRAY2BGR)
    resized_cropped = cv2.cvtColor(imgs_para_mostrar[1], cv2.COLOR_GRAY2BGR)
    triple = np.hstack((resized_canny, resized_cropped, imgs_para_mostrar[2]), casting='safe')

    cv2.imshow("canny_edges", triple)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()