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

def dibujar_lineas(img, lineas, lsd):
    img_lineas = np.zeros_like(img)
    try:
        img_lineas = lsd.drawSegments(img, lineas)
    except ValueError as ve:
        print("No hay lineas que dibujar.")
    return img_lineas

# LineSegementDetector construnctor
line_sd = cv2.createLineSegmentDetector()


cap = cv2.VideoCapture("test1.mp4")
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

# New feature, might be a function later
    scale_percent = 40 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    resized_canny = cv2.resize(canny_image, dim)
    resized_final = cv2.resize(imagen_lsd, dim)
    resized_cropped = cv2.resize(cropped_canny, dim)



    resized_canny = cv2.cvtColor(resized_canny, cv2.COLOR_GRAY2BGR)
    resized_cropped = cv2.cvtColor(resized_cropped, cv2.COLOR_GRAY2BGR)
    doble = np.hstack((resized_canny, resized_cropped, resized_final), casting='safe')

    cv2.imshow("canny_edges", doble)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()