import cv2

print("Probando indices de cámara...")
for idx in range(5):  # prueba 0,1,2,3,4
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # CAP_DSHOW ayuda en Windows
    if not cap.isOpened():
        print(f"Índice {idx}: no se pudo abrir")
        continue

    ret, frame = cap.read()
    if not ret:
        print(f"Índice {idx}: se abrió pero no hay frame")
        cap.release()
        continue

    print(f"Índice {idx}: OK, mostrando imagen (presiona cualquier tecla para cerrar)")
    cv2.imshow(f"Cam index {idx}", frame)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
