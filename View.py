import cv2

IDX = 1  # <-- prueba primero con 0, luego cambias a 1

cap = cv2.VideoCapture(IDX)

if not cap.isOpened():
    print(f"No se pudo abrir la cámara {IDX}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Sin frame de la cámara")
        break

    cv2.imshow(f"Camara {IDX} - presiona 'q' para salir", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
