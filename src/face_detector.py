# src/face_detector.py
"""
Detector de rostros usando InsightFace
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    FACE_DETECTION_MODEL,
    DETECTION_THRESHOLD,
    ONNX_PROVIDERS,
    MIN_FACE_SIZE
)

class FaceDetector:
    """Detector de rostros usando InsightFace"""
    
    def __init__(self):
        print("Inicializando detector facial...")
        self.app = FaceAnalysis(
            name=FACE_DETECTION_MODEL,
            providers=['CPUExecutionProvider']  # Solo CPU
        )
        # Tamaño reducido para mejor performance
        self.app.prepare(ctx_id=-1, det_thresh=DETECTION_THRESHOLD, det_size=(320, 320))
        print("✓ Detector facial inicializado (CPU optimizado)")
    
    def detect_faces(self, image):
        """
        Detectar rostros en una imagen
        
        Args:
            image: Imagen BGR (OpenCV)
        
        Returns:
            Lista de rostros detectados con información
        """
        # InsightFace requiere RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros
        faces = self.app.get(rgb_image)
        
        # Filtrar por tamaño mínimo
        valid_faces = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Verificar tamaño mínimo
            if width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
                valid_faces.append(face)
        
        return valid_faces
    
    def get_face_embedding(self, face):
        """
        Obtener embedding (vector de características) de un rostro
        
        Args:
            face: Objeto Face de InsightFace
        
        Returns:
            numpy array con el embedding (512 dimensiones)
        """
        return face.normed_embedding
    
    def draw_faces(self, image, faces, names=None, confidences=None):
        """
        Dibujar cuadros alrededor de los rostros detectados
        
        Args:
            image: Imagen BGR
            faces: Lista de rostros detectados
            names: Lista opcional de nombres
            confidences: Lista opcional de confianzas
        
        Returns:
            Imagen con rostros marcados
        """
        output = image.copy()
        
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Color según si hay nombre o no
            if names and names[idx]:
                color = (0, 255, 0)  # Verde - reconocido
                label = names[idx]
                if confidences and confidences[idx]:
                    label += f" ({confidences[idx]:.2%})"
            else:
                color = (0, 0, 255)  # Rojo - no reconocido
                label = "Desconocido"
            
            # Dibujar rectángulo
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar etiqueta
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = max(y1, label_size[1] + 10)
            
            # Fondo para el texto
            cv2.rectangle(
                output,
                (x1, y1_label - label_size[1] - 10),
                (x1 + label_size[0], y1_label),
                color,
                -1
            )
            
            # Texto
            cv2.putText(
                output,
                label,
                (x1, y1_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Dibujar landmarks (puntos faciales) opcionales
            if hasattr(face, 'kps') and face.kps is not None:
                kps = face.kps.astype(int)
                for kp in kps:
                    cv2.circle(output, tuple(kp), 2, (0, 255, 255), -1)
        
        return output
    
    def align_face(self, image, face):
        """
        Alinear rostro para mejorar precisión
        
        Args:
            image: Imagen BGR
            face: Objeto Face de InsightFace
        
        Returns:
            Imagen del rostro alineado (112x112)
        """
        # InsightFace ya hace alignment internamente
        # Aquí solo extraemos el rostro
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Asegurar que el bbox esté dentro de la imagen
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        face_img = image[y1:y2, x1:x2]
        
        # Redimensionar a tamaño estándar
        if face_img.size > 0:
            face_img = cv2.resize(face_img, (112, 112))
        
        return face_img
    
    def get_face_quality(self, face):
        """
        Evaluar calidad del rostro detectado
        
        Returns:
            dict con métricas de calidad
        """
        quality = {
            'detection_score': face.det_score,
            'bbox_area': 0,
            'is_frontal': True
        }
        
        # Calcular área del bbox
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        quality['bbox_area'] = width * height
        
        return quality


# ==================== PRUEBAS ====================
if __name__ == "__main__":
    print("\n=== PRUEBA DE DETECTOR FACIAL ===\n")
    
    # Inicializar detector
    detector = FaceDetector()
    
    # Probar con cámara
    print("\nProbando con cámara en vivo...")
    print("Presiona 'q' para salir\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: No se puede acceder a la cámara")
        exit()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detectar rostros cada 3 frames para mejor rendimiento
        if frame_count % 3 == 0:
            faces = detector.detect_faces(frame)
            
            # Dibujar rostros
            frame = detector.draw_faces(frame, faces)
            
            # Mostrar información
            cv2.putText(
                frame,
                f"Rostros detectados: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Mostrar frame
        cv2.imshow('Prueba de Detector Facial', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Prueba completada")