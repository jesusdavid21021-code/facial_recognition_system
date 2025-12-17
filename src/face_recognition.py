# src/face_recognition.py
"""
Sistema de reconocimiento facial usando embeddings
"""
import pickle
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    EMBEDDINGS_PATH,
    RECOGNITION_THRESHOLD,
    EMBEDDING_SIZE
)
from src.face_detector import FaceDetector
from src.database import DatabaseManager

class FaceRecognizer:
    """Sistema de reconocimiento facial"""

    def __init__(self, detector=None, db=None):
        self.detector = detector or FaceDetector()
        self.db = db or DatabaseManager()
        self.embeddings_db: dict[int, np.ndarray] = {}  # {employee_id: embeddings array}
        self.load_embeddings()

    def _normalize_embedding(self, embedding) -> np.ndarray:
        """Devuelve el embedding normalizado a float32 de tamaño fijo."""

        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if arr.size != EMBEDDING_SIZE:
            raise ValueError("Embedding con tamaño inválido")

        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def _normalize_embedding_list(self, embeddings) -> np.ndarray | None:
        normalized: list[np.ndarray] = []

        for emb in embeddings:
            try:
                normalized.append(self._normalize_embedding(emb))
            except ValueError:
                continue

        if not normalized:
            return None

        return np.vstack(normalized)
    
    def load_embeddings(self):
        """Cargar embeddings guardados"""
        if EMBEDDINGS_PATH.exists():
            with open(EMBEDDINGS_PATH, 'rb') as f:
                raw_db = pickle.load(f)

            loaded = 0
            cleaned_db: dict[int, np.ndarray] = {}
            for emp_id, embeddings in raw_db.items():
                normalized = self._normalize_embedding_list(embeddings)
                if normalized is None:
                    continue

                cleaned_db[int(emp_id)] = normalized
                loaded += 1

            self.embeddings_db = cleaned_db
            print(f"✓ Embeddings cargados: {loaded} empleados")
        else:
            print("⚠ No hay embeddings guardados. Ejecuta entrenamiento primero.")
    
    def save_embeddings(self):
        """Guardar embeddings a disco"""
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(self.embeddings_db, f)
        print(f"✓ Embeddings guardados: {len(self.embeddings_db)} empleados")
    
    def add_employee_embeddings(self, employee_id, embeddings):
        """
        Agregar embeddings de un empleado
        
        Args:
            employee_id: ID del empleado
            embeddings: Lista de embeddings (numpy arrays)
        """
        normalized = self._normalize_embedding_list(embeddings)
        if normalized is None:
            raise ValueError("No se recibieron embeddings válidos")

        self.embeddings_db[int(employee_id)] = normalized
        self.save_embeddings()
        print(
            f"✓ Embeddings agregados para empleado {employee_id}: "
            f"{normalized.shape[0]} fotos"
        )
    
    def remove_employee_embeddings(self, employee_id):
        """Eliminar embeddings de un empleado"""
        if employee_id in self.embeddings_db:
            del self.embeddings_db[employee_id]
            self.save_embeddings()
            print(f"✓ Embeddings eliminados para empleado {employee_id}")
    
    def recognize_face(self, face_embedding):
        """
        Reconocer un rostro comparando con embeddings guardados
        
        Args:
            face_embedding: Embedding del rostro a reconocer
        
        Returns:
            tuple: (employee_id, confidence, employee_data) o (None, 0, None)
        """
        if not self.embeddings_db:
            return None, 0.0, None

        try:
            target = self._normalize_embedding(face_embedding)
        except ValueError:
            return None, 0.0, None

        best_match_id = None
        best_similarity = 0.0

        for employee_id, stored_embeddings in self.embeddings_db.items():
            if stored_embeddings.size == 0:
                continue

            similarities = stored_embeddings @ target

            if similarities.size == 0:
                continue

            top_count = min(5, similarities.size)
            # np.partition evita ordenar todo el array
            top_scores = np.partition(similarities, -top_count)[-top_count:]
            avg_similarity = float(np.mean(top_scores))

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match_id = employee_id

        if best_match_id is not None and best_similarity >= RECOGNITION_THRESHOLD:
            employee_data = self.db.get_employee(best_match_id)
            return best_match_id, best_similarity, employee_data

        return None, best_similarity, None
    
    def recognize_faces_in_frame(self, frame):
        """
        Reconocer todos los rostros en un frame
        
        Args:
            frame: Imagen BGR (OpenCV)
        
        Returns:
            tuple: (frame_anotado, lista_resultados)
        """
        # Detectar rostros
        faces = self.detector.detect_faces(frame)
        
        results = []
        names = []
        confidences = []
        
        for face in faces:
            # Obtener embedding
            embedding = self.detector.get_face_embedding(face)
            
            # Reconocer
            employee_id, confidence, employee_data = self.recognize_face(embedding)
            
            if employee_data:
                name = f"{employee_data['nombre']} {employee_data['apellido']}"
                cargo = employee_data['cargo']
                label = f"{name} - {cargo}"
            else:
                label = None
            
            names.append(label)
            confidences.append(confidence)
            
            results.append({
                'employee_id': employee_id,
                'confidence': confidence,
                'employee_data': employee_data,
                'bbox': face.bbox
            })
        
        # Dibujar rostros en el frame
        annotated_frame = self.detector.draw_faces(frame, faces, names, confidences)
        
        return annotated_frame, results
    
    def get_recognition_stats(self):
        """Obtener estadísticas del sistema de reconocimiento"""
        stats = {
            'total_employees_trained': len(self.embeddings_db),
            'total_embeddings': sum(embs.shape[0] for embs in self.embeddings_db.values()),
            'avg_embeddings_per_employee': 0,
            'threshold': RECOGNITION_THRESHOLD
        }
        
        if stats['total_employees_trained'] > 0:
            stats['avg_embeddings_per_employee'] = (
                stats['total_embeddings'] / stats['total_employees_trained']
            )
        
        return stats


# ==================== PRUEBAS ====================
if __name__ == "__main__":
    print("\n=== PRUEBA DE RECONOCIMIENTO FACIAL ===\n")
    
    recognizer = FaceRecognizer()
    
    # Mostrar estadísticas
    stats = recognizer.get_recognition_stats()
    print("\nEstadísticas:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if stats['total_employees_trained'] == 0:
        print("\n⚠ No hay empleados entrenados.")
        print("Ejecuta el entrenamiento primero con empleados registrados.")
    else:
        print("\n✓ Sistema listo para reconocimiento")
        print("\nIniciando reconocimiento en vivo...")
        print("Presiona 'q' para salir")
        
        import cv2
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Reconocer rostros
            annotated_frame, results = recognizer.recognize_faces_in_frame(frame)
            
            # Mostrar resultados
            for result in results:
                if result['employee_data']:
                    print(f"✓ Reconocido: {result['employee_data']['nombre']} "
                          f"({result['confidence']:.2%})")
            
            cv2.imshow('Reconocimiento Facial', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()