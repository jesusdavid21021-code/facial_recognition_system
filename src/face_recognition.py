# src/face_recognition.py
"""
Sistema de reconocimiento facial usando embeddings
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
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
        self.embeddings_db = {}  # {employee_id: [embeddings]}
        self.load_embeddings()
    
    def load_embeddings(self):
        """Cargar embeddings guardados"""
        if EMBEDDINGS_PATH.exists():
            with open(EMBEDDINGS_PATH, 'rb') as f:
                self.embeddings_db = pickle.load(f)
            print(f"✓ Embeddings cargados: {len(self.embeddings_db)} empleados")
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
        self.embeddings_db[employee_id] = embeddings
        self.save_embeddings()
        print(f"✓ Embeddings agregados para empleado {employee_id}: {len(embeddings)} fotos")
    
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
            return None, 0, None
        
        best_match_id = None
        best_similarity = 0
        
        # Comparar con cada empleado
        for employee_id, stored_embeddings in self.embeddings_db.items():
            # Calcular similitud con cada embedding del empleado
            similarities = []
            for stored_emb in stored_embeddings:
                # Similitud coseno (entre -1 y 1, normalizado a 0-1)
                sim = cosine_similarity(
                    face_embedding.reshape(1, -1),
                    stored_emb.reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            
            # Tomar el promedio de las mejores similitudes
            top_similarities = sorted(similarities, reverse=True)[:5]
            avg_similarity = np.mean(top_similarities)
            
            # Actualizar mejor match
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match_id = employee_id
        
        # Verificar si supera el umbral
        if best_similarity >= RECOGNITION_THRESHOLD:
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
            'total_embeddings': sum(len(embs) for embs in self.embeddings_db.values()),
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