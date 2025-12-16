# src/training.py
"""
Sistema de captura de fotos y entrenamiento de embeddings
"""
import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PHOTOS_PER_EMPLOYEE,
    CAPTURE_INTERVAL,
    create_employee_photo_dir,
    get_employee_photo_dir,
    MIN_PHOTOS_FOR_TRAINING,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)
from src.face_detector import FaceDetector
from src.database import DatabaseManager

class PhotoCaptureSystem:
    """Sistema de captura de fotos para entrenamiento"""

    def __init__(self, detector=None, db=None):
        self.detector = detector or FaceDetector()
        self.db = db or DatabaseManager()
        self.cap = None
    
    def capture_photos_for_employee(
        self,
        employee_id,
        num_photos=PHOTOS_PER_EMPLOYEE,
        camera_index: int | None = None,
    ):
        """
        Capturar fotos de un empleado para entrenamiento
        
        Args:
            employee_id: ID del empleado
            num_photos: Número de fotos a capturar
        
        Returns:
            bool: True si la captura fue exitosa
        """
        # Verificar que el empleado existe
        employee = self.db.get_employee(employee_id)
        if not employee:
            print(f"✗ Error: Empleado {employee_id} no encontrado")
            return False
        
        print(f"\n{'='*60}")
        print(f"CAPTURA DE FOTOS - {employee['nombre']} {employee['apellido']}")
        print(f"{'='*60}")
        print(f"Se capturarán {num_photos} fotos")
        print(f"Instrucciones:")
        print(f"  - Mira a la cámara")
        print(f"  - Mueve ligeramente la cabeza (diferentes ángulos)")
        print(f"  - Haz diferentes expresiones")
        print(f"  - Presiona ESPACIO para iniciar")
        print(f"  - Presiona 'q' para cancelar")
        print(f"{'='*60}\n")
        
        # Crear directorio para fotos
        photo_dir = create_employee_photo_dir(employee_id)
        
        # Calcular el índice inicial para NO sobreescribir fotos previas
        existing_photos = sorted(photo_dir.glob("foto_*.jpg"))
        start_index = len(existing_photos) + 1

        
        # Inicializar cámara
        if camera_index is None:
            camera_index = CAMERA_INDEX

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(
                "✗ Error: No se puede acceder a la cámara seleccionada "
                f"(índice {camera_index})"
            )
            return False
        
        # Configurar cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        captured_photos = 0
        capturing = False
        last_capture_time = 0
        
        print("Esperando para iniciar... (presiona ESPACIO)")
        
        while captured_photos < num_photos:
            ret, frame = self.cap.read()
            if not ret:
                print("✗ Error al capturar frame")
                break
            
            # Detectar rostros
            faces = self.detector.detect_faces(frame)
            
            # Mostrar información en el frame
            display_frame = frame.copy()
            
            if len(faces) == 0:
                # Sin rostros detectados
                cv2.putText(
                    display_frame,
                    "No se detecta rostro - Acercate a la camara",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            elif len(faces) > 1:
                # Múltiples rostros
                cv2.putText(
                    display_frame,
                    "Multiples rostros - Solo debe haber una persona",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            else:
                # Un rostro detectado - OK
                face = faces[0]
                bbox = face.bbox.astype(int)
                
                # Dibujar rectángulo verde
                cv2.rectangle(
                    display_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2
                )
                
                # Si está capturando, guardar foto
                if capturing:
                    current_time = time.time()
                    if current_time - last_capture_time >= CAPTURE_INTERVAL:
                        # Extraer y guardar rostro
                        face_img = self.detector.align_face(frame, face)
                        
                        if face_img.size > 0:
                            photo_index = start_index + captured_photos
                            photo_path = photo_dir / f"foto_{photo_index:03d}.jpg"
                            cv2.imwrite(str(photo_path), face_img)
                            captured_photos += 1
                            last_capture_time = current_time
                            
                            print(
                                f"✓ Foto {captured_photos}/{num_photos} capturada"
                                f"(archivo: {photo_path.name})"
                            )
                
                # Mostrar progreso
                status_text = f"Fotos: {captured_photos}/{num_photos}"
                if not capturing:
                    status_text += " - Presiona ESPACIO para iniciar"
                
                cv2.putText(
                    display_frame,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Barra de progreso
            progress_width = int((captured_photos / num_photos) * (display_frame.shape[1] - 20))
            cv2.rectangle(
                display_frame,
                (10, display_frame.shape[0] - 30),
                (10 + progress_width, display_frame.shape[0] - 10),
                (0, 255, 0),
                -1
            )
            
            # Mostrar frame
            cv2.imshow(f'Captura de Fotos - {employee["nombre"]}', display_frame)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # ESPACIO - iniciar/pausar
                capturing = not capturing
                print("▶ Captura iniciada" if capturing else "⏸ Captura pausada")
            elif key == ord('q'):  # Q - cancelar
                print("\n✗ Captura cancelada por el usuario")
                self.cap.release()
                cv2.destroyAllWindows()
                return False
        
        # Captura completada
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Actualizar base de datos
        self.db.update_employee_photos(employee_id, captured_photos)
        
        print(f"\n{'='*60}")
        print(f"✓ CAPTURA COMPLETADA")
        print(f"  Total de fotos: {captured_photos}")
        print(f"  Guardadas en: {photo_dir}")
        print(f"{'='*60}\n")
        
        return True


class TrainingSystem:
    """Sistema de entrenamiento de embeddings"""

    def __init__(self, detector=None, db=None):
        self.detector = detector or FaceDetector()
        self.db = db or DatabaseManager()
    
    def train_employee(self, employee_id):
        """
        Entrenar embeddings de un empleado
        
        Args:
            employee_id: ID del empleado
        
        Returns:
            list: Lista de embeddings generados
        """
        employee = self.db.get_employee(employee_id)
        if not employee:
            print(f"✗ Error: Empleado {employee_id} no encontrado")
            return None
        
        photo_dir = get_employee_photo_dir(employee_id)
        if not photo_dir.exists():
            print(f"✗ Error: No hay fotos para el empleado {employee_id}")
            return None
        
        # Obtener todas las fotos
        photo_paths = list(photo_dir.glob("*.jpg"))
        
        if len(photo_paths) < MIN_PHOTOS_FOR_TRAINING:
            print(f"✗ Error: Se necesitan al menos {MIN_PHOTOS_FOR_TRAINING} fotos")
            print(f"  Fotos encontradas: {len(photo_paths)}")
            return None
        
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO - {employee['nombre']} {employee['apellido']}")
        print(f"{'='*60}")
        print(f"Procesando {len(photo_paths)} fotos...")
        
        embeddings = []
        
        for photo_path in tqdm(photo_paths, desc="Extrayendo embeddings"):
            # Cargar imagen
            img = cv2.imread(str(photo_path))
            if img is None:
                continue
            
            # Detectar rostro
            faces = self.detector.detect_faces(img)
            
            if len(faces) == 1:
                # Obtener embedding
                embedding = self.detector.get_face_embedding(faces[0])
                embeddings.append(embedding)
            else:
                print(f"  ⚠ Foto omitida (rostros detectados: {len(faces)}): {photo_path.name}")
        
        if len(embeddings) == 0:
            print(f"✗ Error: No se pudieron extraer embeddings")
            return None
        
        print(f"\n✓ Embeddings extraídos: {len(embeddings)}/{len(photo_paths)}")
        print(f"{'='*60}\n")
        
        return embeddings
    
    def train_all_employees(self):
        """
        Entrenar todos los empleados activos
        
        Returns:
            dict: {employee_id: embeddings}
        """
        employees = self.db.get_all_employees(active_only=True)
        
        if not employees:
            print("⚠ No hay empleados registrados")
            return {}
        
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO DE TODOS LOS EMPLEADOS")
        print(f"{'='*60}")
        print(f"Total de empleados: {len(employees)}\n")
        
        all_embeddings = {}
        
        for employee in employees:
            employee_id = employee['id']
            embeddings = self.train_employee(employee_id)
            
            if embeddings:
                all_embeddings[employee_id] = embeddings
        
        print(f"\n{'='*60}")
        print(f"✓ ENTRENAMIENTO COMPLETADO")
        print(f"  Empleados entrenados: {len(all_embeddings)}/{len(employees)}")
        print(f"{'='*60}\n")
        
        return all_embeddings


# ==================== PRUEBAS ====================
if __name__ == "__main__":
    import sys
    
    print("\n=== SISTEMA DE CAPTURA Y ENTRENAMIENTO ===\n")
    
    db = DatabaseManager()
    shared_detector = FaceDetector()
    
    # Verificar si hay empleados
    employees = db.get_all_employees()
    
    if not employees:
        print("No hay empleados registrados.")
        print("Creando empleado de prueba...\n")
        
        emp_id = db.add_employee("Test", "Usuario", "Desarrollador", 25)
        
        if emp_id:
            # Capturar fotos
            capture_system = PhotoCaptureSystem(detector=shared_detector)
            success = capture_system.capture_photos_for_employee(emp_id, num_photos=30)

            if success:
                # Entrenar
                training_system = TrainingSystem(detector=shared_detector)
                embeddings = training_system.train_employee(emp_id)
                
                if embeddings:
                    print(f"\n✓ Sistema probado exitosamente")
                    print(f"  Embeddings generados: {len(embeddings)}")
    else:
        print(f"Empleados registrados: {len(employees)}")
        for emp in employees:
            print(f"  - {emp['nombre']} {emp['apellido']} (ID: {emp['id']}, Fotos: {emp['num_fotos']})")
        
        print("\nOpciones:")
        print("1. Capturar fotos para un empleado")
        print("2. Entrenar un empleado")
        print("3. Entrenar todos los empleados")
        
        choice = input("\nElige una opción (1-3): ").strip()
        
        if choice == "1":
            emp_id = int(input("ID del empleado: "))
            capture_system = PhotoCaptureSystem(detector=shared_detector)
            capture_system.capture_photos_for_employee(emp_id)

        elif choice == "2":
            emp_id = int(input("ID del empleado: "))
            training_system = TrainingSystem(detector=shared_detector)
            training_system.train_employee(emp_id)

        elif choice == "3":
            training_system = TrainingSystem(detector=shared_detector)
            training_system.train_all_employees()
