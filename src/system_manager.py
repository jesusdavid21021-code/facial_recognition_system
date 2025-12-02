# src/system_manager.py
"""
Gestor principal del sistema - Integra todos los módulos
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import DatabaseManager
from src.face_recognition import FaceRecognizer
from src.training import PhotoCaptureSystem, TrainingSystem

class FacialRecognitionSystem:
    """Gestor principal del sistema de reconocimiento facial"""
    
    def __init__(self):
        print("Inicializando sistema...")
        self.db = DatabaseManager()
        self.recognizer = FaceRecognizer()
        self.capture_system = PhotoCaptureSystem()
        self.training_system = TrainingSystem()
        print("✓ Sistema inicializado\n")
    
    def register_and_train_employee(self, nombre, apellido, cargo, edad, num_photos=50):
        """
        Registrar un nuevo empleado y entrenar su reconocimiento
        
        Args:
            nombre: Nombre del empleado
            apellido: Apellido del empleado
            cargo: Cargo del empleado
            edad: Edad del empleado
            num_photos: Número de fotos a capturar (default: 50)
        
        Returns:
            bool: True si el proceso fue exitoso
        """
        print(f"\n{'='*60}")
        print(f"REGISTRO Y ENTRENAMIENTO DE NUEVO EMPLEADO")
        print(f"{'='*60}\n")
        
        # 1. Registrar en base de datos
        print("Paso 1: Registrando en base de datos...")
        employee_id = self.db.add_employee(nombre, apellido, cargo, edad)
        
        if not employee_id:
            print("✗ Error al registrar empleado")
            return False
        
        print(f"✓ Empleado registrado con ID: {employee_id}\n")
        
        # 2. Capturar fotos
        print("Paso 2: Capturando fotos...")
        success = self.capture_system.capture_photos_for_employee(employee_id, num_photos)
        
        if not success:
            print("✗ Error en captura de fotos")
            self.db.delete_employee(employee_id)
            return False
        
        # 3. Entrenar embeddings
        print("\nPaso 3: Entrenando modelo de reconocimiento...")
        embeddings = self.training_system.train_employee(employee_id)
        
        if not embeddings:
            print("✗ Error al entrenar embeddings")
            return False
        
        # 4. Guardar embeddings
        print("Paso 4: Guardando embeddings...")
        self.recognizer.add_employee_embeddings(employee_id, embeddings)
        
        print(f"\n{'='*60}")
        print(f"✓ PROCESO COMPLETADO EXITOSAMENTE")
        print(f"  Empleado: {nombre} {apellido}")
        print(f"  ID: {employee_id}")
        print(f"  Fotos capturadas: {len(embeddings)}")
        print(f"  Estado: Listo para reconocimiento")
        print(f"{'='*60}\n")
        
        return True
    
    def retrain_employee(self, employee_id, num_photos=50):
        """
        Re-entrenar un empleado existente
        
        Args:
            employee_id: ID del empleado
            num_photos: Número de fotos nuevas a capturar
        
        Returns:
            bool: True si el proceso fue exitoso
        """
        employee = self.db.get_employee(employee_id)
        if not employee:
            print(f"✗ Error: Empleado {employee_id} no encontrado")
            return False
        
        print(f"\n{'='*60}")
        print(f"RE-ENTRENAMIENTO - {employee['nombre']} {employee['apellido']}")
        print(f"{'='*60}\n")
        
        # Capturar nuevas fotos
        print("Capturando nuevas fotos...")
        success = self.capture_system.capture_photos_for_employee(employee_id, num_photos)
        
        if not success:
            return False
        
        # Re-entrenar
        print("\nRe-entrenando modelo...")
        embeddings = self.training_system.train_employee(employee_id)
        
        if not embeddings:
            return False
        
        # Actualizar embeddings
        self.recognizer.add_employee_embeddings(employee_id, embeddings)
        
        print(f"\n✓ Re-entrenamiento completado\n")
        return True
    
    def train_all_pending(self):
        """
        Entrenar todos los empleados que tienen fotos pero no embeddings
        """
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO MASIVO")
        print(f"{'='*60}\n")
        
        employees = self.db.get_all_employees(active_only=True)
        trained_count = 0
        
        for employee in employees:
            employee_id = employee['id']
            
            # Verificar si ya está entrenado
            if employee_id in self.recognizer.embeddings_db:
                print(f"⏭ {employee['nombre']} {employee['apellido']} - Ya entrenado")
                continue
            
            # Verificar si tiene fotos
            if employee['num_fotos'] == 0:
                print(f"⏭ {employee['nombre']} {employee['apellido']} - Sin fotos")
                continue
            
            # Entrenar
            print(f"\nEntrenando: {employee['nombre']} {employee['apellido']}...")
            embeddings = self.training_system.train_employee(employee_id)
            
            if embeddings:
                self.recognizer.add_employee_embeddings(employee_id, embeddings)
                trained_count += 1
                print(f"✓ Completado")
        
        print(f"\n{'='*60}")
        print(f"✓ ENTRENAMIENTO MASIVO COMPLETADO")
        print(f"  Empleados entrenados: {trained_count}")
        print(f"{'='*60}\n")
    
    
    def delete_employee_gui(self, employee_id):
        """Eliminar un empleado desde la interfaz gráfica (sin pedir CONFIRMAR por consola)."""
        employee = self.db.get_employee(employee_id)
        if not employee:
            return False

        # Eliminar embeddings
        self.recognizer.remove_employee_embeddings(employee_id)

        # Eliminar de base de datos
        self.db.permanently_delete_employee(employee_id)

        # Eliminar fotos del disco
        from src.config import get_employee_photo_dir
        import shutil
        photo_dir = get_employee_photo_dir(employee_id)
        if photo_dir.exists():
            shutil.rmtree(photo_dir)

        return True

    def delete_employee(self, employee_id):
        """
        Eliminar un empleado completamente
        
        Args:
            employee_id: ID del empleado
        """
        employee = self.db.get_employee(employee_id)
        if not employee:
            print(f"✗ Error: Empleado {employee_id} no encontrado")
            return False
        
        print(f"\n¿Eliminar empleado {employee['nombre']} {employee['apellido']}?")
        confirm = input("Escriba 'CONFIRMAR' para eliminar: ").strip()
        
        if confirm != "CONFIRMAR":
            print("✗ Eliminación cancelada")
            return False
        
        # Eliminar embeddings
        self.recognizer.remove_employee_embeddings(employee_id)
        
        # Eliminar de base de datos
        self.db.permanently_delete_employee(employee_id)
        
        # Eliminar fotos
        from src.config import get_employee_photo_dir
        import shutil
        photo_dir = get_employee_photo_dir(employee_id)
        if photo_dir.exists():
            shutil.rmtree(photo_dir)
        
        print(f"✓ Empleado eliminado completamente\n")
        return True
    
    def show_system_status(self):
        """Mostrar estado general del sistema"""
        print(f"\n{'='*60}")
        print(f"ESTADO DEL SISTEMA")
        print(f"{'='*60}\n")
        
        # Estadísticas de base de datos
        db_stats = self.db.get_statistics()
        print("Base de Datos:")
        print(f"  - Empleados activos: {db_stats.get('total_empleados', 0)}")
        print(f"  - Accesos hoy: {db_stats.get('accesos_hoy', 0)}")
        print(f"  - Accesos permitidos: {db_stats.get('total_permitido', 0)}")
        print(f"  - Accesos denegados: {db_stats.get('total_denegado', 0)}")
        
        # Estadísticas de reconocimiento
        rec_stats = self.recognizer.get_recognition_stats()
        print(f"\nSistema de Reconocimiento:")
        print(f"  - Empleados entrenados: {rec_stats['total_employees_trained']}")
        print(f"  - Total embeddings: {rec_stats['total_embeddings']}")
        print(f"  - Promedio por empleado: {rec_stats['avg_embeddings_per_employee']:.1f}")
        print(f"  - Umbral de reconocimiento: {rec_stats['threshold']:.2f}")
        
        # Listar empleados
        employees = self.db.get_all_employees(active_only=True)
        print(f"\nEmpleados Registrados ({len(employees)}):")
        for emp in employees:
            trained = "✓" if emp['id'] in self.recognizer.embeddings_db else "✗"
            print(f"  {trained} ID:{emp['id']} - {emp['nombre']} {emp['apellido']} "
                  f"({emp['cargo']}) - {emp['num_fotos']} fotos")
        
        print(f"\n{'='*60}\n")


# ==================== MENÚ INTERACTIVO ====================
def main_menu():
    """Menú principal del sistema"""
    system = FacialRecognitionSystem()
    
    while True:
        print("\n" + "="*60)
        print("SISTEMA DE RECONOCIMIENTO FACIAL - MENÚ PRINCIPAL")
        print("="*60)
        print("\n1. Registrar nuevo empleado")
        print("2. Ver estado del sistema")
        print("3. Re-entrenar empleado existente")
        print("4. Entrenar todos los pendientes")
        print("5. Eliminar empleado")
        print("6. Probar reconocimiento en vivo")
        print("7. Ver registros de acceso")
        print("0. Salir")
        
        choice = input("\nSeleccione una opción: ").strip()
        
        if choice == "1":
            print("\n--- REGISTRO DE NUEVO EMPLEADO ---")
            nombre = input("Nombre: ").strip()
            apellido = input("Apellido: ").strip()
            cargo = input("Cargo: ").strip()
            edad = int(input("Edad: ").strip())
            
            system.register_and_train_employee(nombre, apellido, cargo, edad)
        
        elif choice == "2":
            system.show_system_status()
        
        elif choice == "3":
            emp_id = int(input("\nID del empleado: ").strip())
            system.retrain_employee(emp_id)
        
        elif choice == "4":
            system.train_all_pending()
        
        elif choice == "5":
            emp_id = int(input("\nID del empleado: ").strip())
            system.delete_employee(emp_id)
        
        elif choice == "6":
            print("\n--- RECONOCIMIENTO EN VIVO ---")
            print("Presiona 'q' para salir\n")
            
            import cv2
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated_frame, results = system.recognizer.recognize_faces_in_frame(frame)
                
                # Registrar accesos
                for result in results:
                    if result['employee_data']:
                        system.db.log_access(
                            result['employee_id'],
                            'permitido',
                            result['confidence']
                        )
                
                cv2.imshow('Reconocimiento Facial - Presiona Q para salir', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == "7":
            logs = system.db.get_access_logs()
            print(f"\n--- ÚLTIMOS {min(10, len(logs))} REGISTROS ---")
            for log in logs[:10]:
                nombre = log.get('nombre', 'Desconocido')
                apellido = log.get('apellido', '')
                print(f"{log['fecha_hora']} - {nombre} {apellido} - "
                      f"{log['tipo_acceso']} ({log['confianza']:.2%})")
        
        elif choice == "0":
            print("\n¡Hasta luego!")
            break
        
        else:
            print("\n✗ Opción inválida")


if __name__ == "__main__":
    main_menu()