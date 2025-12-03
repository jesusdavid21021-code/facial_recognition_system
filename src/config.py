# src/config.py
"""
Configuración global del sistema de reconocimiento facial
"""
import json
import os
from pathlib import Path

# ==================== RUTAS DEL PROYECTO ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMPLOYEES_DIR = DATA_DIR / "employees"
DATABASE_DIR = DATA_DIR / "database"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Crear directorios si no existen
for directory in [EMPLOYEES_DIR, DATABASE_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Rutas de archivos específicos
DATABASE_PATH = DATABASE_DIR / "employees.db"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.pkl"
ACCESS_LOG_PATH = LOGS_DIR / "access_logs.csv"
PREFERENCES_PATH = DATA_DIR / "preferences.json"

# ==================== CONFIGURACIÓN DE CÁMARA ====================
CAMERA_INDEX = 0  # 0 para cámara por defecto
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
# Límite superior de índices a probar al enumerar dispositivos
CAMERA_SCAN_MAX_INDEX = 10
# Número de fallos consecutivos antes de detener la enumeración
CAMERA_SCAN_FAIL_STREAK_LIMIT = 3

# ==================== CONFIGURACIÓN DE CAPTURA ====================
PHOTOS_PER_EMPLOYEE = 50  # Número de fotos a capturar por empleado
CAPTURE_INTERVAL = 0.1  # Segundos entre capturas (10 fotos/segundo)
MIN_FACE_SIZE = 80  # Tamaño mínimo del rostro en píxeles

# ==================== CONFIGURACIÓN DE INSIGHTFACE ====================
FACE_DETECTION_MODEL = "buffalo_l"  # Modelo de InsightFace
DETECTION_THRESHOLD = 0.5  # Confianza mínima para detectar un rostro
RECOGNITION_THRESHOLD = 0.6  # Umbral de similitud (más bajo = más estricto)
# Valores recomendados:
# 0.4-0.5: Muy estricto (98-99% precisión, puede rechazar válidos)
# 0.6: Balanceado (95-97% precisión) ⭐ RECOMENDADO
# 0.7-0.8: Permisivo (90-95% precisión, más falsos positivos)

# Providers para ONNX (prioridad: CUDA > CPU)
ONNX_PROVIDERS = ['CPUExecutionProvider']

# ==================== CONFIGURACIÓN DE ENTRENAMIENTO ====================
EMBEDDING_SIZE = 512  # Tamaño del vector de características
MIN_PHOTOS_FOR_TRAINING = 30  # Mínimo de fotos requeridas

# ==================== CONFIGURACIÓN DE INTERFAZ ====================
WINDOW_TITLE = "Sistema de Control de Acceso Facial"
WINDOW_SIZE = "1200x800"
THEME = "dark-blue"  # Tema de CustomTkinter

# Colores de estado
COLOR_SUCCESS = "#2ecc71"  # Verde
COLOR_DANGER = "#e74c3c"   # Rojo
COLOR_WARNING = "#f39c12"  # Amarillo
COLOR_INFO = "#3498db"     # Azul

# ==================== CONFIGURACIÓN DE ACCESO ====================
ACCESS_GRANTED_MESSAGE = "ACCESO PERMITIDO"
ACCESS_DENIED_MESSAGE = "ACCESO DENEGADO"
DISPLAY_TIME = 3  # Segundos para mostrar mensaje
# Período de bloqueo tras un acceso permitido (en horas)
ACCESS_BLOCK_HOURS = 12  # Durante este tiempo no se vuelve a permitir la entrada

# Tiempo mínimo en segundos para considerar que es un nuevo intento de acceso
ACCESS_MIN_REENTRY_SECONDS = 60  # 1 minuto

# Cada cuántos segundos se registra un rostro desconocido para no llenar la base de datos
UNKNOWN_LOG_INTERVAL = 10


# ==================== CONFIGURACIÓN DE LOGS ====================
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_ENCODING = "utf-8"

# ==================== CONFIGURACIÓN DE EXPORTACIÓN ====================
EXPORT_FORMATS = ["CSV", "Excel"]
DEFAULT_EXPORT_FORMAT = "Excel"

# ==================== INFORMACIÓN DEL SISTEMA ====================
APP_NAME = "FaceAccess Pro"
APP_VERSION = "1.0.0"
APP_AUTHOR = "SENA - Proyecto de Reconocimiento Facial"

# ==================== CONFIGURACIÓN DE SEGURIDAD ====================
ENABLE_ANTI_SPOOFING = False  # Detección de fotos/videos (requiere modelo adicional)
MAX_FAILED_ATTEMPTS = 3  # Intentos máximos antes de bloqueo temporal
LOCKOUT_TIME = 60  # Segundos de bloqueo tras fallos

# ==================== FUNCIONES AUXILIARES ====================
def get_employee_photo_dir(employee_id):
    """Retorna el directorio de fotos de un empleado"""
    return EMPLOYEES_DIR / f"empleado_{employee_id:03d}"

def create_employee_photo_dir(employee_id):
    """Crea el directorio de fotos de un empleado"""
    photo_dir = get_employee_photo_dir(employee_id)
    photo_dir.mkdir(parents=True, exist_ok=True)
    return photo_dir

def print_config():
    """Imprime la configuración actual (para debugging)"""
    print("=" * 60)
    print(f"{APP_NAME} v{APP_VERSION}")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Database: {DATABASE_PATH}")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print(f"Access Logs: {ACCESS_LOG_PATH}")
    print(f"Photos per employee: {PHOTOS_PER_EMPLOYEE}")
    print(f"Recognition threshold: {RECOGNITION_THRESHOLD}")
    print(f"ONNX Providers: {ONNX_PROVIDERS}")
    print("=" * 60)


def load_preferences() -> dict:
    """Carga las preferencias desde disco si existen."""

    if not PREFERENCES_PATH.exists():
        return {}

    try:
        with open(PREFERENCES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_preferences(preferences: dict) -> None:
    """Guarda las preferencias en disco (crea el archivo si no existe)."""

    try:
        with open(PREFERENCES_PATH, "w", encoding="utf-8") as f:
            json.dump(preferences, f, ensure_ascii=False, indent=2)
    except Exception:
        # Preferimos no romper el flujo si no se puede guardar
        pass


def get_last_camera_index(default: int = CAMERA_INDEX) -> int:
    """Devuelve el índice de cámara guardado o el valor por defecto."""

    prefs = load_preferences()
    try:
        return int(prefs.get("last_camera_index", default))
    except Exception:
        return default


def set_last_camera_index(index: int) -> None:
    """Actualiza el índice de cámara en las preferencias."""

    prefs = load_preferences()
    prefs["last_camera_index"] = int(index)
    save_preferences(prefs)


if __name__ == "__main__":
    print_config()
