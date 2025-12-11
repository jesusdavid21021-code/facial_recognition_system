# src/database.py
"""
Gestión de base de datos SQLite para empleados y registros de acceso
"""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from src.config import DATABASE_PATH, LOG_DATE_FORMAT, get_closing_time

class DatabaseManager:
    """Gestor de base de datos para el sistema de reconocimiento facial"""
    
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Crear conexión a la base de datos"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Permite acceso por nombre de columna
        return conn
    
    def init_database(self):
        """Inicializar tablas de la base de datos"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Tabla de empleados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                apellido TEXT NOT NULL,
                cargo TEXT NOT NULL,
                edad INTEGER NOT NULL,
                fecha_registro TEXT NOT NULL,
                num_fotos INTEGER DEFAULT 0,
                activo BOOLEAN DEFAULT 1,
                UNIQUE(nombre, apellido)
            )
        ''')
        
        # Tabla de registros de acceso
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER,
                fecha_hora TEXT NOT NULL,
                tipo_acceso TEXT NOT NULL,
                confianza REAL,
                FOREIGN KEY (employee_id) REFERENCES employees (id)
            )
        ''')
        
        # Índices para búsquedas rápidas
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_employee_activo 
            ON employees(activo)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_access_date 
            ON access_logs(fecha_hora)
        ''')
        
        conn.commit()
        conn.close()
        print(f"✓ Base de datos inicializada: {self.db_path}")
    
    # ==================== OPERACIONES CON EMPLEADOS ====================
    
    def add_employee(self, nombre, apellido, cargo, edad):
        """Agregar nuevo empleado"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            fecha_registro = datetime.now().strftime(LOG_DATE_FORMAT)
            cursor.execute('''
                INSERT INTO employees (nombre, apellido, cargo, edad, fecha_registro)
                VALUES (?, ?, ?, ?, ?)
            ''', (nombre, apellido, cargo, edad, fecha_registro))
            
            employee_id = cursor.lastrowid
            conn.commit()
            print(f"✓ Empleado agregado: {nombre} {apellido} (ID: {employee_id})")
            return employee_id
        
        except sqlite3.IntegrityError:
            print(f"✗ Error: Empleado {nombre} {apellido} ya existe")
            return None
        
        finally:
            conn.close()
    
    def get_employee(self, employee_id):
        """Obtener datos de un empleado por ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM employees WHERE id = ?', (employee_id,))
        employee = cursor.fetchone()
        conn.close()
        
        return dict(employee) if employee else None
    
    def get_all_employees(self, active_only=True):
        """Obtener todos los empleados"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute('SELECT * FROM employees WHERE activo = 1 ORDER BY apellido, nombre')
        else:
            cursor.execute('SELECT * FROM employees ORDER BY apellido, nombre')
        
        employees = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return employees
    
    def update_employee_photos(self, employee_id, num_fotos):
        """Actualizar número de fotos de un empleado"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE employees 
            SET num_fotos = ? 
            WHERE id = ?
        ''', (num_fotos, employee_id))
        
        conn.commit()
        conn.close()
    
    def delete_employee(self, employee_id):
        """Eliminar (desactivar) un empleado"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE employees 
            SET activo = 0 
            WHERE id = ?
        ''', (employee_id,))
        
        conn.commit()
        conn.close()
        print(f"✓ Empleado {employee_id} desactivado")
    
    def permanently_delete_employee(self, employee_id):
        """Eliminar permanentemente un empleado y sus registros"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Eliminar registros de acceso
        cursor.execute('DELETE FROM access_logs WHERE employee_id = ?', (employee_id,))
        
        # Eliminar empleado
        cursor.execute('DELETE FROM employees WHERE id = ?', (employee_id,))
        
        conn.commit()
        conn.close()
        print(f"✓ Empleado {employee_id} eliminado permanentemente")
    
    # ==================== OPERACIONES CON REGISTROS DE ACCESO ====================
    
    def log_access(self, employee_id, tipo_acceso, confianza=None):
        """Registrar un acceso (permitido, denegado o desconocido)."""
        # Normalizar confianza a float (evitar np.float32, bytes, etc.)
        if confianza is not None:
            try:
                confianza = float(confianza)
            except (TypeError, ValueError):
                # Si no se puede convertir, la guardamos como None
                confianza = None

        conn = self.get_connection()
        cursor = conn.cursor()

        fecha_hora = datetime.now().strftime(LOG_DATE_FORMAT)

        cursor.execute(
            """
            INSERT INTO access_logs (employee_id, fecha_hora, tipo_acceso, confianza)
            VALUES (?, ?, ?, ?)
            """,
            (employee_id, fecha_hora, tipo_acceso, confianza),
        )

        conn.commit()
        conn.close()


    def get_recent_access_logs(self, limit=100):
        """Devuelve los últimos 'limit' registros de acceso con datos del empleado."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT al.id,
                   al.employee_id,
                   al.fecha_hora,
                   al.tipo_acceso,
                   al.confianza,
                   e.nombre,
                   e.apellido,
                   e.cargo
            FROM access_logs al
            LEFT JOIN employees e ON al.employee_id = e.id
            ORDER BY al.fecha_hora DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_last_access_for_employee(self, employee_id):
        """Devuelve el último acceso PERMITIDO de un empleado o None si no existe."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, employee_id, fecha_hora, tipo_acceso, confianza
            FROM access_logs
            WHERE employee_id = ? 
                AND (
                    tipo_acceso = 'permitido'
                    OR tipo_acceso = 'entrada'
                    OR tipo_acceso = 'salida'
                  )
            ORDER BY datetime(fecha_hora) DESC
            LIMIT 1
            """,
            (employee_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return dict(row)

    def get_access_logs(self, employee_id=None, start_date=None, end_date=None):
        """
        Devuelve registros de acceso con filtros opcionales:
        - employee_id: solo ese empleado
        - start_date / end_date: cadenas 'YYYY-MM-DD'
        """
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT al.id,
                   al.employee_id,
                   al.fecha_hora,
                   al.tipo_acceso,
                   al.confianza,
                   e.nombre,
                   e.apellido,
                   e.cargo
            FROM access_logs al
            LEFT JOIN employees e ON al.employee_id = e.id
            WHERE 1 = 1
        """
        params = []

        if employee_id is not None:
            query += " AND al.employee_id = ?"
            params.append(employee_id)

        if start_date is not None:
            # Se asume formato 'YYYY-MM-DD'
            query += " AND date(al.fecha_hora) >= date(?)"
            params.append(start_date)

        if end_date is not None:
            query += " AND date(al.fecha_hora) <= date(?)"
            params.append(end_date)

        query += " ORDER BY al.fecha_hora DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_statistics(self):
        """
        Devuelve un dict con estadísticas generales:
        - total_empleados
        - accesos_hoy
        - accesos_permitidos
        - accesos_denegados
        - accesos_desconocidos
        """
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Empleados activos
        cursor.execute("SELECT COUNT(*) AS c FROM employees WHERE activo = 1")
        total_empleados = cursor.fetchone()["c"]

        # Accesos hoy
        hoy = datetime.now().strftime("%Y-%m-%d")
        cursor.execute(
            """
            SELECT COUNT(*) AS c
            FROM access_logs
            WHERE date(fecha_hora) = date(?)
            """,
            (hoy,),
        )
        accesos_hoy = cursor.fetchone()["c"]

        # Detalle por tipo
        cursor.execute(
            """
            SELECT tipo_acceso, COUNT(*) AS c
            FROM access_logs
            GROUP BY tipo_acceso
            """
        )
        rows = cursor.fetchall()
        conn.close()

        accesos_permitidos = 0
        accesos_denegados = 0
        accesos_desconocidos = 0

        for r in rows:
            t = r["tipo_acceso"]
            c = r["c"]
            #Entrada x salida ---NO TOCAR
            if t in ("permitido", "entrada", "salida"):
                accesos_permitidos += c
            elif t == "denegado":
                accesos_denegados += c
            elif t == "desconocido":
                accesos_desconocidos += c

        return {
            "total_empleados": total_empleados,
            "accesos_hoy": accesos_hoy,
            "accesos_permitidos": accesos_permitidos,
            "accesos_denegados": accesos_denegados,
            "accesos_desconocidos": accesos_desconocidos,
        }

    def export_access_logs_to_csv(self, filename, start_date=None, end_date=None):
        """
        Exporta los registros de acceso a un CSV en disco.
        Este método NO lo usa la GUI actual (que exporta lo que hay en la tabla),
        pero queda disponible por si quieres sacar reportes desde código.
        """
        import csv

        logs = self.get_access_logs(
            employee_id=None,
            start_date=start_date,
            end_date=end_date,
        )

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                ["id", "fecha_hora", "employee_id", "nombre", "apellido",
                 "tipo_acceso", "confianza"]
            )
            for log in logs:
                writer.writerow(
                    [
                        log.get("id"),
                        log.get("fecha_hora"),
                        log.get("employee_id"),
                        log.get("nombre"),
                        log.get("apellido"),
                        log.get("tipo_acceso"),
                        log.get("confianza"),
                    ]
                )

    def _get_default_report_date(self) -> str:
        """
        Devuelve la fecha (YYYY-MM-DD) que se usará para el informe de cierre.

        - Si el informe se genera DESPUÉS de la hora de cierre -> usa hoy.
        - Si se genera ANTES de la hora de cierre (por la mañana) -> usa ayer.
        """
        now = datetime.now()
        
        try:
            closing_str = get_closing_time()
            hh, mm = map(int, closing_str.split(":"))
            closing_today = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except Exception:
            # Fallback: cierre casi a medianoche
            closing_today = now.replace(hour=23, minute=59, second=0, microsecond=0)

        if now >= closing_today:
            report_date = now.date()
        else:
            report_date = now.date() - timedelta(days=1)

        return report_date.strftime("%Y-%m-%d")

    def get_employees_with_open_entry(self, date_str: str | None = None):
        """
        Devuelve una lista de empleados cuya ÚLTIMA marca del día es ENTRADA,
        es decir, no registraron SALIDA en esa jornada.

        - date_str: 'YYYY-MM-DD'. Si es None, se usa la fecha calculada
          por _get_default_report_date() según la hora de cierre.
        """
        if date_str is None:
            date_str = self._get_default_report_date()
            
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT al.id,
                al.employee_id,
                al.fecha_hora,
                al.tipo_acceso,
                al.confianza,
                e.nombre,
                e.apellido,
                e.cargo
            FROM access_logs al
            JOIN (
                -- Último registro (entrada/salida) por empleado en esa fecha
                SELECT employee_id, MAX(fecha_hora) AS max_fecha
                FROM access_logs
                WHERE date(fecha_hora) = ?
                  AND tipo_acceso IN ('entrada', 'salida')
                GROUP BY employee_id
            ) t
              ON al.employee_id = t.employee_id
             AND al.fecha_hora = t.max_fecha
            JOIN employees e ON e.id = al.employee_id
            WHERE al.tipo_acceso = 'entrada'
            ORDER BY e.apellido, e.nombre
            """,
            (date_str,),
        )
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def export_open_entries_report(self, filename: str, date_str: str | None = None):
        """
        Exporta a CSV el informe de empleados que no registraron salida.
        """
        import csv
        
        rows = self.get_employees_with_open_entry(date_str)
        
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    "employee_id",
                    "nombre",
                    "apellido",
                    "cargo",
                    "fecha_ultima_entrada",
                    "confianza",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r.get("employee_id"),
                        r.get("nombre"),
                        r.get("apellido"),
                        r.get("cargo"),
                        r.get("fecha_hora"),
                        f"{r.get('confianza', 0):.4f}",
                    ]
                )
                
        