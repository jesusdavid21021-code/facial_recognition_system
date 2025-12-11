# src/gui_app.py
"""
Interfaz gráfica para el sistema de reconocimiento facial de empleados.
Usa CustomTkinter y se apoya en FacialRecognitionSystem.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import cv2
import time
from datetime import datetime
import warnings
import re

from src.system_manager import FacialRecognitionSystem
from src.config import (
    WINDOW_TITLE,
    WINDOW_SIZE,
    THEME,
    RECOGNITION_THRESHOLD,
    CAMERA_INDEX,
    CAMERA_SCAN_FAIL_STREAK_LIMIT,
    CAMERA_SCAN_MAX_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    ACCESS_BLOCK_HOURS,
    ACCESS_MIN_REENTRY_SECONDS,
    UNKNOWN_LOG_INTERVAL,
    COLOR_SUCCESS,
    COLOR_DANGER,
    COLOR_WARNING,
    get_last_camera_index,
    set_last_camera_index,
    get_closing_time,
    set_closing_time,
)

# Silenciar algunos FutureWarning de insightface/numpy
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="insightface.utils.transform",
)


class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración básica de la ventana
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme(THEME)
        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_SIZE)

        # Sistema principal (DB + detector + recognizer + training)
        self.system = FacialRecognitionSystem()
        self.db = self.system.db

        # Widgets
        self.nombre_entry = None
        self.apellido_entry = None
        self.cargo_entry = None
        self.edad_entry = None
        self.employees_tree = None
        self.log_text = None
        self.access_status_label = None
        self.stats_label = None
        self.camera_combo = None

        # Estado interno para mensajes de acceso y desconocidos
        self.access_status_after_id = None
        self.last_unknown_log_ts = 0.0
        self.selected_camera_index = get_last_camera_index(default=CAMERA_INDEX)
        self._camera_label_to_index = {}

        # Construir UI
        self._build_ui()

        # Cargar empleados
        self.refresh_employees()
        # Cargar estadísticas iniciales
        self._refresh_access_stats()
        self.closing_time_var = tk.StringVar(value=get_closing_time())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _on_save_closing_time(self):
        """
        Guarda la hora de cierre en preferencias (formato HH:MM, 24h).
        """
        value = self.closing_time_var.get().strip()
        
        # Validar formato básico HH:MM
        if not re.match(r"^\d{2}:\d{2}$", value):
            messagebox.showerror(
                "Hora inválida",
                "Por favor ingresa la hora en formato HH:MM, por ejemplo 18:30."
            )
            return
        
        try:
            hh, mm = map(int, value.split(":"))
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Hora inválida",
                "La hora debe estar entre 00:00 y 23:59."
            )
            return
        
        # Guardar en preferencias
        set_closing_time(value)
        
        messagebox.showinfo(
            "Hora de cierre guardada",
            f"La hora de cierre ha sido establecida en {value}."
        )
    
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # Panel izquierdo: formulario de empleado
        form_frame = ctk.CTkFrame(self)
        form_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        form_frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            form_frame,
            text="Registro de empleado",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(10, 20))

        # Nombre
        ctk.CTkLabel(form_frame, text="Nombre:").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        self.nombre_entry = ctk.CTkEntry(form_frame)
        self.nombre_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Apellido
        ctk.CTkLabel(form_frame, text="Apellido:").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        self.apellido_entry = ctk.CTkEntry(form_frame)
        self.apellido_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        # Cargo
        ctk.CTkLabel(form_frame, text="Cargo:").grid(
            row=3, column=0, sticky="e", padx=5, pady=5
        )
        self.cargo_entry = ctk.CTkEntry(form_frame)
        self.cargo_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        # Edad
        ctk.CTkLabel(form_frame, text="Edad:").grid(
            row=4, column=0, sticky="e", padx=5, pady=5
        )
        self.edad_entry = ctk.CTkEntry(form_frame)
        self.edad_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        # Botón registrar + entrenar
        register_btn = ctk.CTkButton(
            form_frame,
            text="Registrar y entrenar",
            command=self.on_register_employee,
        )
        register_btn.grid(row=5, column=0, columnspan=2, pady=(15, 10))

        # Botón reconocimiento en vivo
        live_btn = ctk.CTkButton(
            form_frame,
            text="Reconocimiento en vivo",
            command=self.on_live_recognition,
        )
        live_btn.grid(row=6, column=0, columnspan=2, pady=(5, 10))

        # Panel derecho: lista de empleados + acciones
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=0)
        right_frame.grid_columnconfigure(0, weight=1)

        employees_label = ctk.CTkLabel(
            right_frame,
            text="Empleados",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        employees_label.grid(
            row=0, column=0, sticky="w", pady=(10, 5), padx=10
        )

        # Treeview de empleados
        cols = ("id", "nombre", "apellido", "cargo", "edad", "activo", "fotos")
        self.employees_tree = ttk.Treeview(
            right_frame, columns=cols, show="headings", height=10
        )
        for col in cols:
            self.employees_tree.heading(col, text=col.capitalize())
            self.employees_tree.column(col, width=80, anchor="center")

        self.employees_tree.grid(
            row=1, column=0, sticky="nsew", padx=10, pady=5
        )

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            right_frame, orient="vertical", command=self.employees_tree.yview
        )
        self.employees_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=5)

        # Botones de acciones sobre empleados
        actions_frame = ctk.CTkFrame(right_frame)
        actions_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        actions_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        retrain_btn = ctk.CTkButton(
            actions_frame,
            text="Re-entrenar",
            command=self.on_retrain_employee,
        )
        retrain_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        delete_btn = ctk.CTkButton(
            actions_frame,
            text="Eliminar",
            fg_color=COLOR_DANGER,
            hover_color="#c0392b",
            command=self.on_delete_employee,
        )
        delete_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        refresh_btn = ctk.CTkButton(
            actions_frame,
            text="Actualizar lista",
            command=self.refresh_employees,
        )
        refresh_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        logs_btn = ctk.CTkButton(
            actions_frame,
            text="Ver accesos",
            command=self.on_show_access_logs,
        )
        logs_btn.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        camera_frame = ctk.CTkFrame(actions_frame)
        camera_frame.grid(
            row=1, column=0, columnspan=4, padx=5, pady=(10, 5), sticky="ew"
        )
        camera_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(camera_frame, text="Cámara de captura:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )

        self.camera_combo = ctk.CTkComboBox(
            camera_frame,
            values=[],
            width=260,
            command=self._on_camera_selected,
        )
        self.camera_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        refresh_cameras_btn = ctk.CTkButton(
            camera_frame,
            text="Detectar cámaras",
            width=140,
            command=self._refresh_camera_combo,
        )
        refresh_cameras_btn.grid(row=0, column=2, padx=5, pady=5)

        # Panel inferior: log de eventos + cartel de acceso + stats
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=10,
            pady=(0, 10),
            sticky="nsew",
        )
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            log_frame,
            text="Eventos / mensajes",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        # Cartel de estado de acceso
        self.access_status_label = ctk.CTkLabel(
            log_frame,
            text="Sin eventos de acceso",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white",
        )
        self.access_status_label.grid(
            row=0, column=1, sticky="e", padx=10, pady=5
        )

        self.log_text = ctk.CTkTextbox(log_frame, height=100)
        self.log_text.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="nsew",
            padx=10,
            pady=(0, 10),
        )
        self.log_text.configure(state="disabled")

        # Resumen de estadísticas
        self.stats_label = ctk.CTkLabel(
            log_frame,
            text="Estadísticas: (sin datos aún)",
            font=ctk.CTkFont(size=12),
            text_color="white",
        )
        self.stats_label.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=(0, 8),
        )

        self._refresh_camera_combo()

    # ------------------------------------------------------------------
    # Cámaras
    # ------------------------------------------------------------------
    def _format_camera_label(self, index: int, width: int | None, height: int | None) -> str:
        size = (
            f"{width}x{height}"
            if width is not None and height is not None
            else "Resolución desconocida"
        )
        return f"{index} - {size}"

    def _enumerate_cameras(
        self,
        max_index: int = CAMERA_SCAN_MAX_INDEX,
        fail_streak_limit: int = CAMERA_SCAN_FAIL_STREAK_LIMIT,
    ):
        """Recorre los índices de cámara con un tope de fallos consecutivos.

        Algunos drivers (p. ej. cámaras Intel RealSense) imprimen errores a stderr
        cuando se consulta un índice inexistente. Para evitar el spam, cortamos el
        escaneo cuando acumulamos `fail_streak_limit` fallos seguidos después de
        haber encontrado al menos una cámara válida.
        """

        cameras = []
        consecutive_failures = 0

        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                consecutive_failures += 1

                if cameras and consecutive_failures >= fail_streak_limit:
                    break

                continue

            consecutive_failures = 0

            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()

            if width > 0 and height > 0:
                cameras.append((idx, int(width), int(height)))

        return cameras

    def _refresh_camera_combo(self):
        """Detecta cámaras disponibles y actualiza el combo."""

        cameras = self._enumerate_cameras()

        if not cameras:
            cameras = [(self.selected_camera_index, None, None)]

        self._camera_label_to_index = {}
        values = []
        for idx, width, height in cameras:
            label = self._format_camera_label(idx, width, height)
            self._camera_label_to_index[label] = idx
            values.append(label)

        self.camera_combo.configure(values=values)

        selected_label = next(
            (label for label, idx in self._camera_label_to_index.items() if idx == self.selected_camera_index),
            values[0],
        )

        self.selected_camera_index = self._camera_label_to_index.get(
            selected_label, self.selected_camera_index
        )
        self.camera_combo.set(selected_label)
        set_last_camera_index(self.selected_camera_index)

    def _on_camera_selected(self, choice: str):
        if not choice:
            return

        idx = self._camera_label_to_index.get(choice)

        if idx is None:
            try:
                idx = int(choice.split("-", 1)[0].strip())
            except (ValueError, AttributeError):
                return

        self.selected_camera_index = idx
        set_last_camera_index(idx)

    # ------------------------------------------------------------------
    # Utilidades de log, estadísticas y cartel
    # ------------------------------------------------------------------
    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _refresh_access_stats(self):
        """Actualiza el resumen de estadísticas en la parte inferior."""
        try:
            stats = self.db.get_statistics()
        except Exception as e:
            self.stats_label.configure(
                text=f"Error obteniendo estadísticas: {e}"
            )
            return

        total_emp = stats.get("total_empleados", 0)
        accesos_hoy = stats.get("accesos_hoy", 0)
        permitidos = stats.get("accesos_permitidos", 0)
        denegados = stats.get("accesos_denegados", 0)
        desconocidos = stats.get("accesos_desconocidos", 0)

        text = (
            f"Empleados activos: {total_emp}  |  "
            f"Accesos hoy: {accesos_hoy}  |  "
            f"Permitidos: {permitidos}  |  "
            f"Denegados: {denegados}  |  "
            f"Desconocidos: {desconocidos}"
        )
        self.stats_label.configure(text=text)

    def _show_access_status(self, status: str, emp_data=None):
        """Actualiza el cartel [ENTRADA/SALIDA/DENEGADO/DESCONOCIDO]."""

        # Cancelar reseteo previo si lo hay
        if self.access_status_after_id is not None:
            try:
                self.after_cancel(self.access_status_after_id)
            except Exception:
                pass
            self.access_status_after_id = None

        if status in ("permitido", "entrada", "salida"):
            nombre = (
                f"{emp_data['nombre']} {emp_data['apellido']}"
                if emp_data
                else ""
            )
            
            if status == "entrada":
                prefix = "[ENTRADA]"
            elif status == "salida":
                prefix = "[SALIDA]"
            else:
                prefix = "[ACCESO PERMITIDO]"
                
            text = f"{prefix} {nombre}".strip()
            color = COLOR_SUCCESS
            
        elif status == "denegado":
            nombre = (
                f"{emp_data['nombre']} {emp_data['apellido']}"
                if emp_data
                else ""
            )
            text = f"[ACCESO DENEGADO] {nombre}".strip()
            color = COLOR_DANGER
            
        elif status == "desconocido":
            text = "[ACCESO DESCONOCIDO]"
            color = COLOR_WARNING
            
        else:
            text = status
            color = "white"
            
        self.access_status_label.configure(text=text, text_color=color)

        # Volver al estado neutro después de 5 segundos
        self.access_status_after_id = self.after(
            5000,
            lambda: self.access_status_label.configure(
                text="Sin eventos de acceso", text_color="white"
            ),
        )

    # ------------------------------------------------------------------
    # Lógica de empleados
    # ------------------------------------------------------------------
    def refresh_employees(self):
        # Limpiar tabla
        for item in self.employees_tree.get_children():
            self.employees_tree.delete(item)

        employees = self.db.get_all_employees(active_only=False)

        for emp in employees:
            self.employees_tree.insert(
                "",
                "end",
                values=(
                    emp["id"],
                    emp["nombre"],
                    emp["apellido"],
                    emp["cargo"],
                    emp["edad"],
                    "Sí" if emp["activo"] else "No",
                    emp.get("num_fotos", 0),
                ),
            )

        self._append_log(f"✓ Empleados cargados: {len(employees)}")

    def _get_selected_employee_id(self):
        sel = self.employees_tree.selection()
        if not sel:
            messagebox.showwarning(
                "Atención", "Selecciona un empleado en la tabla."
            )
            return None

        values = self.employees_tree.item(sel[0], "values")
        return int(values[0])

    def on_register_employee(self):
        nombre = self.nombre_entry.get().strip()
        apellido = self.apellido_entry.get().strip()
        cargo = self.cargo_entry.get().strip()
        edad_str = self.edad_entry.get().strip()

        if not nombre or not apellido or not cargo or not edad_str:
            messagebox.showerror("Error", "Todos los campos son obligatorios.")
            return

        try:
            edad = int(edad_str)
        except ValueError:
            messagebox.showerror(
                "Error", "La edad debe ser un número entero."
            )
            return

        self._append_log(f"▶ Registrando y entrenando a {nombre} {apellido}...")

        success = self.system.register_and_train_employee(
            nombre=nombre,
            apellido=apellido,
            cargo=cargo,
            edad=edad,
            num_photos=None,
        )

        if success:
            messagebox.showinfo(
                "Éxito", "Empleado registrado y entrenado correctamente."
            )
            self._append_log(
                f"✓ Empleado {nombre} {apellido} listo para reconocimiento."
            )
            self.refresh_employees()
            self._refresh_access_stats()
        else:
            messagebox.showerror("Error", "Hubo un problema en el proceso.")
            self._append_log("✗ Error al registrar/entrenar empleado.")

    def on_retrain_employee(self):
        emp_id = self._get_selected_employee_id()
        if emp_id is None:
            return

        self._append_log(f"▶ Re-entrenando empleado ID {emp_id}...")
        success = self.system.retrain_employee(emp_id)

        if success:
            messagebox.showinfo("Éxito", "Empleado re-entrenado correctamente.")
            self._append_log(
                f"✓ Re-entrenamiento completado para ID {emp_id}."
            )
        else:
            messagebox.showerror(
                "Error", "No se pudo re-entrenar el empleado."
            )
            self._append_log(
                f"✗ Error al re-entrenar empleado ID {emp_id}."
            )

    def on_delete_employee(self):
        emp_id = self._get_selected_employee_id()
        if emp_id is None:
            return

        if not messagebox.askyesno(
            "Confirmar eliminación",
            f"¿Eliminar por completo al empleado ID {emp_id}?",
        ):
            return

        self._append_log(f"▶ Eliminando empleado ID {emp_id}...")
        success = self.system.delete_employee_gui(emp_id)

        if success:
            messagebox.showinfo("Éxito", "Empleado eliminado.")
            self._append_log(f"✓ Empleado ID {emp_id} eliminado.")
            self.refresh_employees()
            self._refresh_access_stats()
        else:
            messagebox.showerror(
                "Error", "No se pudo eliminar el empleado."
            )
            self._append_log(
                f"✗ Error al eliminar empleado ID {emp_id}."
            )

    def on_show_access_logs(self):
        """Muestra una ventana con los últimos accesos registrados,
        con filtros y exportación a CSV.
        """
        # Cargamos inicialmente los últimos N accesos
        logs = self.db.get_recent_access_logs(limit=200)

        win = ctk.CTkToplevel(self)
        win.title("Registros de acceso")
        win.geometry("900x480")

        # ---------- Filtros ----------
        filter_frame = ctk.CTkFrame(win)
        filter_frame.pack(fill="x", padx=10, pady=(10, 0))

        # Nombre / apellido (búsqueda por texto)
        ctk.CTkLabel(filter_frame, text="Nombre/apellido:").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        name_entry = ctk.CTkEntry(filter_frame, width=140)
        name_entry.grid(
            row=0, column=1, padx=5, pady=5, sticky="w"
        )
        
        # Hora de cierre (HH:MM)
        closing_label = ctk.CTkLabel(
            filter_frame,
            text="Hora de cierre (HH:MM):"
        )
        closing_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")

        closing_entry = ctk.CTkEntry(
            filter_frame,
            width=80,
            textvariable=self.closing_time_var
        )
        
        closing_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
          
        closing_save_btn = ctk.CTkButton(
            filter_frame,
            text="Guardar hora",
            width=100,
            command=self._on_save_closing_time
        )
        
        closing_save_btn.grid(row=2, column=2, padx=10, pady=5, sticky="w")                                 
            
        # Tipo de acceso
        ctk.CTkLabel(filter_frame, text="Tipo acceso:").grid(
            row=0, column=2, padx=5, pady=5, sticky="e"
        )
        tipo_combo = ctk.CTkComboBox(
            filter_frame,
            values=["Todos", "entrada", "salida", "desconocido"],
            width=130,
        )
        tipo_combo.set("Todos")
        tipo_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Fecha desde / hasta (YYYY-MM-DD)
        ctk.CTkLabel(filter_frame, text="Desde (YYYY-MM-DD):").grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        desde_entry = ctk.CTkEntry(filter_frame, width=140)
        desde_entry.grid(
            row=1, column=1, padx=5, pady=5, sticky="w"
        )

        ctk.CTkLabel(filter_frame, text="Hasta (YYYY-MM-DD):").grid(
            row=1, column=2, padx=5, pady=5, sticky="e"
        )
        hasta_entry = ctk.CTkEntry(filter_frame, width=140)
        hasta_entry.grid(
            row=1, column=3, padx=5, pady=5, sticky="w"
        )

        # Botones de acción
        buttons_frame = ctk.CTkFrame(win)
        buttons_frame.pack(fill="x", padx=10, pady=(5, 0))
        buttons_frame.grid_columnconfigure((0, 1), weight=1)

        def apply_filters():
            """Aplica filtros usando get_access_logs + filtrado en memoria."""
            nombre_filtro = name_entry.get().strip().lower()
            tipo_filtro = tipo_combo.get()
            desde = desde_entry.get().strip() or None
            hasta = hasta_entry.get().strip() or None

            # Validación simple de fechas
            if desde and len(desde) != 10:
                messagebox.showerror(
                    "Error",
                    "La fecha 'Desde' debe tener formato YYYY-MM-DD.",
                )
                return
            if hasta and len(hasta) != 10:
                messagebox.showerror(
                    "Error",
                    "La fecha 'Hasta' debe tener formato YYYY-MM-DD.",
                )
                return

            # Obtener desde BD con rango de fechas
            base_logs = self.db.get_access_logs(
                employee_id=None,
                start_date=desde,
                end_date=hasta,
            )

            # Filtrar en memoria por nombre y tipo_acceso
            filtered = []
            for log in base_logs:
                nom = (
                    (log.get("nombre") or "")
                    + " "
                    + (log.get("apellido") or "")
                ).strip().lower()
                if nombre_filtro and nombre_filtro not in nom:
                    continue
                if (
                    tipo_filtro != "Todos"
                    and log.get("tipo_acceso") != tipo_filtro
                ):
                    continue
                filtered.append(log)

            # Actualizar tabla con filtered
            reload_tree(filtered)

        def export_csv():
            """Exporta los registros actualmente mostrados en la tabla a un CSV."""
            # Recuperar todas las filas de la tabla
            rows = []
            for item_id in tree.get_children():
                vals = tree.item(item_id, "values")
                rows.append(vals)

            if not rows:
                messagebox.showinfo(
                    "Exportar CSV",
                    "No hay registros para exportar.",
                )
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("Todos los archivos", "*.*")],
                title="Guardar registros como CSV",
            )
            if not file_path:
                return

            # Escribir CSV
            import csv

            headers = [
                "fecha_hora",
                "empleado",
                "tipo_acceso",
                "confianza",
            ]
            with open(
                file_path, "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(headers)
                for row in rows:
                    writer.writerow(row)

            messagebox.showinfo(
                "Exportar CSV",
                f"Registros exportados a:\n{file_path}",
            )

        apply_btn = ctk.CTkButton(
            buttons_frame,
            text="Aplicar filtros",
            command=apply_filters,
        )
        apply_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        export_btn = ctk.CTkButton(
            buttons_frame,
            text="Exportar CSV",
            command=export_csv,
        )
        export_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # ---------- Tabla de resultados ----------
        cols = ("fecha_hora", "empleado", "resultado", "confianza")
        tree = ttk.Treeview(win, columns=cols, show="headings")

        tree.heading("fecha_hora", text="Fecha y hora")
        tree.heading("empleado", text="Empleado")
        tree.heading("resultado", text="Resultado")
        tree.heading("confianza", text="Confianza")

        tree.column("fecha_hora", width=170, anchor="center")
        tree.column("empleado", width=260, anchor="w")
        tree.column("resultado", width=110, anchor="center")
        tree.column("confianza", width=100, anchor="center")

        tree.pack(fill="both", expand=True, padx=10, pady=10)

        def reload_tree(data_logs):
            # Vaciar tabla
            for item in tree.get_children():
                tree.delete(item)

            # Rellenar filas
            for log in data_logs:
                if log.get("nombre") and log.get("apellido"):
                    empleado = f"{log['nombre']} {log['apellido']}"
                else:
                    empleado = "Desconocido"

                conf_val = log.get("confianza")
                conf_txt = ""
                if conf_val is not None:
                    try:
                        # Convertir a float aunque venga como bytes o string
                        conf_val = float(conf_val)
                        conf_txt = f"{conf_val:.2%}"
                    except (TypeError, ValueError):
                        # Si no se puede convertir, mostrar el valor "tal cual"
                        conf_txt = str(conf_val)

                tree.insert(
                    "",
                    "end",
                    values=(
                        log["fecha_hora"],
                        empleado,
                        log["tipo_acceso"],
                        conf_txt,
                    ),
                )

        # Cargar datos iniciales en la tabla
        reload_tree(logs)

    # ------------------------------------------------------------------
    # Lógica de decisión de acceso
    # ------------------------------------------------------------------
    def _decide_access(self, emp_id: int, emp_data: dict, confidence: float):
        """
        Lógica de Negocio Entrada / Salida

        - Si no hay accesos previos -> ENTRADA
        - Si el último acceso fue ENTRADA -> SALIDA (mismo día)
        - Si el último acceso fue SALIDA -> ENTRADA (mismo día)
        - Si se detecta un mismo empleado antes de ACCESS_MIN_REENTRY_SECONDS
        desde el último acceso -> no se registra nada
        - Si el último acceso es de un día anterior -> ENTRADA (nuevo día)
        """
        now = datetime.now()
        last = self.db.get_last_access_for_employee(emp_id)

        # Si no hay último acceso, registramos como entrada
        if not last:
            next_event = "entrada"
        else:
            # Intentar obtener un datetime a partir del campo fecha_hora
            last_dt = None
            last_ts = last.get("fecha_hora")

            if isinstance(last_ts, datetime):
                last_dt = last_ts
            elif isinstance(last_ts, str):
                try:
                    last_dt = datetime.fromisoformat(last_ts)
                except Exception:
                    try:
                        last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        last_dt = None

            # Si no pudimos parsear la fecha, asumimos entrada para evitar bloquear el acceso
            if last_dt is None:
                next_event = "entrada"
            else:
                # 🟢 Si el último registro es de OTRO día, empezamos un ciclo nuevo
                if last_dt.date() < now.date():
                    next_event = "entrada"
                else:
                    # Mismo día: aplicamos ventana anti-rebote
                    delta_sec = (now - last_dt).total_seconds()

                    if delta_sec < ACCESS_MIN_REENTRY_SECONDS:
                        self._append_log(
                            "ℹ Reconocido de nuevo "
                            "(mismo acceso, no se registra): "
                            f"{emp_data['nombre']} {emp_data['apellido']} "
                            f"(conf: {confidence:.2%})"
                        )
                        return

                    last_type = last.get("tipo_acceso", "entrada")
                    if last_type == "salida":
                        next_event = "entrada"
                    else:
                        next_event = "salida"

        # Guardar en BD
        self.db.log_access(emp_id, next_event, confidence)

        # Mensajes de interfaz
        if next_event == "entrada":
            self._append_log(
                f"✓ ENTRADA registrada: "
                f"{emp_data['nombre']} {emp_data['apellido']} "
                f"(conf: {confidence:.2%})"
            )
        else:
            self._append_log(
                f"✓ SALIDA registrada: "
                f"{emp_data['nombre']} {emp_data['apellido']} "
                f"({confidence:.2%})"
            )

        self._show_access_status(next_event, emp_data)
        self._refresh_access_stats()
    # ------------------------------------------------------------------
    # Reconocimiento en vivo
    # ------------------------------------------------------------------
    def on_live_recognition(self):
        """Lanza reconocimiento en vivo usando la cámara seleccionada."""
        self._refresh_camera_combo()
        self._append_log(
            f"▶ Iniciando reconocimiento en vivo (cámara {self.selected_camera_index})..."
        )

        cap = cv2.VideoCapture(self.selected_camera_index)
        if not cap.isOpened():
            messagebox.showerror(
                "Error",
                "No se puede abrir la cámara seleccionada. "
                "Verifica la conexión o el índice elegido.",
            )
            self._append_log("✗ No se pudo abrir la cámara.")
            return

        # Forzar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        recognition_counter = {}  # employee_id -> frames consecutivos

        while True:
            ret, frame = cap.read()
            if not ret:
                self._append_log("✗ Error al leer frame de la cámara.")
                break

            (
                annotated_frame,
                results,
            ) = self.system.recognizer.recognize_faces_in_frame(frame)

            unknown_detected = False

            # Lógica de estabilización + control de acceso
            for result in results:
                emp_id = result.get("employee_id")
                emp_data = result.get("employee_data")
                conf = result.get("confidence", 0.0)

                if emp_data and conf >= RECOGNITION_THRESHOLD:
                    # Empleado conocido
                    count = recognition_counter.get(emp_id, 0) + 1
                    recognition_counter[emp_id] = count

                    # Por ejemplo: 5 frames seguidos por encima del umbral
                    if count == 5:
                        self._decide_access(emp_id, emp_data, conf)
                else:
                    # Desconocido o por debajo del umbral
                    if emp_id in recognition_counter:
                        recognition_counter[emp_id] = 0
                    unknown_detected = True

            # Registrar rostro desconocido cada UNKNOWN_LOG_INTERVAL
            if unknown_detected and results:
                now_ts = time.time()
                if now_ts - self.last_unknown_log_ts > UNKNOWN_LOG_INTERVAL:
                    self.db.log_access(None, "desconocido", 0.0)
                    self._append_log("⚠ Rostro desconocido detectado.")
                    self._show_access_status("desconocido", None)
                    self.last_unknown_log_ts = now_ts

            cv2.imshow(
                "Reconocimiento en vivo - Presiona 'q' para salir",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._append_log("⏹ Reconocimiento en vivo detenido.")
        # Actualizar estadísticas tras la sesión en vivo
        self._refresh_access_stats()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
