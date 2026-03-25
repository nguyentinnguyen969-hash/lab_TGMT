"""
Test AI Decision Making - Su dung Webcam Laptop
Khong can ESP32, MQTT, chi test YOLO + Decision Maker
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from ultralytics import YOLO


# ========== CONFIG ==========
YOLO_MODEL_PATH = "yolov8n-seg.pt"
YOLO_CONFIDENCE = 0.6

# Vung nguy hiem
DANGER_ZONE_X = (0.3, 0.7)
DANGER_ZONE_Y = (0.5, 1.0)
DANGER_SIZE_THRESHOLD = 0.15

# Obstacle classes
OBSTACLE_CLASSES = [0, 2, 16]  # person, car, dog

# Color
COLOR_SAFE = (0, 255, 255)
COLOR_DANGER = (255, 0, 0)
COLOR_OBSTACLE = (0, 0, 255)


# ========== YOLO DETECTOR ==========
class SimpleYOLODetector:
    def __init__(self):
        print(f"[YOLO] Dang tai model: {YOLO_MODEL_PATH}")
        self.model = YOLO(YOLO_MODEL_PATH)
        self.class_names = self.model.names
        print(f"[YOLO] Model da tai! Classes: {len(self.class_names)}")
    
    def detect(self, frame, confidence):
        """Phat hien vat the"""
        if frame is None:
            return None, None, []
        
        results = self.model.predict(frame, conf=confidence, verbose=False)
        
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else np.array([])
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes else np.array([])
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes else np.array([])
        
        h, w = frame.shape[:2]
        objects = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(classes[i])
            conf = float(confs[i])
            conf_percent = conf * 100
            class_name = self.class_names[class_id]
            
            center_x = (x1 + x2) / 2 / w
            center_y = (y1 + y2) / 2 / h
            obj_width = (x2 - x1) / w
            obj_height = (y2 - y1) / h
            obj_size = obj_width * obj_height
            
            in_danger_zone = (
                DANGER_ZONE_X[0] <= center_x <= DANGER_ZONE_X[1] and
                DANGER_ZONE_Y[0] <= center_y <= DANGER_ZONE_Y[1]
            )
            
            is_obstacle = class_id in OBSTACLE_CLASSES
            is_dangerous = in_danger_zone and obj_size > DANGER_SIZE_THRESHOLD and is_obstacle
            
            obj = {
                'box': (x1, y1, x2, y2),
                'class_id': class_id,
                'class_name': class_name,
                'confidence': conf,
                'confidence_percent': conf_percent,
                'center_x': center_x,
                'center_y': center_y,
                'size': obj_size,
                'is_obstacle': is_obstacle,
                'in_danger_zone': in_danger_zone,
                'is_dangerous': is_dangerous
            }
            
            objects.append(obj)
        
        annotated_frame = self.draw_detections(frame.copy(), objects)
        return results, annotated_frame, objects
    
    def draw_detections(self, frame, objects):
        """Ve detection len frame"""
        h, w = frame.shape[:2]
        
        # Ve vung nguy hiem
        danger_x1 = int(w * DANGER_ZONE_X[0])
        danger_x2 = int(w * DANGER_ZONE_X[1])
        danger_y1 = int(h * DANGER_ZONE_Y[0])
        danger_y2 = int(h * DANGER_ZONE_Y[1])
        
        cv2.rectangle(frame, (danger_x1, danger_y1), (danger_x2, danger_y2), COLOR_SAFE, 2)
        cv2.putText(frame, "DANGER ZONE", (danger_x1+5, danger_y1+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SAFE, 2)
        
        # Dem objects
        dangerous_count = len([o for o in objects if o['is_dangerous']])
        obstacle_count = len([o for o in objects if o['is_obstacle']])
        
        # Info panel
        cv2.putText(frame, f"Objects: {len(objects)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Obstacles: {obstacle_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(frame, f"Dangerous: {dangerous_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Canh bao
        if dangerous_count > 0:
            cv2.putText(frame, "!!! DANGER !!!", (w//2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Ve tung object
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            
            if obj['is_dangerous']:
                color = COLOR_DANGER
                thickness = 3
            elif obj['is_obstacle']:
                color = COLOR_OBSTACLE
                thickness = 2
            else:
                color = (0, 255, 0)
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            conf_percent = obj['confidence_percent']
            label = f"{obj['class_name']} {conf_percent:.1f}%"
            if obj['is_dangerous']:
                label += " [DANGER!]"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
            cv2.putText(frame, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


# ========== DECISION MAKER ==========
class SimpleDecisionMaker:
    def __init__(self):
        self.last_command = "STOP"
        self.command_history = []
        self.max_history = 5
    
    def decide_action(self, objects):
        """Quyet dinh hanh dong"""
        if not objects:
            return "FORWARD", 0
        
        dangerous_obstacles = [obj for obj in objects if obj['is_dangerous']]
        
        if not dangerous_obstacles:
            return "FORWARD", 0
        
        obstacle_positions = [obj['center_x'] for obj in dangerous_obstacles]
        avg_position = np.mean(obstacle_positions)
        
        left_obstacles = [obj for obj in dangerous_obstacles if obj['center_x'] < 0.4]
        center_obstacles = [obj for obj in dangerous_obstacles if 0.4 <= obj['center_x'] <= 0.6]
        right_obstacles = [obj for obj in dangerous_obstacles if obj['center_x'] > 0.6]
        
        command = "STOP"
        arrow = 0
        
        if len(center_obstacles) > 0:
            if len(left_obstacles) < len(right_obstacles):
                command = "LEFT"
                arrow = -90
            elif len(right_obstacles) < len(left_obstacles):
                command = "RIGHT"
                arrow = 90
            else:
                command = "STOP"
                arrow = 0
        elif avg_position < 0.4:
            command = "RIGHT"
            arrow = 90
        elif avg_position > 0.6:
            command = "LEFT"
            arrow = -90
        else:
            command = "FORWARD"
            arrow = 0
        
        self.command_history.append(command)
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)
        
        if len(self.command_history) >= 3:
            recent_commands = self.command_history[-3:]
            for cmd in set(recent_commands):
                if recent_commands.count(cmd) >= 2:
                    command = cmd
                    break
        
        self.last_command = command
        return command, arrow
    
    def draw_direction_arrow(self, frame, command, arrow_angle):
        """Ve mui ten chi huong"""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h - 80
        
        color_map = {
            'FORWARD': (0, 255, 0),
            'LEFT': (255, 255, 0),
            'RIGHT': (255, 255, 0),
            'STOP': (0, 0, 255),
            'BACK': (128, 0, 128)
        }
        
        color = color_map.get(command, (255, 255, 255))
        
        arrow_length = 60
        arrow_angle_rad = np.radians(arrow_angle - 90)
        
        end_x = int(center_x + arrow_length * np.cos(arrow_angle_rad))
        end_y = int(center_y + arrow_length * np.sin(arrow_angle_rad))
        
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                       color, 8, tipLength=0.3)
        
        cv2.putText(frame, command, (center_x-40, center_y+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        return frame


# ========== TEST APP ==========
class TestAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Test AI Decision - Webcam Laptop")
        self.root.geometry("1400x800")
        self.root.configure(bg="#0a0a0a")
        
        self.yolo_detector = None
        self.decision_maker = None
        
        self.is_running = False
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        self.current_confidence = YOLO_CONFIDENCE
        
        self.setup_ui()
        self.load_ai()
    
    def setup_ui(self):
        """Thiet lap giao dien"""
        
        main_container = tk.Frame(self.root, bg="#0a0a0a")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top bar
        top_bar = tk.Frame(main_container, bg="#1a1a1a", height=60)
        top_bar.pack(fill=tk.X, pady=(0, 10))
        top_bar.pack_propagate(False)
        
        tk.Label(
            top_bar,
            text="TEST AI DECISION MAKER",
            bg="#1a1a1a",
            fg="#ffffff",
            font=("Arial", 16, "bold")
        ).pack(side=tk.LEFT, padx=20, pady=15)
        
        self.fps_label = tk.Label(
            top_bar,
            text="FPS: 0",
            bg="#1a1a1a",
            fg="#00ff00",
            font=("Arial", 12)
        )
        self.fps_label.pack(side=tk.LEFT, padx=20)
        
        self.btn_start = tk.Button(
            top_bar,
            text="BAT DAU TEST",
            command=self.start_test,
            bg="#00aa00",
            fg="white",
            font=("Arial", 11, "bold"),
            width=15,
            height=2,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.btn_start.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Middle section
        middle_section = tk.Frame(main_container, bg="#0a0a0a")
        middle_section.pack(fill=tk.BOTH, expand=True)
        
        # Video
        left_panel = tk.Frame(middle_section, bg="#0a0a0a")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_container = tk.Frame(left_panel, bg="#000000", highlightbackground="#333333", highlightthickness=2)
        video_container.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(
            video_container,
            bg="#000000",
            text="NHAN BAT DAU TEST",
            fg="#666666",
            font=("Arial", 20)
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Right panel
        right_panel = tk.Frame(middle_section, bg="#1a1a1a", width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)
        
        # Confidence slider
        confidence_frame = tk.LabelFrame(
            right_panel,
            text="NGUONG NHAN DIEN (%)",
            bg="#1a1a1a",
            fg="#ffffff",
            font=("Arial", 11, "bold")
        )
        confidence_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.confidence_value_label = tk.Label(
            confidence_frame,
            text=f"{int(self.current_confidence * 100)}%",
            bg="#1a1a1a",
            fg="#00ff00",
            font=("Arial", 18, "bold")
        )
        self.confidence_value_label.pack(pady=(10, 5))
        
        self.confidence_slider = tk.Scale(
            confidence_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            bg="#1a1a1a",
            fg="#ffffff",
            troughcolor="#333333",
            activebackground="#00aa00",
            highlightthickness=0,
            length=280,
            command=self.on_confidence_change
        )
        self.confidence_slider.set(int(self.current_confidence * 100))
        self.confidence_slider.pack(padx=15, pady=(0, 10))
        
        # AI Decision
        decision_frame = tk.LabelFrame(
            right_panel,
            text="QUYET DINH AI",
            bg="#1a1a1a",
            fg="#ffffff",
            font=("Arial", 11, "bold")
        )
        decision_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.decision_label = tk.Label(
            decision_frame,
            text="LENH: -",
            bg="#1a1a1a",
            fg="#00ff00",
            font=("Arial", 20, "bold")
        )
        self.decision_label.pack(pady=15)
        
        # Stats
        stats_frame = tk.LabelFrame(
            right_panel,
            text="THONG KE",
            bg="#1a1a1a",
            fg="#ffffff",
            font=("Arial", 11, "bold")
        )
        stats_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.objects_label = tk.Label(
            stats_frame,
            text="Vat the: 0",
            bg="#1a1a1a",
            fg="#cccccc",
            font=("Consolas", 10),
            anchor=tk.W
        )
        self.objects_label.pack(fill=tk.X, padx=10, pady=3)
        
        self.obstacles_label = tk.Label(
            stats_frame,
            text="Vat can: 0",
            bg="#1a1a1a",
            fg="#cccccc",
            font=("Consolas", 10),
            anchor=tk.W
        )
        self.obstacles_label.pack(fill=tk.X, padx=10, pady=3)
        
        self.dangerous_label = tk.Label(
            stats_frame,
            text="Nguy hiem: 0",
            bg="#1a1a1a",
            fg="#ff3333",
            font=("Consolas", 10),
            anchor=tk.W
        )
        self.dangerous_label.pack(fill=tk.X, padx=10, pady=3)
        
        # Log
        log_frame = tk.LabelFrame(
            right_panel,
            text="NHAT KY",
            bg="#1a1a1a",
            fg="#ffffff",
            font=("Arial", 11, "bold")
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            log_frame,
            bg="#0a0a0a",
            fg="#00ff00",
            font=("Consolas", 9),
            yscrollcommand=log_scroll.set,
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        log_scroll.config(command=self.log_text.yview)
        
        self.log_text.tag_config("info", foreground="#00ff00")
        self.log_text.tag_config("command", foreground="#00aaff")
        self.log_text.tag_config("warning", foreground="#ffaa00")
    
    def on_confidence_change(self, value):
        """Thay doi confidence threshold"""
        confidence_percent = int(float(value))
        self.current_confidence = confidence_percent / 100.0
        self.confidence_value_label.config(text=f"{confidence_percent}%")
    
    def load_ai(self):
        """Tai YOLO va Decision Maker"""
        self.add_log("Dang tai YOLO model...", "info")
        try:
            self.yolo_detector = SimpleYOLODetector()
            self.decision_maker = SimpleDecisionMaker()
            self.add_log("YOLO model da tai", "info")
        except Exception as e:
            self.add_log(f"Loi YOLO: {str(e)}", "warning")
    
    def start_test(self):
        """Bat dau test voi webcam"""
        if self.is_running:
            # Dang chay, nut de STOP
            self.is_running = False
            self.btn_start.config(text="BAT DAU TEST", bg="#00aa00")
            self.add_log("Da dung test", "warning")
            if self.cap:
                self.cap.release()
                self.cap = None
        else:
            # Bat dau test
            self.add_log("Bat dau test voi webcam laptop...", "info")
            self.cap = cv2.VideoCapture(0)
            
            if self.cap.isOpened():
                self.is_running = True
                self.start_time = time.time()
                self.frame_count = 0
                self.btn_start.config(text="DUNG TEST", bg="#cc0000")
                self.add_log("Webcam da mo", "info")
                threading.Thread(target=self.update_frame_loop, daemon=True).start()
            else:
                self.add_log("Khong mo duoc webcam", "warning")
    
    def update_frame_loop(self):
        """Vong lap xu ly frame"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed
                
                # Process voi YOLO
                if self.yolo_detector and self.decision_maker:
                    results, annotated, objects = self.yolo_detector.detect(frame, self.current_confidence)
                    
                    if annotated is not None:
                        command, arrow = self.decision_maker.decide_action(objects)
                        annotated = self.decision_maker.draw_direction_arrow(annotated, command, arrow)
                        
                        # Cap nhat UI
                        self.root.after(0, self.update_stats, command, objects)
                        self.root.after(0, self.display_frame, annotated)
                else:
                    self.root.after(0, self.display_frame, frame)
                
                time.sleep(0.01)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)
        
        if self.cap:
            self.cap.release()
    
    def update_stats(self, command, objects):
        """Cap nhat thong ke"""
        self.decision_label.config(text=f"LENH: {command}")
        
        dangerous_count = len([o for o in objects if o['is_dangerous']])
        obstacle_count = len([o for o in objects if o['is_obstacle']])
        
        self.objects_label.config(text=f"Vat the: {len(objects)}")
        self.obstacles_label.config(text=f"Vat can: {obstacle_count}")
        self.dangerous_label.config(text=f"Nguy hiem: {dangerous_count}")
        
        # Log lenh
        self.add_log(f"AI: {command}", "command")
    
    def display_frame(self, frame):
        """Hien thi frame"""
        try:
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            
            if label_w > 1 and label_h > 1:
                scale = min(label_w / w, label_h / h)
                new_w = int(w * scale * 0.95)
                new_h = int(h * scale * 0.95)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk, text="")
        except Exception:
            pass
    
    def add_log(self, message, tag="info"):
        """Them log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_line, tag)
        self.log_text.see(tk.END)
        
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 100:
            self.log_text.delete('1.0', '2.0')
    
    def on_closing(self):
        """Dong ung dung"""
        self.is_running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    
    style = ttk.Style()
    style.theme_use('clam')
    
    app = TestAIApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("=" * 60)
    print("  TEST AI DECISION MAKER")
    print("  Webcam Laptop - No ESP32/MQTT")
    print("=" * 60)
    
    root.mainloop()


if __name__ == "__main__":
    main()

