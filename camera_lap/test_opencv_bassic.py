import cv2
from ultralytics import YOLO
import time


class YOLOCameraApp:
    def __init__(self):
        print("=" * 60)
        print("=" * 60)
        
        # Khởi tạo biến
        self.cap = None
        self.model = None
        self.conf_threshold = 0.5
        self.is_running = False
        
        # Load model YOLOv8 segmentation
        self.load_model()
        
    def load_model(self):
        """Load YOLOv8 segmentation model"""
        try:
            print("\n🔄 Đang tải model YOLOv8n-seg...")
            self.model = YOLO('yolov8n-seg.pt')  # YOLOv8 nano segmentation
            print("✅ Model đã tải thành công!")
        except Exception as e:
            print(f"❌ Lỗi tải model: {str(e)}")
            exit(1)
            
    def start_camera(self):
        """Bắt đầu camera stream"""
        print("\n🎥 Đang khởi động camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Không thể mở camera!")
            return False
            
        # Cài đặt độ phân giải
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Camera đã sẵn sàng!")
        print("\n" + "=" * 60)
        print("⌨️  PHÍM ĐIỀU KHIỂN:")
        print("=" * 60)
        print("  [Q] hoặc [ESC] - Thoát chương trình")
        print("  [+] hoặc [=]   - Tăng ngưỡng tin cậy")
        print("  [-]            - Giảm ngưỡng tin cậy")
        print("  [SPACE]        - Tạm dừng/Tiếp tục")
        print("  [S]            - Chụp ảnh màn hình")
        print("=" * 60)
        
        return True
        
    def process_video(self):
        """Xử lý video stream với YOLOv8 segmentation"""
        self.is_running = True
        frame_count = 0
        fps_time = time.time()
        fps = 0
        paused = False
        last_frame = None
        screenshot_count = 0
        
        while self.is_running:
            if not paused:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Không đọc được frame!")
                    break
                
                # Chạy YOLOv8 segmentation
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Vẽ kết quả segmentation
                annotated_frame = results[0].plot()
                last_frame = annotated_frame.copy()
                
                # Đếm số lượng đối tượng phát hiện
                detections = results[0].boxes
                num_objects = len(detections)
                
                # Tính FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    current_time = time.time()
                    fps = 10 / (current_time - fps_time)
                    fps_time = current_time
                
                # Hiển thị thông tin trên frame
                info_y = 30
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Confidence: {self.conf_threshold:.2f}", (10, info_y + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, info_y + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Hiển thị danh sách đối tượng phát hiện
                if num_objects > 0:
                    detected_classes = {}
                    for box in detections:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        detected_classes[class_name] = detected_classes.get(class_name, 0) + 1
                    
                    # Hiển thị danh sách
                    y_offset = info_y + 110
                    for idx, (cls, count) in enumerate(detected_classes.items()):
                        if idx < 10:  # Chỉ hiển thị 10 đối tượng đầu
                            text = f"{cls}: {count}"
                            cv2.putText(annotated_frame, text, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            y_offset += 30
                
                # Hiển thị hướng dẫn
                help_text = "Q/ESC: Thoat | +/-: Nguong | SPACE: Tam dung | S: Chup anh"
                cv2.putText(annotated_frame, help_text, (10, annotated_frame.shape[0] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Hiển thị frame
                cv2.imshow('YOLOv8 Segmentation - Nhan dien vat the', annotated_frame)
            else:
                # Khi tạm dừng, hiển thị frame cuối cùng với text "PAUSED"
                if last_frame is not None:
                    paused_frame = last_frame.copy()
                    cv2.putText(paused_frame, "PAUSED", (paused_frame.shape[1]//2 - 100, paused_frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.imshow('YOLOv8 Segmentation - Nhan dien vat the', paused_frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q hoặc ESC
                print("\n👋 Đang thoát...")
                break
            elif key == ord('+') or key == ord('='):  # Tăng confidence
                self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
                print(f"⬆️  Ngưỡng tin cậy: {self.conf_threshold:.2f}")
            elif key == ord('-'):  # Giảm confidence
                self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                print(f"⬇️  Ngưỡng tin cậy: {self.conf_threshold:.2f}")
            elif key == ord(' '):  # SPACE - Tạm dừng
                paused = not paused
                if paused:
                    print("⏸️  Đã tạm dừng")
                else:
                    print("▶️  Tiếp tục")
            elif key == ord('s') or key == ord('S'):  # Chụp ảnh
                if last_frame is not None:
                    screenshot_count += 1
                    filename = f"screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, last_frame)
                    print(f"📸 Đã lưu ảnh: {filename}")
        
        self.stop_camera()
        
    def stop_camera(self):
        """Dừng camera stream"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Đã đóng camera và cửa sổ")
        
    def run(self):
        """Chạy ứng dụng"""
        if self.start_camera():
            self.process_video()
        else:
            print("❌ Không thể khởi động ứng dụng!")


def main():
    print("\n")
  
    app = YOLOCameraApp()
    app.run()
    
    print("\n" + "=" * 60)
    print("👋 Cảm ơn bạn đã sử dụng!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

