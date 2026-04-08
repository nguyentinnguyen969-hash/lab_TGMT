import cv2 as cv
import easyocr
import threading
from datetime import datetime

# 1. Khởi tạo reader (Sử dụng CPU)
reader = easyocr.Reader(['en'], gpu=False) 

video_path = "plate2.mp4"
cap = cv.VideoCapture(video_path)

# Biến lưu trữ kết quả và trạng thái luồng
last_detected_text = ""
is_processing = False 
history_file = "history_plates.txt"

def save_to_file(text, conf):
    """Lưu biển số vào file txt kèm thời gian"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Biển số: {text} - Độ tin cậy: {conf:.2f}\n")

def perform_ocr(roi_img):
    """Hàm chạy ngầm để quét chữ"""
    global last_detected_text, is_processing
    
    results = reader.readtext(roi_img, 
                              allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ",
                              paragraph=False,
                              mag_ratio=2)
    
    if results:
        results.sort(key=lambda x: x[2], reverse=True)
        text = results[0][1].upper()
        confidence = results[0][2]
        
        # CHỈ LẤY ĐỘ TIN CẬY TRÊN 0.6
        if confidence >= 0.6:
            # 1. Xuất ra Terminal
            print(f"---> PHÁT HIỆN: {text} | Confidence: {confidence:.2f}")
            
            # 2. Lưu vào biến hiển thị trên màn hình
            last_detected_text = text
            
            # 3. Ghi vào file text để lưu lại lịch sử
            save_to_file(text, confidence)

    is_processing = False

# Tạo tiêu đề trong file text khi bắt đầu chạy
with open(history_file, "a", encoding="utf-8") as f:
    f.write(f"\n--- BẮT ĐẦU PHIÊN QUÉT MỚI ({datetime.now()}) ---\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blur, 30, 150)
    
    contours, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_size = frame.shape[0] * frame.shape[1]

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / h
        area_ratio = (w * h) / img_size
        
        # Lọc khung hình biển số
        if (2.0 < aspect_ratio < 5.5) and (0.0005 < area_ratio < 0.02):
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            if not is_processing:
                is_processing = True
                plate_roi = gray[y:y+h, x:x+w]
                # Tiền xử lý để tăng độ nét
                plate_roi = cv.resize(plate_roi, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
                _, plate_roi = cv.threshold(plate_roi, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                thread = threading.Thread(target=perform_ocr, args=(plate_roi,))
                thread.daemon = True
                thread.start()

            if last_detected_text:
                cv.putText(frame, last_detected_text, (x, y - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow("License Plate Detection", frame)

    if cv.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print(f"--- Đã lưu lịch sử vào file: {history_file} ---")
