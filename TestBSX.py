import cv2 as cv
import easyocr
import threading

# 1. Khởi tạo reader (Sử dụng CPU)
reader = easyocr.Reader(['en'], gpu=False) 

video_path = "plate2.mp4"
cap = cv.VideoCapture(video_path)

# Biến lưu trữ kết quả và trạng thái luồng
last_detected_text = ""
is_processing = False  # Cờ kiểm tra xem OCR có đang bận không

def perform_ocr(roi_img):
    """Hàm chạy ngầm để quét chữ mà không làm đứng video"""
    global last_detected_text, is_processing
    
    # Quét với các tham số tối ưu độ chính xác
    # results trả về định dạng: [(bbox, text, confidence)]
    results = reader.readtext(roi_img, 
                              allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ",
                              paragraph=False,
                              mag_ratio=2)
    
    if results:
        # Sắp xếp để lấy kết quả có độ tin cậy (confidence) cao nhất
        results.sort(key=lambda x: x[2], reverse=True)
        text = results[0][1]
        confidence = results[0][2]
        
        # CHỈ CẬP NHẬT NẾU ĐỘ TIN CẬY TRÊN 40% (0.4)
        # Bạn có thể tăng lên 0.6 hoặc 0.7 để khắt khe hơn
        if confidence > 0.4:
            last_detected_text = text.upper()
            print(f"Detected: {last_detected_text} (Conf: {confidence:.2f})")

    is_processing = False # Giải phóng cờ sau khi quét xong

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý ảnh cơ bản để tìm khung biển số
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blur, 30, 150)
    
    contours, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_size = frame.shape[0] * frame.shape[1]

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / h
        area_ratio = (w * h) / img_size
        
        # Lọc vùng có hình dáng giống biển số xe
        if (2.0 < aspect_ratio < 5.5) and (0.0005 < area_ratio < 0.02):
            
            # 2. VẼ BOX MÀU ĐỎ (Luôn vẽ để mượt hình)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # 3. TIỀN XỬ LÝ VÙNG BIỂN SỐ ĐỂ TĂNG ĐỘ CHÍNH XÁC
            if not is_processing:
                is_processing = True
                
                # Cắt vùng biển số
                plate_roi = gray[y:y+h, x:x+w]
                
                # Phóng to vùng biển số giúp EasyOCR đọc tốt hơn
                plate_roi = cv.resize(plate_roi, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
                
                # Tăng tương phản bằng Thresholding
                _, plate_roi = cv.threshold(plate_roi, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                # Chạy OCR trong luồng riêng biệt
                thread = threading.Thread(target=perform_ocr, args=(plate_roi,))
                thread.daemon = True # Tự động tắt luồng khi đóng chương trình
                thread.start()

            # 4. HIỂN THỊ KẾT QUẢ
            if last_detected_text:
                cv.putText(frame, last_detected_text, (x, y - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hiển thị video chính
    cv.imshow("License Plate Detection - Smooth & Accurate", frame)

    # ĐIỀU CHỈNH TỐC ĐỘ: 
    # 1: Nhanh nhất (tùy CPU)
    # 33: Tốc độ bình thường (30fps)
    # 100: Chậm (Slow motion)
    if cv.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
