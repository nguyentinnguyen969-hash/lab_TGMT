import cv2 as cv
import easyocr
import threading

# Khởi tạo reader
reader = easyocr.Reader(['en'], gpu=False) 

video_path = "plate2.mp4"
cap = cv.VideoCapture(video_path)


# Biến lưu trữ kết quả và trạng thái
last_detected_text = ""
is_processing = False  # Cờ kiểm tra xem OCR có đang bận quét không

def perform_ocr(roi_img):
    """Hàm chạy trong luồng riêng để không làm treo video"""
    global last_detected_text, is_processing
    result = reader.readtext(roi_img, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ", detail=0)
    if result:
        last_detected_text = result[0].upper()
    is_processing = False # Quét xong, giải phóng cờ

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
        
        # Lọc vùng nghi ngờ là biển số
        if (2.0 < aspect_ratio < 5.5) and (0.0005 < area_ratio < 0.02):
            # Vẽ box màu đỏ cho mọi frame (tạo cảm giác mượt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Nếu OCR không bận, đẩy vùng ROI vào luồng phụ để quét
            if not is_processing:
                is_processing = True
                plate_roi = gray[y:y+h, x:x+w]
                # Tạo thread mới để chạy OCR độc lập với vòng lặp video
                thread = threading.Thread(target=perform_ocr, args=(plate_roi,))
                thread.start()

            # Hiển thị kết quả cũ nhất có được
            if last_detected_text:
                cv.putText(frame, last_detected_text, (x, y - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Hiển thị video
    cv.imshow("BSX Detection - Smooth Mode", frame)

    # Thoát khi bấm 'q' hoặc khi video kết thúc
    if cv.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()