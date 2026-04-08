import cv2 as cv
import easyocr

img_path = "BSX.jpg"

img = cv.imread(img_path, cv.IMREAD_COLOR)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, buinary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
clean__img = cv.fastNlMeansDenoising(buinary, h=10)

edges = cv.Canny(clean__img, 30, 120)

keypoints = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, keypoints)

contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

plates = []
int =0
img_size = img.shape[0] * img.shape[1]
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    assert_ratio = w / h #tỉ lệ chiều dài và chiều cao
    area_ratio = w * h / img_size #tỉ lệ diện tích của biển số
    if (1.0< assert_ratio < 6.0) and (0.005 < area_ratio < 0.15):
        plates.append((x, y, w, h))
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
def crop_plate(x, y, w, h, pad =5):
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])
    return img[y1:y2, x1:x2]

for i, (x, y, w, h) in enumerate(plates):
    plate_img = crop_plate(x, y, w, h)
    reader = easyocr.Reader(['en'])
    gimg = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
    _, binary_plate = cv.threshold(gimg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    result = reader.readtext(binary_plate, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", detail=1)
    for (bbox, text, conf) in result:
        print(f"text: {text}, confidence: {conf}")



cv.imshow("Image", clean__img)
cv.waitKey(0)
cv.destroyAllWindows()