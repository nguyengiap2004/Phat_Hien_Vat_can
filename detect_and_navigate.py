import cv2
import torch
import time


# ---- 1. Cấu hình camera ----
ip_camera_url = 'http://10.143.151.11:8080/video'  # Thay bằng IP điện thoại thật
cap_phone = cv2.VideoCapture(ip_camera_url)
cap_laptop = cv2.VideoCapture(0)  # Thường là 0, có thể thử 1 nếu không được

# ---- 2. Load YOLO model ----
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# ---- 3. Đặt cờ trạng thái cho từng camera ----
phone_ok = True
laptop_ok = True

while True:
    # Đọc frame từng camera
    ret1, frame1 = cap_phone.read() if phone_ok else (False, None)
    ret2, frame2 = cap_laptop.read() if laptop_ok else (False, None)

    # Xử lý YOLO và hiển thị cho camera điện thoại
    if ret1 and frame1 is not None:
        results1 = model(frame1)
        for *box, conf, cls in results1.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame1, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.imshow('Camera IP (Dien thoai)', frame1)
        phone_ok = True
    else:
        if phone_ok:  # chỉ in thông báo khi trạng thái mới chuyển sang lỗi
            print("Mất kết nối hoặc không nhận được frame từ camera điện thoại!")
        phone_ok = False
        # Hiển thị khung đen hoặc overlay thông báo
        black1 = (frame1 if frame1 is not None else 
                  cv2.putText(
                      255 * (cv2.imread('') if cv2.imread('') is not None else 
                      cv2.UMat(240,320,cv2.CV_8UC3)), 
                      "Camera IP Loi!", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (0,0,255), 2))
        cv2.imshow('Camera IP (Dien thoai)', black1)

    # Xử lý YOLO và hiển thị cho camera laptop
    if ret2 and frame2 is not None:
        results2 = model(frame2)
        for *box, conf, cls in results2.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame2, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.imshow('Camera Laptop', frame2)
        laptop_ok = True
    else:
        if laptop_ok:
            print("Mất kết nối hoặc không nhận được frame từ camera laptop!")
        laptop_ok = False
        black2 = (frame2 if frame2 is not None else 
                  cv2.putText(
                      255 * (cv2.imread('') if cv2.imread('') is not None else 
                      cv2.UMat(240,320,cv2.CV_8UC3)), 
                      "Camera laptop Loi!", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (0,0,255), 2))
        cv2.imshow('Camera Laptop', black2)


    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap_phone.release()
cap_laptop.release()
cv2.destroyAllWindows()