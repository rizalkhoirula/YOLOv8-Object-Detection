# from ultralytics import YOLO
# import cv2

# # Load model terlatih dari Google Drive
# model = YOLO('runs/coco_training/weights/best.pt')  


# # Inisialisasi kamera
# cap = cv2.VideoCapture(0)  # 0 untuk kamera default

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Tidak dapat membaca frame dari kamera.")
#         break

#     # Deteksi objek pada frame
#     results = model(frame)

#     # Gambar kotak deteksi dan label
#     for box in results[0].boxes:
#         # Koordinat bounding box
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
        
#         # Tingkat kepercayaan dan label kelas
#         confidence = box.conf[0]
#         cls = int(box.cls[0])
#         label = f"{model.names[cls]} {confidence:.2%}"

#         # Gambar kotak dan label pada frame
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Kotak hijau
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Tampilkan frame dengan deteksi
#     cv2.imshow('YOLOv8 Object Detection', frame)

#     # Keluar jika tombol 'q' ditekan
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Bersihkan resource
# cap.release()
# cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import pyttsx3
from collections import defaultdict
import time

# Load model YOLO
model = YOLO('runs/coco_training/weights/best.pt')

# Inisialisasi TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Kecepatan bicara

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Konfigurasi
confidence_threshold = 0.5  # Ambang batas kepercayaan lebih rendah
cooldown = 2  # Cooldown lebih pendek

# Untuk melacak waktu terakhir tiap objek diucapkan
last_spoken = defaultdict(float)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    # Deteksi objek
    results = model(frame)
    current_objects = []  # Menggunakan list untuk menyimpan semua deteksi
    
    # Proses hasil deteksi
    for box in results[0].boxes:
        conf = box.conf[0].item()
        if conf >= confidence_threshold:
            cls = int(box.cls[0])
            obj_name = model.names[cls]
            current_objects.append(obj_name)  # Menambahkan semua deteksi

            # Gambar bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{obj_name} {conf:.2%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Logic pengucapan suara
    current_time = time.time()
    spoken_in_frame = set()  # Mencegah pengulangan dalam frame yang sama
    
    for obj in current_objects:
        # Cek cooldown dan pastikan objek belum diucapkan dalam frame ini
        if obj not in spoken_in_frame and (current_time - last_spoken[obj]) > cooldown:
            engine.say(f"{obj}")
            last_spoken[obj] = current_time
            spoken_in_frame.add(obj)  # Tandai objek yang sudah diucapkan
    
    engine.runAndWait()

    # Tampilkan frame
    cv2.imshow('Multi-Object Detection', frame)

    # Keluar dengan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()