from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading
import queue
from collections import defaultdict

class ObjectDetector:
    def __init__(self):
        # Inisialisasi model
        self.model = YOLO('runs/coco_training/weights/best.pt')
        self.frame_queue = queue.Queue(maxsize=2)  # Batasi antrian frame
        self.result_queue = queue.Queue(maxsize=2)
        self.running = True

        # Inisialisasi TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.last_spoken = defaultdict(float)
        self.cooldown = 1.5  # Cooldown global

        # Start threads
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()

    def detection_worker(self):
        """Thread untuk pemrosesan model"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                results = self.model(frame, verbose=False)  # Nonaktifkan logging
                self.result_queue.put(results)
            except queue.Empty:
                continue

    def tts_worker(self):
        """Thread untuk text-to-speech"""
        while self.running:
            current_time = time.time()
            try:
                results = self.result_queue.get(timeout=0.5)
                spoken_objects = set()
                
                # Proses semua deteksi
                for box in results[0].boxes:
                    if box.conf[0] > 0.5:
                        obj_name = self.model.names[int(box.cls[0])]
                        if (current_time - self.last_spoken[obj_name]) > self.cooldown:
                            spoken_objects.add(obj_name)
                
                # Gabungkan objek terdeteksi
                if spoken_objects:
                    message = "Detected: " + ", ".join(spoken_objects)
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                    for obj in spoken_objects:
                        self.last_spoken[obj] = current_time
                        
            except queue.Empty:
                continue

    def run(self):
        cap = cv2.VideoCapture(0)
        try:
            while self.running:
                # Baca frame kamera
                ret, frame = cap.read()
                if not ret:
                    break

                # Masukkan frame ke antrian
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame.copy())

                # Gambar bounding box dari hasil terbaru
                try:
                    results = self.result_queue.get_nowait()
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()  # Ambil nilai confidence
                        obj_name = self.model.names[int(box.cls[0])]  # Ambil nama objek
                        label = f"{obj_name} {conf:.2%}"  # Format label dengan nama objek dan confidence
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Tambahkan label ke frame
                except queue.Empty:
                    pass

                # Tampilkan frame
                cv2.imshow('Stable Detection', frame)
                
                # Exit dengan 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.tts_engine.stop()

if __name__ == "__main__":
    detector = ObjectDetector()
    detector.run()