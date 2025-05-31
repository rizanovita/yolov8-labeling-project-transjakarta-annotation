🌱 YOLOv8 Object Detection - Data Labeling Portfolio🌱
Ini adalah proyek portofolio data labeling dan pelatihan model YOLOv8 yang saya kerjakan menggunakan dataset custom untuk mendeteksi objek **Bus Transjakarta**, **Mobil**, dan **Motor**.

Proyek ini mencakup proses:
- Data labeling (annotasi bounding box)
- Setup konfigurasi YOLOv8
- Training dan validasi model
- Inference untuk menguji model

📁 Struktur Folder 
yolov8-labeling-project/

├── [📄 Lihat file data.yaml](./data.yaml)
├── 📂 [Buka folder `train`](./train/) # Data training
├── 📂 [Buka folder `valid`](./valid/) # Data validasi
├── 📂 [Buka folder `test`](./test/) # Data pengujian
├── README.md # Dokumentasi proyek
├── [📄 Lihat file best(2).pt](./best(2).pt) # Model hasil training
├── [📄 Lihat file results.md](./results.md)  # Catatan hasil training/inference

---

## 📦 Dataset

Dataset dibuat dan dilabeli secara manual menggunakan Roboflow, lalu diekspor dalam format YOLOv8. Label terdiri dari beberapa kelas objek relevan dengan kasus nyata.

- Jumlah gambar train: **20**
- Jumlah gambar valid: **6**
- Jumlah kelas: **(otomatis terdeteksi dari `data.yaml`)**

---

> ⚠️ Gambar & label hanya digunakan untuk keperluan edukasi dan demonstrasi portofolio.

![Contoh Labeling](data/project-1-10-_jpeg.rf.9a1e90283a56dad6466514b11ab35344.jpg)
![Contoh Labeling](data/project-1-14-_jpeg.rf.76b8f90753c37dbea623b4ce8c7fd4e7.jpg)
![Contoh Labeling](data/project-1-21-_jpeg.rf.7f25fb3bfdafaf649bbf27d4f3f34252.jpg)

---

## ⚙️ Model Training

Model dilatih menggunakan YOLOv8 dari library `ultralytics`.

- Training dilakukan di Google Colab 
- Hasil disimpan di: `yolov8-training/best.pt`

---

## 🔍 Inference & Evaluasi

![Contoh Labeling](data/screenshot-1748088234389.png)
![Contoh Labeling](data/screenshot-1748088154910.png)
![Contoh Labeling](data/screenshot-1748088168669.png)
![Contoh Labeling](data/screenshot-1748088189008.png)
![Contoh Labeling](data/screenshot-1748088234389.png)

Confusion matrix
![Contoh Labeling](data/confusion_matrix.png)

Results
![Contoh Labeling](data/results.png)


