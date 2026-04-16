

# 1. LKM 4 — PROJECT (KELOMPOK 2: FACE RECOGNITION)

## A. Dataset Capture (Ambil Data Wajah)

---

### a) Import library

```python
import cv2
import os
```

**Penjelasan:**

* `cv2` → untuk buka kamera dan deteksi wajah
* `os` → untuk akses file dan folder dataset

**Essay:**

> Sintaks ini digunakan untuk mengimpor library OpenCV untuk pengolahan citra dan library OS untuk pengelolaan file dan folder.

---

### b) Membuka kamera

```python
cam = cv2.VideoCapture(0)
```

**Penjelasan:**

* Membuka webcam utama (index 0)

**Essay:**

> Sintaks ini digunakan untuk mengakses webcam agar sistem dapat mengambil gambar wajah secara real-time.

---

### c) Load Haar Cascade

```python
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```

**Penjelasan:**

* Memuat model deteksi wajah dari file XML

**Essay:**

> Sintaks ini digunakan untuk memuat model Haar Cascade yang digunakan dalam proses deteksi wajah pada gambar.

---

### d) Ambil input ID dan nama

```python
face_id = input("Masukkan ID: ")
face_name = input("Masukkan Nama: ")
```

**Penjelasan:**

* Mengambil ID dan nama user sebagai label dataset

**Essay:**

> Sintaks ini digunakan untuk mengambil input ID dan nama user yang akan digunakan sebagai label identitas dalam dataset wajah.

---

### e) Membaca frame kamera

```python
ret, img = cam.read()
```

**Penjelasan:**

* `ret` → status berhasil/gagal
* `img` → frame dari kamera

---

### f) Flip gambar

```python
img = cv2.flip(img, 1)
```

**Penjelasan:**

* Membalik gambar horizontal (mirror)

---

### g) Konversi ke grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Penjelasan:**

* Mengubah gambar ke grayscale agar deteksi lebih cepat

---

### h) Deteksi wajah

```python
faces = face_detector.detectMultiScale(gray, 1.3, 5)
```

**Penjelasan:**

* Mendeteksi posisi wajah dalam frame

---

### i) Gambar kotak wajah

```python
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
```

**Penjelasan:**

* Menandai wajah dengan bounding box

---

### j) Simpan wajah ke dataset

```python
cv2.imwrite(
    f"dataset/User.{face_id}.{count}.jpg",
    gray[y:y+h, x:x+w]
)
```

**Penjelasan:**

* Crop wajah dari gambar
* Simpan ke folder dataset

**Essay:**

> Sintaks ini digunakan untuk memotong area wajah dari frame dan menyimpannya ke dataset sebagai data training.

---

---

## B. Training Model (LBPH)

---

### a) Membuat model LBPH

```python
recognizer = cv2.face.LBPHFaceRecognizer_create()
```

**Penjelasan:**

* Membuat model face recognition

---

### b) Ambil semua gambar dataset

```python
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
```

**Penjelasan:**

* Mengambil semua file dalam dataset

---

### c) Baca gambar dan ubah ke array

```python
PIL_img = Image.open(imagePath).convert('L')
img_numpy = np.array(PIL_img, 'uint8')
```

**Penjelasan:**

* Convert gambar ke grayscale dan array

---

### d) Ambil ID dari nama file

```python
id = int(os.path.split(imagePath)[-1].split(".")[1])
```

**Penjelasan:**

* Mengambil label ID dari nama file

---

### e) Deteksi wajah dataset

```python
faces = detector.detectMultiScale(img_numpy)
```

---

### f) Simpan data training

```python
faceSamples.append(img_numpy[y:y+h, x:x+w])
ids.append(id)
```

---

### g) Training model

```python
recognizer.train(faceSamples, np.array(ids))
```

**Essay:**

> Sintaks ini digunakan untuk melatih model LBPH menggunakan dataset wajah dan label ID.

---

### h) Simpan model

```python
recognizer.save('trainer/trainer.yml')
```

**Penjelasan:**

* Menyimpan hasil training

---

---

## C. Face Recognition Real-Time

---

### a) Load model

```python
recognizer.read('trainer/trainer.yml')
```

---

### b) Baca labels

```python
names[int(id_str)] = name
```

---

### c) Prediksi wajah

```python
id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
```

**Penjelasan:**

* Menebak identitas wajah

---

### d) Cek confidence

```python
if confidence < 70:
    name = names[id]
else:
    name = "Unknown"
```

---

### e) Tampilkan nama

```python
cv2.putText(img, name, (x,y), font, 1, (255,255,255), 2)
```

---

# INTI KELOMPOK 2 (WAJIB PAHAM)

Alur project:

1. Ambil dataset wajah dari kamera
2. Simpan ke folder dataset
3. Training model LBPH
4. Load model
5. Deteksi wajah real-time
6. Prediksi identitas wajah

---

# KALIMAT AMAN BUAT UTS (KHUSUS KELOMPOK 2)

Kalau keluar soal:

* `detectMultiScale()`

> Digunakan untuk mendeteksi wajah pada gambar grayscale.

* `cv2.VideoCapture(0)`

> Digunakan untuk membuka webcam agar dapat mengambil gambar secara real-time.

* `recognizer.train()`

> Digunakan untuk melatih model face recognition menggunakan dataset wajah.

* `recognizer.predict()`

> Digunakan untuk memprediksi identitas wajah berdasarkan model yang telah dilatih.

* `cv2.imwrite()`

> Digunakan untuk menyimpan hasil crop wajah ke dalam dataset.

---

