

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

Mantap, ini yang lu maksud 👍
Sekarang gue bikinin **LKM 4 → KELOMPOK 3** dengan format **sama persis kayak sebelumnya (per sintaks + penjelasan + siap hafalan + GitHub)**.

Ini gue ambil dari file kelompok 3 yang lu upload  dan gue rapihin jadi versi belajar.

---

# 5. LKM 4 — PROJECT (KELOMPOK 3: FACE DETECTION + CNN)

## A. Tahap 1 — Ambil Dataset (Google Drive)

---

### a) Import Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Penjelasan:**

* `from google.colab import drive`
  Mengimpor modul Google Drive di Colab
* `drive.mount('/content/drive')`
  Menghubungkan Google Drive ke Colab agar dataset bisa diakses

**Essay:**

> Sintaks ini digunakan untuk menghubungkan Google Drive ke Google Colab sehingga dataset dapat diakses dan digunakan dalam proses pengolahan data. 

---

## B. Tahap 2 — Preprocessing (Resize)

---

### a) Import library

```python
import os
import cv2
```

**Penjelasan:**

* `os` → akses folder
* `cv2` → pengolahan citra

---

### b) Menentukan path dataset

```python
input_folder = "/dataset_asli"
output_folder = "/dataset_resize"
target_size = (224, 224)
```

**Penjelasan:**

* `input_folder` → lokasi dataset awal
* `output_folder` → lokasi hasil resize
* `target_size` → ukuran gambar standar

---

### c) Loop semua file gambar

```python
for root, dirs, files in os.walk(input_folder):
    for file in files:
```

**Penjelasan:**

* `os.walk()` → menelusuri semua folder dan subfolder
* `for file in files` → membaca setiap file

---

### d) Cek file gambar

```python
if file.lower().endswith(('.jpg','.png','.jpeg')):
```

**Penjelasan:**

* Mengecek apakah file adalah gambar

---

### e) Baca gambar

```python
img = cv2.imread(img_path)
```

---

### f) Resize gambar

```python
img_resized = cv2.resize(img, target_size)
```

---

### g) Simpan hasil resize

```python
cv2.imwrite(os.path.join(save_dir, file), img_resized)
```

---

**Essay inti Tahap 2:**

> Tahap ini digunakan untuk melakukan preprocessing dataset dengan cara mengubah ukuran semua gambar menjadi 224x224 piksel agar seragam dan siap digunakan dalam proses training model. 

---

## C. Tahap 2 — Face Detection (Crop Wajah)

---

### a) Load Haar Cascade

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```

---

### b) Konversi grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

### c) Deteksi wajah

```python
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

---

### d) Crop wajah

```python
for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
```

---

### e) Resize wajah

```python
face = cv2.resize(face, (224,224))
```

---

### f) Simpan wajah

```python
cv2.imwrite(os.path.join(save_dir, file), face)
```

---

**Essay inti:**

> Tahap ini digunakan untuk mendeteksi wajah menggunakan Haar Cascade, kemudian memotong (cropping) area wajah dan menyimpannya sebagai dataset yang lebih fokus pada objek wajah. 

---

## D. Tahap 3 — Load Dataset & Split Data

---

### a) Import TensorFlow

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

---

### b) Membuat ImageDataGenerator

```python
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```

---

### c) Load data training

```python
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
```

---

### d) Load data validation

```python
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

---

**Essay inti:**

> Tahap ini digunakan untuk memuat dataset ke dalam model, melakukan normalisasi nilai piksel, serta membagi data menjadi training dan validation dengan perbandingan 80% dan 20%. 

---

## E. Tahap 4 — Training Model (Transfer Learning)

---

### a) Import model

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
```

---

### b) Load MobileNetV2

```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
```

---

### c) Freeze layer

```python
for layer in base_model.layers:
    layer.trainable = False
```

---

### d) Tambah layer baru

```python
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)
```

---

### e) Gabung model

```python
transfer_model = Model(inputs=base_model.input, outputs=predictions)
```

---

### f) Compile model

```python
transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### g) Training model

```python
history = transfer_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
```

---

**Essay inti:**

> Tahap ini digunakan untuk melatih model menggunakan transfer learning dengan MobileNetV2 agar dapat mengenali wajah berdasarkan dataset yang telah diproses sebelumnya. 

---

## F. Tahap 5 — Visualisasi Accuracy

---

### a) Import matplotlib

```python
import matplotlib.pyplot as plt
```

---

### b) Plot accuracy

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training','Validation'])
plt.title("Accuracy")
plt.show()
```

---

**Essay:**

> Sintaks ini digunakan untuk menampilkan grafik akurasi training dan validation untuk melihat performa model selama proses training. 

---

## G. Tahap 6 — Testing Model

---

### a) Load gambar

```python
img = cv2.imread(img_path)
```

---

### b) Convert grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

### c) Deteksi wajah

```python
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

---

### d) Crop wajah

```python
face = img[y:y+h, x:x+w]
```

---

### e) Resize + preprocessing

```python
face_resized = cv2.resize(face, (224,224))
img_array = image.img_to_array(face_resized)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
```

---

### f) Prediksi

```python
prediction = transfer_model.predict(img_array)
predicted_class = labels[np.argmax(prediction)]
```

---

### g) Tampilkan hasil

```python
cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
cv2.putText(img, predicted_class, (x,y-10), ...)
```

---

**Essay inti:**

> Tahap ini digunakan untuk menguji model dengan gambar baru, melakukan deteksi wajah, preprocessing, kemudian memprediksi identitas wajah menggunakan model yang telah dilatih. 

---

# INTI KELOMPOK 3 (WAJIB PAHAM)

Alur:

1. Ambil dataset dari Drive
2. Resize gambar
3. Deteksi & crop wajah
4. Load dataset ke CNN
5. Training MobileNetV2
6. Evaluasi
7. Testing (prediksi wajah)

---

# PERBEDAAN KELOMPOK 2 vs 3 (BIAR LU PAHAM DALAM)

| Kelompok   | Metode                            |
| ---------- | --------------------------------- |
| Kelompok 2 | LBPH (klasik OpenCV)              |
| Kelompok 3 | CNN + MobileNetV2 (Deep Learning) |

---

# KALIMAT AMAN UTS (KELOMPOK 3)

* `ImageDataGenerator()`

> Digunakan untuk preprocessing dataset dan membagi data menjadi training dan validation.

* `MobileNetV2()`

> Digunakan sebagai model pretrained untuk transfer learning dalam klasifikasi wajah.

* `model.fit()`

> Digunakan untuk melatih model menggunakan dataset training.

* `detectMultiScale()`

> Digunakan untuk mendeteksi wajah pada gambar.

* `predict()`

> Digunakan untuk memprediksi kelas wajah berdasarkan model.

---



