Siap, gua bikinin **penjelasan per sintaks** yang bisa lo hafalin buat ujian essay tulis tangan. Jadi kalau nanti dosen kasih potongan kode, lo tinggal jelasin fungsi tiap barisnya.

Ini gue rangkum dari **LKM 1–4** yang lo upload: pembacaan citra dan basic image processing di LKM 1, preprocessing dataset di LKM 2, CNN + transfer learning di LKM 3, dan project kelompok face recognition di LKM 4.    

---

# 1. LKM 1 — DASAR PEMBACAAN CITRA

## a) Import library

```python
import cv2
import numpy as np
```

**Penjelasan:**

* `import cv2`
  Untuk memanggil library **OpenCV**, yaitu library utama buat pengolahan citra dan computer vision.
* `import numpy as np`
  Untuk memanggil **NumPy**, biasanya dipakai untuk operasi array/matriks. Dalam citra digital, gambar itu sebenarnya disimpan sebagai array angka. 

**Kalau ditanya di essay:**

> Baris ini digunakan untuk mengimpor library OpenCV dan NumPy. OpenCV dipakai untuk membaca, menampilkan, dan mengolah citra, sedangkan NumPy dipakai karena citra digital direpresentasikan dalam bentuk array numerik.

---

## b) Membaca gambar

```python
img = cv2.imread('cctv.jpg')
```

**Penjelasan:**

* `cv2.imread()` dipakai untuk **membaca file gambar** dari penyimpanan.
* `'cctv.jpg'` adalah nama file gambar yang mau dibuka.
* Hasilnya disimpan ke variabel `img`.

**Makna penting:**
Kalau file berhasil dibaca, `img` berisi data citra. Kalau gagal, `img` biasanya bernilai `None`. 

---

## c) Mengecek apakah gambar berhasil dibaca

```python
if img is None:
    print("Gambar tidak ditemukan!")
else:
    print("Tipe data gambar:", type(img))
    print("Dimensi gambar:", img.shape)
```

**Penjelasan:**

* `if img is None:`
  Mengecek apakah gambar gagal dibaca.
* `print("Gambar tidak ditemukan!")`
  Menampilkan pesan error kalau file tidak ada atau path salah.
* `type(img)`
  Menampilkan tipe data variabel gambar, biasanya array NumPy.
* `img.shape`
  Menampilkan ukuran citra dalam format:

  * tinggi
  * lebar
  * jumlah channel

**Contoh hasil:** `(tinggi, lebar, 3)`
Angka `3` berarti gambar berwarna punya 3 channel. 

**Kalimat essay:**

> Sintaks ini digunakan untuk memastikan file gambar berhasil dibaca. Jika gambar tidak ditemukan maka program menampilkan pesan error. Jika berhasil, program menampilkan tipe data citra dan dimensinya yang terdiri dari tinggi, lebar, dan jumlah channel.

---

## d) Menampilkan gambar

```python
cv2.imshow('CCTV Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Penjelasan:**

* `cv2.imshow('CCTV Image', img)`
  Menampilkan gambar pada jendela baru dengan judul `"CCTV Image"`.
* `cv2.waitKey(0)`
  Menunggu tombol keyboard ditekan. Angka `0` artinya tunggu terus sampai ada input.
* `cv2.destroyAllWindows()`
  Menutup semua jendela gambar yang dibuka OpenCV. 

**Kalimat essay:**

> Baris ini digunakan untuk menampilkan citra ke layar, menahan jendela agar tidak langsung tertutup, lalu menutup seluruh jendela setelah ada input dari keyboard.

---

## e) Mengakses nilai piksel

```python
pixel = img[100, 150]
print("Nilai piksel (BGR):", pixel)

blue = img[100, 150, 0]
print("Channel Biru:", blue)
```

**Penjelasan:**

* `img[100, 150]`
  Mengambil nilai piksel pada koordinat baris 100 dan kolom 150.
* Hasil piksel biasanya berupa 3 nilai channel warna.
* Di OpenCV, urutan warna adalah **BGR**, bukan RGB.
* `img[100,150,0]`
  Mengambil nilai channel ke-0, yaitu **Biru**. 

**Kalimat essay:**

> Sintaks ini digunakan untuk mengambil nilai piksel pada koordinat tertentu. Karena gambar berwarna disimpan dalam format BGR, maka satu piksel memiliki tiga nilai intensitas yang merepresentasikan channel biru, hijau, dan merah.

---

# 2. LKM 1 PERTEMUAN 2 — OPERASI DASAR CITRA

## a) Menampilkan citra di Colab

```python
from google.colab.patches import cv2_imshow
cv2_imshow(img)
```

**Penjelasan:**

* `cv2_imshow` dipakai khusus di Google Colab karena `cv2.imshow()` sering tidak berjalan normal di sana.
* Fungsinya tetap sama: menampilkan gambar. 

---

## b) Resize gambar

```python
resize_img = cv2.resize(img, (width_baru, height_baru))
```

**Penjelasan:**

* `cv2.resize()` digunakan untuk **mengubah ukuran gambar**.
* `(width_baru, height_baru)` menentukan ukuran baru dalam piksel.

**Essay:**

> Sintaks ini digunakan untuk memperkecil atau memperbesar gambar agar ukurannya sesuai kebutuhan pemrosesan.

---

## c) Rotasi citra sederhana

```python
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
```

**Penjelasan:**

* `cv2.rotate()` digunakan untuk memutar gambar.
* `cv2.ROTATE_90_CLOCKWISE` berarti diputar 90 derajat searah jarum jam. 

---

## d) Flipping citra

```python
flip_h = cv2.flip(img, 1)
flip_v = cv2.flip(img, 0)
```

**Penjelasan:**

* `cv2.flip(img, 1)` → membalik gambar secara horizontal.
* `cv2.flip(img, 0)` → membalik gambar secara vertikal.

**Essay:**

> Fungsi flip digunakan untuk merefleksikan citra terhadap sumbu tertentu, baik horizontal maupun vertikal.

---

## e) Konversi grayscale

```python
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Penjelasan:**

* `cv2.cvtColor()` dipakai untuk mengubah format warna.
* `cv2.COLOR_BGR2GRAY` berarti gambar berwarna BGR diubah menjadi grayscale.
* Setelah grayscale, gambar dari 3 channel menjadi 1 channel intensitas abu-abu. 

**Essay:**

> Sintaks ini digunakan untuk mengubah citra berwarna menjadi citra grayscale agar proses analisis lebih sederhana karena hanya memiliki satu channel intensitas.

---

## f) Menyimpan gambar

```python
cv2.imwrite('gambar.png', gray_image)
```

**Penjelasan:**

* `cv2.imwrite()` digunakan untuk menyimpan gambar hasil pemrosesan ke file baru.
* `'gambar.png'` adalah nama file output. 

---

## g) Penjumlahan dan pengurangan citra

```python
added_image = cv2.add(image1, image2)
subtracted_image = cv2.subtract(image1, image2)
```

**Penjelasan:**

* `cv2.add()` menjumlahkan nilai piksel dua gambar.
  Hasilnya biasanya membuat gambar lebih terang.
* `cv2.subtract()` mengurangi nilai piksel gambar pertama dengan gambar kedua.
  Hasilnya bisa membuat citra lebih gelap atau menonjolkan perbedaan. 

---

## h) Perkalian citra dengan skalar

```python
multiplied_image = cv2.multiply(img, 1.5)
```

**Penjelasan:**

* `cv2.multiply()` mengalikan setiap nilai piksel dengan angka tertentu.
* `1.5` berarti intensitas piksel dinaikkan 1,5 kali.
* Biasanya gambar jadi lebih terang. 

---

## i) Atur kecerahan dan kontras

```python
alpha = 1.5
beta = 10
adjusted_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
```

**Penjelasan:**

* `cv2.convertScaleAbs()` dipakai buat mengatur intensitas piksel.
* `alpha` mengontrol **kontras**.
* `beta` mengontrol **kecerahan**.
* Semakin besar `alpha`, perbedaan gelap-terang makin tegas.
* Semakin besar `beta`, gambar makin terang. 

---

## j) Gaussian Blur

```python
gambar_blur = cv2.GaussianBlur(img, (5, 5), 0)
```

**Penjelasan:**

* `cv2.GaussianBlur()` digunakan untuk mengaburkan gambar.
* `(5,5)` adalah ukuran kernel blur.
* Tujuan blur: mengurangi noise dan detail halus. 

---

## k) Canny edge detection

```python
canny = cv2.Canny(img, 100, 200)
```

**Penjelasan:**

* `cv2.Canny()` digunakan untuk mendeteksi tepi objek.
* `100` dan `200` adalah threshold bawah dan atas.
* Output biasanya berupa garis tepi putih di latar gelap. 

---

## l) Scaling dengan transformasi

```python
M_scaling = np.float32([[scaling_x, 0, 0], [0, scaling_y, 0]])
gambar_scaling = cv2.warpAffine(img, M_scaling, (int(lebar * scaling_x), int(tinggi * scaling_y)))
```

**Penjelasan:**

* `M_scaling` adalah matriks transformasi scaling.
* `cv2.warpAffine()` menerapkan transformasi ke gambar.
* Hasilnya ukuran gambar berubah sesuai skala. 

---

## m) Rotasi dengan matriks

```python
M_rotasi = cv2.getRotationMatrix2D(titik_rotasi, sudut_rotasi, 1)
gambar_rotasi = cv2.warpAffine(img, M_rotasi, (lebar, tinggi))
```

**Penjelasan:**

* `cv2.getRotationMatrix2D()` membuat matriks rotasi.
* `titik_rotasi` biasanya titik tengah gambar.
* `sudut_rotasi` menentukan derajat putaran.
* `1` berarti skala tetap.
* `cv2.warpAffine()` menjalankan rotasi tersebut. 

---

## n) Translasi

```python
M = np.float32([[1, 0, 50], [0, 1, 30]])
gambar_translasi = cv2.warpAffine(img, M, (lebar, tinggi))
```

**Penjelasan:**

* Matriks ini menggeser gambar:

  * `50` piksel ke kanan
  * `30` piksel ke bawah
* Translasi adalah pergeseran posisi gambar tanpa mengubah bentuknya. 

---

## o) Histogram grayscale

```python
histogram = cv2.calcHist([gambar], [0], None, [256], [0, 256])
```

**Penjelasan:**

* `cv2.calcHist()` digunakan untuk menghitung histogram citra.
* `[0]` berarti channel grayscale.
* `[256]` berarti jumlah bin dari 0 sampai 255.
* Histogram menunjukkan distribusi intensitas piksel. 

**Essay:**

> Histogram digunakan untuk menganalisis sebaran intensitas piksel sehingga bisa membantu memahami tingkat kecerahan dan kontras citra.

---

# 3. LKM 2 — PREPROCESSING DATASET FOTO

## a) Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Penjelasan:**

* Digunakan untuk menghubungkan Google Drive ke Google Colab.
* Setelah di-mount, file di Drive bisa diakses lewat path `/content/drive`. 

---

## b) Copy dataset

```python
import shutil

source = "/content/drive/MyDrive/DatasetArtis"
destination = "/content/drive/MyDrive/Dataset_Praktikum_Artis"

shutil.copytree(source, destination, dirs_exist_ok=True)
```

**Penjelasan:**

* `import shutil` memanggil library untuk operasi file/folder.
* `source` = folder asal dataset.
* `destination` = folder tujuan.
* `shutil.copytree()` menyalin seluruh isi folder dataset ke lokasi baru.
* `dirs_exist_ok=True` artinya kalau folder tujuan sudah ada, proses tetap jalan. 

---

## c) Analisis ukuran gambar

```python
import os
import cv2

dataset_path = "/content/drive/MyDrive/Dataset_Praktikum_Artis"
sizes = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(root, file))
            if img is not None:
                h, w = img.shape[:2]
                sizes.append((w, h))

print("Jumlah gambar:", len(sizes))
print("Contoh ukuran:", sizes[:10])
```

**Penjelasan per sintaks:**

* `os.walk(dataset_path)`
  Menelusuri semua folder dan subfolder dalam dataset.
* `file.lower().endswith(...)`
  Memilih file yang berekstensi gambar.
* `cv2.imread(...)`
  Membaca gambar.
* `img.shape[:2]`
  Mengambil tinggi (`h`) dan lebar (`w`) gambar.
* `sizes.append((w, h))`
  Menyimpan ukuran gambar ke list.
* `len(sizes)`
  Menampilkan jumlah gambar yang berhasil dibaca. 

**Essay:**

> Kode ini digunakan untuk mengecek ukuran gambar pada dataset. Program membaca semua file gambar, mengambil tinggi dan lebar masing-masing citra, lalu menyimpannya agar bisa diketahui apakah ukuran gambar pada dataset sudah seragam atau masih bervariasi.

---

## d) Resize semua gambar jadi 224x224

```python
input_folder = "/content/drive/MyDrive/Dataset_Praktikum_Artis"
output_folder = "/content/drive/MyDrive/Dataset_Resized"

target_size = (224, 224)

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_resized = cv2.resize(img, target_size)

            relative_path = os.path.relpath(root, input_folder)
            save_dir = os.path.join(output_folder, relative_path)
            os.makedirs(save_dir, exist_ok=True)

            cv2.imwrite(os.path.join(save_dir, file), img_resized)
```

**Penjelasan:**

* `target_size = (224, 224)`
  Menentukan ukuran standar semua gambar.
* `cv2.resize(img, target_size)`
  Mengubah ukuran gambar.
* `os.path.relpath(root, input_folder)`
  Mengambil struktur folder relatif dari folder asal.
* `os.makedirs(save_dir, exist_ok=True)`
  Membuat folder tujuan jika belum ada.
* `cv2.imwrite(...)`
  Menyimpan hasil resize. 

**Essay:**

> Kode ini digunakan untuk standarisasi ukuran citra menjadi 224 × 224 piksel. Hal ini penting agar semua gambar memiliki ukuran yang sama sebelum digunakan pada proses training model.

---

## e) Haar Cascade face detection + cropping

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```

**Penjelasan:**

* `cv2.CascadeClassifier(...)` membuat objek detektor wajah.
* File XML berisi model Haar Cascade bawaan OpenCV untuk deteksi wajah depan. 

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

* Mengubah gambar ke grayscale agar deteksi lebih ringan dan cepat.

```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)
```

**Penjelasan:**

* `detectMultiScale()` mencari wajah dalam gambar.
* `scaleFactor=1.3` mengatur skala pencarian objek.
* `minNeighbors=5` mengatur ketelitian deteksi agar false detection berkurang. 

```python
for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224,224))
```

**Penjelasan:**

* `(x, y, w, h)` adalah koordinat kotak wajah:

  * `x, y` titik awal
  * `w` lebar
  * `h` tinggi
* `img[y:y+h, x:x+w]` melakukan **cropping** area wajah.
* Lalu wajah hasil crop di-resize jadi 224x224. 

**Essay lengkap:**

> Kode ini digunakan untuk mendeteksi wajah pada setiap gambar menggunakan algoritma Haar Cascade. Setelah wajah ditemukan, area wajah dipotong berdasarkan koordinat deteksi, kemudian diubah ukurannya menjadi 224 × 224 piksel agar seragam dan siap dipakai pada proses klasifikasi.

---

# 4. LKM 3 — CNN BASELINE DAN TRANSFER LEARNING

## a) Preprocessing dataset

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

**Penjelasan:**

* `ImageDataGenerator` digunakan untuk menyiapkan dataset gambar sebelum training. 

```python
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```

**Penjelasan:**

* `rescale=1./255`
  Menormalkan nilai piksel dari rentang 0–255 menjadi 0–1.
* `validation_split=0.2`
  Membagi 20% data untuk validasi dan 80% untuk training. 

---

## b) Load dataset dari folder

```python
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

**Penjelasan:**

* `flow_from_directory()`
  Membaca dataset gambar yang disusun per folder kelas.
* `target_size=(224,224)`
  Semua gambar disesuaikan ke ukuran 224x224.
* `batch_size=32`
  Sekali training, model membaca 32 gambar.
* `class_mode='categorical'`
  Label dibuat dalam bentuk one-hot encoding karena klasifikasi multi-kelas.
* `subset='training'` / `'validation'`
  Menentukan apakah data untuk training atau validasi. 

---

## c) Membuat model CNN baseline

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```

**Penjelasan:**

* `Sequential` → model layer berurutan.
* `Conv2D` → layer konvolusi untuk ekstraksi fitur gambar.
* `MaxPooling2D` → mengurangi ukuran fitur map.
* `Flatten` → mengubah data 2D/3D jadi 1D.
* `Dense` → fully connected layer untuk klasifikasi. 

```python
baseline_model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128,activation='relu'),
    Dense(train_data.num_classes,activation='softmax')
])
```

**Penjelasan per layer:**

* `Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3))`
  Layer konvolusi pertama dengan 32 filter ukuran 3x3.
  `relu` dipakai sebagai fungsi aktivasi.
  `input_shape=(224,224,3)` berarti input gambar berukuran 224x224 dengan 3 channel warna.
* `MaxPooling2D(2,2)`
  Mengecilkan ukuran feature map agar komputasi lebih ringan.
* `Conv2D(64,(3,3),activation='relu')`
  Layer konvolusi kedua dengan 64 filter.
* `Flatten()`
  Mengubah output feature map menjadi vektor 1 dimensi.
* `Dense(128, activation='relu')`
  Layer dense untuk belajar pola klasifikasi.
* `Dense(train_data.num_classes, activation='softmax')`
  Layer output sesuai jumlah kelas. Softmax dipakai untuk klasifikasi multi-kelas. 

---

## d) Compile model

```python
baseline_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Penjelasan:**

* `optimizer='adam'`
  Algoritma untuk memperbarui bobot model.
* `loss='categorical_crossentropy'`
  Fungsi loss untuk klasifikasi multi-kelas.
* `metrics=['accuracy']`
  Metrik evaluasi yang dipakai adalah akurasi. 

---

## e) Training model

```python
history_baseline = baseline_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
```

**Penjelasan:**

* `.fit()` digunakan untuk melatih model.
* `train_data` adalah data latih.
* `validation_data=val_data` dipakai untuk mengecek performa tiap epoch.
* `epochs=10` berarti pelatihan dilakukan 10 kali putaran penuh terhadap dataset. 

---

## f) Transfer learning MobileNetV2

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
```

**Penjelasan:**

* `MobileNetV2` adalah model pretrained.
* `GlobalAveragePooling2D` merangkum feature map jadi vektor.
* `Model` dipakai untuk membangun model baru berbasis model pretrained. 

```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
```

**Penjelasan:**

* `weights='imagenet'` → memakai bobot hasil training dari dataset ImageNet.
* `include_top=False` → layer klasifikasi bawaan dihapus karena mau disesuaikan dengan dataset sendiri.
* `input_shape=(224,224,3)` → ukuran input gambar. 

```python
for layer in base_model.layers:
    layer.trainable = False
```

**Penjelasan:**

* Semua layer pada base model dibekukan.
* Artinya bobot awal tidak diubah saat training.
* Tujuannya agar model memanfaatkan fitur umum yang sudah dipelajari sebelumnya. 

```python
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

transfer_model = Model(inputs=base_model.input, outputs=predictions)
```

**Penjelasan:**

* `base_model.output` mengambil output fitur dari MobileNetV2.
* `GlobalAveragePooling2D()` meratakan fitur dengan cara rata-rata global.
* `Dense(..., activation='softmax')` jadi layer klasifikasi akhir.
* `Model(...)` menyusun model transfer learning baru. 

---

## g) Compile dan train transfer model

```python
transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_transfer = transfer_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
```

**Penjelasan:**
Sama seperti baseline, cuma model yang dipakai adalah model transfer learning. 

---

## h) Plot perbandingan akurasi

```python
plt.plot(history_baseline.history['accuracy'])
plt.plot(history_transfer.history['accuracy'])
plt.legend(['Baseline', 'Transfer Learning'])
plt.title("Perbandingan Accuracy")
plt.show()
```

**Penjelasan:**

* `history_baseline.history['accuracy']` mengambil riwayat akurasi model baseline.
* `history_transfer.history['accuracy']` mengambil riwayat akurasi model transfer learning.
* `plt.legend(...)` memberi keterangan garis.
* `plt.title(...)` memberi judul grafik.
* `plt.show()` menampilkan grafik. 

**Essay:**

> Grafik ini digunakan untuk membandingkan performa akurasi antara model baseline CNN dan model transfer learning selama proses training.

---

# 5. LKM 4 — PROJECT FACE RECOGNITION

## a) Import utama project

```python
import cv2
import csv
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from camera_utils import open_camera
from attendance import mark_attendance
from train import train_model
```

**Penjelasan:**

* `cv2` untuk computer vision.
* `csv` untuk baca/tulis file user dan absensi.
* `os` untuk file/folder.
* `tkinter` untuk GUI.
* `PIL` untuk menampilkan frame kamera di GUI.
* `open_camera` untuk membuka webcam.
* `mark_attendance` untuk simpan absensi.
* `train_model` untuk melatih ulang model pengenalan wajah. 

---

## b) Konstanta file utama

```python
USERS_FILE = "users.csv"
DATASET_DIR = "dataset"
MODEL_PATH = "models/trainer.yml"
```

**Penjelasan:**

* `USERS_FILE` → file data user
* `DATASET_DIR` → folder wajah user
* `MODEL_PATH` → file model hasil training 

---

## c) Membaca model pengenalan

```python
self.recognizer = cv2.face.LBPHFaceRecognizer_create()
self.recognizer.read(MODEL_PATH)
```

**Penjelasan:**

* `LBPHFaceRecognizer_create()` membuat model face recognition metode **LBPH**.
* `read(MODEL_PATH)` memuat model yang sudah pernah dilatih. 

**Essay:**

> Sintaks ini digunakan untuk membuat dan memuat model pengenalan wajah sehingga sistem dapat mengenali user berdasarkan dataset yang sudah dilatih sebelumnya.

---

## d) Deteksi wajah di kamera

```python
faces = self.detector.detectMultiScale(gray, 1.3, 5)
```

**Penjelasan:**

* `gray` = frame grayscale
* `1.3` = scale factor
* `5` = minimum neighbors
* Hasilnya koordinat wajah yang terdeteksi. 

---

## e) Prediksi identitas wajah

```python
user_id, confidence = self.recognizer.predict(face)
```

**Penjelasan:**

* `predict(face)` menebak wajah itu milik user mana.
* `user_id` = ID hasil prediksi.
* `confidence` = tingkat keyakinan model.
* Dalam kode, jika `confidence < 70`, wajah dianggap cukup cocok. 

**Essay:**

> Fungsi predict digunakan untuk mencocokkan wajah input dengan model yang telah dilatih. Hasilnya berupa ID user dan nilai confidence. Semakin baik kecocokannya, sistem dapat mengenali identitas user dengan lebih akurat.

---

## f) Simpan absensi

```python
success = mark_attendance(user_id, nama, nim)
```

**Penjelasan:**

* Memanggil fungsi untuk mencatat user ke file absensi.
* Biasanya dicek agar user tidak absen dua kali di hari yang sama. 

---

## g) Register user baru

```python
nama = simpledialog.askstring("Register", "Masukkan Nama:", parent=parent)
nim = simpledialog.askstring("Register", "Masukkan NIM:", parent=parent)
```

**Penjelasan:**

* Memunculkan kotak dialog input untuk nama dan NIM user. 

```python
user_id = get_next_user_id()
folder_name = f"user_{user_id:03d}"
user_folder = os.path.join(DATASET_DIR, folder_name)
os.makedirs(user_folder, exist_ok=True)
```

**Penjelasan:**

* `get_next_user_id()` mengambil ID berikutnya.
* `f"user_{user_id:03d}"` membentuk nama folder seperti `user_001`.
* `os.makedirs()` membuat folder penyimpanan dataset user. 

---

## h) Simpan hasil capture wajah

```python
face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
img_path = os.path.join(user_folder, f"{count_ref[0] + 1}.jpg")
cv2.imwrite(img_path, face_img)
```

**Penjelasan:**

* Wajah dipotong dari frame.
* Diubah ukurannya jadi 200x200.
* Disimpan ke file JPG dalam folder user. 

---

## i) Tulis user ke CSV

```python
with open(USERS_FILE, "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([user_id, nama, nim, folder_name])
```

**Penjelasan:**

* Membuka `users.csv` dalam mode append (`"a"`).
* `writer.writerow(...)` menambahkan data user baru ke file. 

---

## j) Training model wajah

```python
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
```

**Penjelasan:**

* Membuat model LBPH.
* Menyiapkan list untuk data wajah dan label ID. 

```python
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
faces.append(img)
labels.append(user_id)
```

**Penjelasan:**

* Membaca gambar wajah dalam grayscale.
* Menambahkan gambar ke data training.
* Menambahkan ID user ke label. 

```python
recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_PATH)
```

**Penjelasan:**

* `train()` melatih model dari dataset wajah dan label.
* `save()` menyimpan model ke file `.yml`. 

---

## k) Attendance.py

```python
today = datetime.now().strftime("%Y-%m-%d")
now_time = datetime.now().strftime("%H:%M:%S")
```

**Penjelasan:**

* Mengambil tanggal dan jam saat ini untuk dicatat pada absensi. 

```python
if row["id"] == str(user_id) and row["tanggal"] == today:
    return False
```

**Penjelasan:**

* Mengecek apakah user dengan ID tersebut sudah absen hari ini.
* Kalau sudah, fungsi mengembalikan `False`, artinya absensi tidak dicatat ulang. 

```python
writer.writerow([user_id, nama, nim, today, now_time, status])
```

**Penjelasan:**

* Menulis data absensi baru ke file CSV. 

---

## l) Camera utils

```python
cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
```

**Penjelasan:**

* Membuka kamera dengan backend DirectShow.
* Dipakai supaya webcam lebih mudah terbaca di Windows. 

```python
if cap.isOpened() and _is_frame_valid(cap):
    return cap
```

**Penjelasan:**

* Mengecek apakah kamera berhasil dibuka dan frame yang dihasilkan valid. 

---

# 6. KALIMAT AMAN BUAT ESSAY

Kalau lo lupa detail, pakai pola ini:

**“Sintaks ini digunakan untuk …”**
lalu lanjutkan dengan:

1. input apa yang dipakai,
2. proses apa yang dilakukan,
3. output apa yang dihasilkan.

Contoh:

> Sintaks `cv2.imread()` digunakan untuk membaca file gambar dari penyimpanan ke dalam program. Hasil pembacaan disimpan dalam variabel berupa array citra yang kemudian dapat diproses lebih lanjut.

> Sintaks `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` digunakan untuk mengubah citra berwarna menjadi grayscale agar proses pengolahan lebih sederhana karena hanya memiliki satu channel intensitas.

> Sintaks `model.fit()` digunakan untuk melatih model menggunakan data training selama sejumlah epoch tertentu, sekaligus memantau performa model pada data validasi.

---

# 7. YANG PALING MUNGKIN KELUAR DI UTS

Fokus hafalin ini:

* `cv2.imread`
* `cv2.imshow`, `cv2.waitKey`, `cv2.destroyAllWindows`
* `img.shape`
* akses piksel `img[x,y]`
* `cv2.cvtColor`
* `cv2.resize`
* `cv2.add`, `cv2.subtract`, `cv2.multiply`
* `cv2.convertScaleAbs`
* `cv2.GaussianBlur`
* `cv2.Canny`
* `cv2.calcHist`
* `os.walk`
* `ImageDataGenerator`
* `flow_from_directory`
* `Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`
* `compile`, `fit`
* `MobileNetV2`
* `CascadeClassifier`, `detectMultiScale`
* `LBPHFaceRecognizer_create`, `predict`, `train`, `save`

Kalau mau, next chat gue bisa bikinin **versi super rapih per LKM**:
**LKM 1, LKM 2, LKM 3, LKM 4** masing-masing jadi **catatan hafalan essay** yang tinggal lo pelajari.
