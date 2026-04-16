Oke, paham now.

Lu maunya **LKM 4 juga dibikinin model kayak jawaban pertama**:

* ada **potongan sintaks**
* terus **penjelasan per sintaks**
* gaya bahasa **buat hafalan essay**
* bukan narasi panjang

Jadi gue bikinin **LKM 4 versi “kayak yang pertama”** ya.

---

# 5. LKM 4 — PROJECT COMPUTER VISION

Karena LKM 4 isinya project tiap kelompok beda-beda, gue bikinin yang **inti sintaks paling sering muncul** dari file kelompok yang lu upload:

* **Haar Cascade**
* **LBPH Face Recognition**
* **Capture dataset dari webcam**
* **Training model**
* **Recognition real-time**
* tambahan **versi web Flask**
* tambahan **versi HOG + SVM** buat kelompok yang deteksi kantuk

Jadi ini aman buat lu masukin GitHub juga.

---

# A. LKM 4 — FACE DETECTION / FACE RECOGNITION (UMUM)

## a) Import library dasar

```python
import cv2
import os
import numpy as np
```

**Penjelasan:**

* `import cv2`
  Digunakan untuk memanggil library OpenCV yang berfungsi untuk membaca gambar, membuka kamera, mendeteksi wajah, dan melakukan face recognition.
* `import os`
  Digunakan untuk mengakses folder dan file, misalnya membuat folder dataset, membaca isi folder, dan menggabungkan path file.
* `import numpy as np`
  Digunakan untuk mengolah data numerik dalam bentuk array, terutama saat proses training model.

**Kalimat essay:**

> Baris ini digunakan untuk mengimpor library utama yang dibutuhkan dalam project computer vision, yaitu OpenCV untuk pengolahan citra, os untuk pengelolaan file dan folder, serta NumPy untuk pengolahan data numerik.

---

## b) Membuat folder dataset

```python
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print("Folder dataset telah dibuat.")
```

**Penjelasan:**

* `os.path.exists('dataset')`
  Digunakan untuk mengecek apakah folder dataset sudah ada atau belum.
* `os.makedirs('dataset')`
  Digunakan untuk membuat folder dataset jika folder tersebut belum ada.
* `print(...)`
  Menampilkan pesan bahwa folder berhasil dibuat.

**Kalimat essay:**

> Sintaks ini digunakan untuk memastikan folder dataset tersedia sebelum proses penyimpanan gambar wajah dilakukan. Jika folder belum ada, maka program akan membuat folder baru secara otomatis.

---

## c) Membuka kamera

```python
cam = cv2.VideoCapture(0)
```

**Penjelasan:**

* `cv2.VideoCapture(0)`
  Digunakan untuk mengakses webcam utama pada komputer.
* Angka `0` menunjukkan index kamera default.

**Kalimat essay:**

> Sintaks ini digunakan untuk membuka webcam agar sistem dapat mengambil frame gambar secara real-time dari kamera.

---

## d) Load Haar Cascade

```python
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```

**Penjelasan:**

* `cv2.CascadeClassifier()`
  Digunakan untuk membuat objek detektor wajah.
* `cv2.data.haarcascades`
  Menunjukkan lokasi file XML bawaan OpenCV.
* `'haarcascade_frontalface_default.xml'`
  Merupakan file model Haar Cascade untuk mendeteksi wajah tampak depan.

**Kalimat essay:**

> Sintaks ini digunakan untuk memuat model Haar Cascade yang dipakai dalam proses deteksi wajah pada gambar atau frame kamera.

---

## e) Input ID user

```python
face_id = input("Masukkan ID user (angka): ").strip()
```

**Penjelasan:**

* `input(...)`
  Digunakan untuk meminta user memasukkan ID.
* `.strip()`
  Menghapus spasi kosong di awal atau akhir input.

```python
if face_id.isdigit():
    pass
else:
    print("Masukkan angka yang valid!")
```

**Penjelasan:**

* `isdigit()`
  Mengecek apakah input hanya berisi angka.
* Jika bukan angka, program menampilkan pesan error.

**Kalimat essay:**

> Sintaks ini digunakan untuk mengambil ID user dan memvalidasi agar input yang dimasukkan berupa angka, sehingga dapat digunakan sebagai label identitas saat training model.

---

## f) Input nama user

```python
face_name = input("Masukkan nama user: ").strip()
```

**Penjelasan:**

* Digunakan untuk mengambil nama user yang akan didaftarkan.

```python
if not face_name:
    face_name = f"User_{face_id}"
```

**Penjelasan:**

* Jika user tidak mengisi nama, sistem memberi nama default berdasarkan ID.

**Kalimat essay:**

> Sintaks ini digunakan untuk mengambil nama user sebagai identitas wajah. Jika nama tidak diisi, maka sistem memberikan nama default agar data tetap dapat disimpan.

---

## g) Menyimpan label ID dan nama

```python
with open("labels.txt", "a") as f:
    f.write(f"{face_id}|{face_name}\n")
```

**Penjelasan:**

* `with open(..., "a")`
  Membuka file `labels.txt` dalam mode append, artinya data baru ditambahkan tanpa menghapus data lama.
* `f.write(...)`
  Menyimpan pasangan ID dan nama ke file.
* Tanda `|` digunakan sebagai pemisah antara ID dan nama.

**Kalimat essay:**

> Sintaks ini digunakan untuk menyimpan mapping antara ID user dan nama user ke dalam file teks, sehingga hasil prediksi wajah nantinya dapat diterjemahkan menjadi nama.

---

## h) Membaca frame kamera

```python
ret, img = cam.read()
```

**Penjelasan:**

* `cam.read()`
  Mengambil satu frame dari kamera.
* `ret`
  Menunjukkan apakah frame berhasil dibaca atau tidak.
* `img`
  Menyimpan hasil gambar dari kamera.

```python
if not ret:
    print("Error: Tidak bisa membaca frame dari kamera!")
    break
```

**Penjelasan:**

* Jika `ret` bernilai `False`, artinya kamera gagal membaca frame.
* Program akan menampilkan pesan error lalu keluar dari loop.

**Kalimat essay:**

> Sintaks ini digunakan untuk mengambil frame gambar secara real-time dari kamera. Jika pengambilan gambar gagal, maka program menghentikan proses untuk mencegah error lanjutan.

---

## i) Membalik frame kamera

```python
img = cv2.flip(img, 1)
```

**Penjelasan:**

* `cv2.flip(img, 1)`
  Membalik gambar secara horizontal.
* Tujuannya agar tampilan kamera seperti cermin dan lebih nyaman bagi user.

**Kalimat essay:**

> Sintaks ini digunakan untuk membalik gambar kamera secara horizontal agar tampilan terlihat seperti cermin.

---

## j) Konversi ke grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Penjelasan:**

* `cv2.cvtColor()`
  Digunakan untuk mengubah format warna gambar.
* `cv2.COLOR_BGR2GRAY`
  Mengubah gambar berwarna BGR menjadi grayscale.

**Kalimat essay:**

> Sintaks ini digunakan untuk mengubah frame berwarna menjadi grayscale agar proses deteksi wajah menjadi lebih cepat dan efisien.

---

## k) Deteksi wajah

```python
faces = face_detector.detectMultiScale(gray, 1.3, 5)
```

**Penjelasan:**

* `detectMultiScale()`
  Digunakan untuk mendeteksi banyak wajah dalam gambar.
* `gray`
  Adalah input gambar grayscale.
* `1.3`
  Merupakan scale factor untuk skala pencarian wajah.
* `5`
  Menentukan jumlah minimum tetangga agar deteksi lebih akurat.

**Kalimat essay:**

> Sintaks ini digunakan untuk mendeteksi wajah pada frame grayscale menggunakan Haar Cascade, lalu menghasilkan koordinat area wajah yang ditemukan.

---

## l) Menggambar kotak wajah

```python
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

**Penjelasan:**

* `(x, y, w, h)`
  Menyimpan posisi dan ukuran wajah:

  * `x, y` = titik awal
  * `w` = lebar
  * `h` = tinggi
* `cv2.rectangle()`
  Menggambar kotak di sekitar wajah yang terdeteksi.
* `(255, 0, 0)`
  Warna kotak dalam format BGR.
* `2`
  Ketebalan garis kotak.

**Kalimat essay:**

> Sintaks ini digunakan untuk menandai area wajah yang terdeteksi dengan kotak pembatas agar posisi wajah terlihat jelas pada frame.

---

## m) Crop wajah dan simpan dataset

```python
cv2.imwrite(
    f"dataset/User.{face_id}.{count}.jpg",
    gray[y:y+h, x:x+w]
)
```

**Penjelasan:**

* `gray[y:y+h, x:x+w]`
  Mengambil hanya bagian wajah dari gambar grayscale.
* `cv2.imwrite()`
  Menyimpan hasil crop wajah ke file gambar.
* Nama file berisi ID user dan nomor urutan gambar.

**Kalimat essay:**

> Sintaks ini digunakan untuk memotong area wajah dari frame, lalu menyimpannya ke folder dataset sebagai data training.

---

## n) Menampilkan jumlah gambar yang berhasil diambil

```python
count += 1
print(f"Gambar {count} berhasil diambil")
```

**Penjelasan:**

* `count += 1`
  Menambah jumlah gambar yang berhasil diambil.
* `print(...)`
  Menampilkan informasi progres capture dataset.

**Kalimat essay:**

> Sintaks ini digunakan untuk menghitung jumlah dataset wajah yang berhasil disimpan dan menampilkan progres pengambilan data.

---

## o) Menampilkan frame kamera

```python
cv2.imshow("Dataset Capture", img)
```

**Penjelasan:**

* Menampilkan frame kamera yang sedang diproses.

```python
k = cv2.waitKey(10) & 0xff
if k == 27:
    break
```

**Penjelasan:**

* `cv2.waitKey(10)`
  Menunggu input keyboard selama 10 ms.
* `27`
  Adalah kode tombol ESC.
* Jika ESC ditekan, proses dihentikan.

**Kalimat essay:**

> Sintaks ini digunakan untuk menampilkan hasil kamera secara real-time dan memberi kesempatan kepada user untuk menghentikan proses dengan keyboard.

---

# B. TRAINING MODEL LBPH

## a) Membuat recognizer

```python
recognizer = cv2.face.LBPHFaceRecognizer_create()
```

**Penjelasan:**

* Membuat model pengenalan wajah dengan metode **LBPH** (Local Binary Pattern Histogram).

**Kalimat essay:**

> Sintaks ini digunakan untuk membuat model face recognition berbasis metode LBPH yang akan dilatih menggunakan dataset wajah.

---

## b) Menyiapkan path dataset dan model

```python
path = 'dataset'
trainer_path = 'trainer'
```

**Penjelasan:**

* `path` menunjukkan folder dataset wajah.
* `trainer_path` menunjukkan folder penyimpanan model hasil training.

---

## c) Membuat folder trainer

```python
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)
```

**Penjelasan:**

* Mengecek dan membuat folder `trainer` jika belum ada.

---

## d) Membaca semua gambar dataset

```python
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
```

**Penjelasan:**

* `os.listdir(path)`
  Mengambil semua nama file dalam folder dataset.
* `os.path.join(path, f)`
  Menggabungkan nama folder dan file menjadi path lengkap.
* Hasilnya disimpan di `imagePaths`.

**Kalimat essay:**

> Sintaks ini digunakan untuk mengambil seluruh file gambar pada dataset agar dapat diproses satu per satu saat training model.

---

## e) Mengubah gambar ke grayscale NumPy array

```python
PIL_img = Image.open(imagePath).convert('L')
img_numpy = np.array(PIL_img, 'uint8')
```

**Penjelasan:**

* `Image.open(imagePath)`
  Membuka file gambar.
* `.convert('L')`
  Mengubah gambar menjadi grayscale.
* `np.array(..., 'uint8')`
  Mengubah gambar menjadi array NumPy agar bisa diproses OpenCV.

**Kalimat essay:**

> Sintaks ini digunakan untuk membaca gambar dari dataset, mengubahnya menjadi grayscale, lalu mengonversinya ke array NumPy sebagai input training model.

---

## f) Mengambil ID dari nama file

```python
id = int(os.path.split(imagePath)[-1].split(".")[1])
```

**Penjelasan:**

* `os.path.split(imagePath)[-1]`
  Mengambil nama file saja.
* `.split(".")[1]`
  Mengambil bagian ID dari nama file.
* `int(...)`
  Mengubah ID menjadi bilangan bulat.

**Kalimat essay:**

> Sintaks ini digunakan untuk mengambil label ID user dari nama file gambar, sehingga setiap gambar wajah memiliki identitas yang sesuai.

---

## g) Deteksi wajah dalam dataset

```python
faces = detector.detectMultiScale(img_numpy)
```

**Penjelasan:**

* Mendeteksi area wajah pada gambar dataset sebelum digunakan untuk training.

---

## h) Menyimpan wajah dan label

```python
for (x, y, w, h) in faces:
    faceSamples.append(img_numpy[y:y+h, x:x+w])
    ids.append(id)
```

**Penjelasan:**

* Bagian wajah dipotong dari gambar.
* `faceSamples` menyimpan data wajah.
* `ids` menyimpan label identitas untuk tiap wajah.

**Kalimat essay:**

> Sintaks ini digunakan untuk mengekstraksi area wajah dari setiap gambar dataset dan menyimpan label ID yang sesuai sebagai data training.

---

## i) Melatih model

```python
recognizer.train(faceSamples, np.array(ids))
```

**Penjelasan:**

* `train()`
  Digunakan untuk melatih model LBPH menggunakan data wajah dan label ID.

---

## j) Menyimpan model

```python
recognizer.save('trainer/trainer.yml')
```

**Penjelasan:**

* Menyimpan hasil training ke file `trainer.yml`.

**Kalimat essay:**

> Sintaks ini digunakan untuk menyimpan model hasil training agar dapat digunakan kembali saat proses face recognition real-time.

---

# C. FACE RECOGNITION REAL-TIME

## a) Load model hasil training

```python
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
```

**Penjelasan:**

* Membuat recognizer LBPH.
* Memuat model yang sudah dilatih sebelumnya.

---

## b) Membaca labels.txt

```python
names = {0: "Unknown"}
with open("labels.txt", "r") as f:
    for line in f:
        line = line.strip()
        if "|" in line:
            id_str, name = line.split("|", 1)
            names[int(id_str)] = name
```

**Penjelasan:**

* `names = {0: "Unknown"}`
  Menyiapkan nama default untuk wajah yang tidak dikenali.
* `with open(...)`
  Membuka file labels.
* `split("|", 1)`
  Memisahkan ID dan nama.
* `names[int(id_str)] = name`
  Menyimpan pasangan ID dan nama ke dictionary.

**Kalimat essay:**

> Sintaks ini digunakan untuk membaca file label yang berisi pasangan ID dan nama, sehingga hasil prediksi model dapat ditampilkan dalam bentuk nama user.

---

## c) Prediksi wajah

```python
id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
```

**Penjelasan:**

* `predict(...)`
  Digunakan untuk menebak identitas wajah berdasarkan model hasil training.
* `id`
  Adalah ID hasil prediksi.
* `confidence`
  Menunjukkan tingkat kecocokan prediksi.

**Kalimat essay:**

> Sintaks ini digunakan untuk memprediksi identitas wajah berdasarkan area wajah yang telah dideteksi. Hasilnya berupa ID user dan nilai confidence sebagai tingkat kecocokan.

---

## d) Menentukan nama hasil prediksi

```python
if confidence < 70:
    name = names.get(id, "Unknown")
else:
    name = "Unknown"
```

**Penjelasan:**

* Jika nilai confidence cukup baik, nama diambil dari dictionary `names`.
* Jika tidak cocok, wajah diberi label `Unknown`.

**Kalimat essay:**

> Sintaks ini digunakan untuk menentukan apakah wajah dikenali atau tidak berdasarkan nilai confidence. Jika kecocokan baik, maka nama user ditampilkan, sedangkan jika tidak maka sistem menampilkan Unknown.

---

## e) Menampilkan nama di atas wajah

```python
cv2.putText(img, str(name), (x+5, y-5), font, 1, (255,255,255), 2)
```

**Penjelasan:**

* `cv2.putText()`
  Menampilkan teks pada gambar.
* `str(name)`
  Nama hasil prediksi.
* `(x+5, y-5)`
  Posisi teks di atas kotak wajah.

**Kalimat essay:**

> Sintaks ini digunakan untuk menampilkan nama hasil pengenalan wajah di atas area wajah pada frame kamera.

---

# D. VERSI PROJECT WEB FLASK (KELOMPOK 1)

## a) Import Flask

```python
from flask import Flask, render_template, request, Response, redirect, send_from_directory, flash
```

**Penjelasan:**

* `Flask` untuk membuat aplikasi web.
* `render_template` untuk menampilkan halaman HTML.
* `request` untuk mengambil input dari form.
* `Response` untuk mengirim stream video.
* `redirect` untuk pindah halaman.
* `send_from_directory` untuk menampilkan file dataset.
* `flash` untuk menampilkan notifikasi.

**Kalimat essay:**

> Sintaks ini digunakan untuk mengimpor komponen Flask yang diperlukan dalam membangun dashboard web untuk face detection dan face recognition.

---

## b) Membuat aplikasi Flask

```python
app = Flask(__name__)
app.secret_key = 'face_ai_project'
```

**Penjelasan:**

* `Flask(__name__)`
  Membuat instance aplikasi web Flask.
* `secret_key`
  Digunakan untuk fitur session dan flash message.

---

## c) Route halaman utama

```python
@app.route("/")
def index():
    images = os.listdir(DATASET)
    return render_template("index.html", images=images)
```

**Penjelasan:**

* `@app.route("/")`
  Menentukan route halaman utama.
* `os.listdir(DATASET)`
  Mengambil daftar file dataset.
* `render_template(...)`
  Menampilkan halaman HTML dan mengirim data gambar ke browser.

**Kalimat essay:**

> Sintaks ini digunakan untuk membuat halaman utama dashboard dan menampilkan daftar dataset yang tersedia.

---

## d) Stream video ke browser

```python
@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
```

**Penjelasan:**

* `Response(gen_frames(), ...)`
  Mengirim frame kamera ke browser secara streaming.
* `multipart/x-mixed-replace`
  Format yang digunakan agar video bisa tampil real-time di browser.

**Kalimat essay:**

> Sintaks ini digunakan untuk mengirim stream video kamera ke browser secara real-time sehingga user dapat melihat hasil deteksi langsung dari dashboard web.

---

# E. VERSI HOG + SVM (KELOMPOK 5)

## a) Import HOG dan SVM

```python
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
```

**Penjelasan:**

* `hog`
  Digunakan untuk ekstraksi fitur gambar.
* `train_test_split`
  Digunakan untuk membagi data training dan testing.
* `SVC`
  Digunakan untuk membuat model klasifikasi SVM.

**Kalimat essay:**

> Sintaks ini digunakan untuk menyiapkan proses ekstraksi fitur HOG dan klasifikasi menggunakan metode SVM.

---

## b) Resize dan ekstraksi fitur HOG

```python
img = cv2.resize(img, (64, 64))
features = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys'
)
```

**Penjelasan:**

* `cv2.resize(img, (64,64))`
  Menyamakan ukuran gambar.
* `hog(...)`
  Mengekstraksi fitur bentuk dan arah tepi dari gambar.
* `orientations=9`
  Menentukan jumlah arah gradien.
* `pixels_per_cell=(8,8)`
  Ukuran cell.
* `cells_per_block=(2,2)`
  Ukuran block.
* `block_norm='L2-Hys'`
  Metode normalisasi fitur.

**Kalimat essay:**

> Sintaks ini digunakan untuk menstandarkan ukuran citra dan mengekstraksi fitur HOG, yaitu fitur yang merepresentasikan pola tepi dan bentuk objek dalam gambar.

---

## c) Membagi data training dan testing

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Penjelasan:**

* `test_size=0.2`
  20% data dipakai untuk testing.
* `random_state=42`
  Agar pembagian data konsisten.
* `stratify=y`
  Agar distribusi kelas tetap seimbang.

---

## d) Training model SVM

```python
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
```

**Penjelasan:**

* `SVC(kernel='linear', probability=True)`
  Membuat model SVM dengan kernel linear.
* `model.fit(...)`
  Melatih model menggunakan data training.

**Kalimat essay:**

> Sintaks ini digunakan untuk membuat dan melatih model SVM agar dapat mengklasifikasikan data berdasarkan fitur HOG yang telah diekstraksi.

---

## e) Prediksi model

```python
y_pred = model.predict(X_test)
```

**Penjelasan:**

* Digunakan untuk memprediksi label data testing menggunakan model yang telah dilatih.

---

## f) Menghitung akurasi

```python
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
```

**Penjelasan:**

* `accuracy_score()`
  Menghitung tingkat ketepatan model antara label asli dan hasil prediksi.

---

# F. KALIMAT AMAN BUAT JAWAB ESSAY LKM 4

Kalau nanti di ujian keluar potongan kode LKM 4, lu bisa pakai pola ini:

> Sintaks ini digunakan untuk ...

Contoh:

* **`cv2.VideoCapture(0)`**

  > Sintaks ini digunakan untuk membuka webcam agar sistem dapat membaca frame kamera secara real-time.
* **`cv2.CascadeClassifier(...)`**

  > Sintaks ini digunakan untuk memuat model Haar Cascade yang dipakai dalam proses deteksi wajah.
* **`recognizer.train(faceSamples, np.array(ids))`**

  > Sintaks ini digunakan untuk melatih model LBPH menggunakan data wajah dan label identitas.
* **`recognizer.predict(...)`**

  > Sintaks ini digunakan untuk memprediksi identitas wajah berdasarkan model yang sudah dilatih sebelumnya.
* **`cv2.putText(...)`**

  > Sintaks ini digunakan untuk menampilkan informasi hasil deteksi atau pengenalan wajah pada frame gambar.

---

# G. YANG PALING PENTING DIHAFALIN DARI LKM 4

* `cv2.VideoCapture(0)`
* `cv2.CascadeClassifier(...)`
* `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`
* `detectMultiScale(...)`
* `cv2.rectangle(...)`
* `cv2.imwrite(...)`
* `cv2.face.LBPHFaceRecognizer_create()`
* `recognizer.train(...)`
* `recognizer.save(...)`
* `recognizer.read(...)`
* `recognizer.predict(...)`
* `cv2.putText(...)`
* `with open("labels.txt", "a")`
* `os.makedirs(...)`
* `train_test_split(...)`
* `hog(...)`
* `SVC(...)`

Kalau mau, habis ini gue bikinin **versi gabungan full dari LKM 1, 2, 3, 4** dengan format **persis kayak jawaban pertama**, jadi tinggal lu copas ke GitHub.
