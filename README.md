# Laporan Proyek Machine Learning - Klasifikasi Citra Buah Menggunakan Convolutional Neural Network (CNN)

## Project Overview
Proyek ini berfokus pada pengembangan model *Deep Learning* menggunakan arsitektur *Convolutional Neural Network* (CNN) untuk mengklasifikasikan berbagai jenis buah berdasarkan citra visual. Dataset yang digunakan adalah **Fruits-360** yang mencakup lebih dari 130 kelas buah. Selain pembangunan dan pelatihan model, proyek ini juga mencakup proses konversi model ke berbagai format deployment, yaitu TensorFlow SavedModel, TensorFlow Lite (TFLite), dan TensorFlow.js (TFJS).

## Business Understanding

### Problem Statements
- Bagaimana membangun model visi komputer (*computer vision*) yang mampu mengenali dan mengklasifikasikan ratusan kategori buah dengan tingkat akurasi tinggi?
- Bagaimana cara mengekspor dan mengonversi model CNN yang telah dilatih agar dapat diintegrasikan ke dalam berbagai platform (seluler, web, dan server)?

### Goals
- Mengembangkan arsitektur CNN yang handal untuk klasifikasi multi-kelas citra buah.
- Melakukan evaluasi performa model menggunakan metrik akurasi dan *loss* pada data uji.
- Mengonversi model akhir ke format TFLite dan TFJS untuk kebutuhan *edge deployment*.

### Solution Approach
- **Prapemrosesan Data**: Menggabungkan direktori pelatihan dan pengujian bawaan dataset, melakukan *stratified split* ulang menjadi data latih, validasi, dan pengujian.
- **Arsitektur CNN**: Membangun model *Sequential* yang terdiri dari lapisan konveksi (*Conv2D*), *Max Pooling*, *Flatten*, dan *Dense Layers* dengan fungsi aktivasi *softmax* pada lapisan keluaran (*output layer*).
- **Model Export & Conversion**: Menyimpan model ke format SavedModel, mengonversinya menjadi format `.tflite`, serta menggunakan `tensorflowjs_converter` untuk format web.

## Data Understanding
Dataset yang digunakan adalah **Fruits-360 dataset** dari repositori publik Horea94.
- **Jumlah Kelas**: 131 kategori buah dan sayur[cite: 11].
- **Resolusi Gambar**: Disamakan ke ukuran standar 100x100 piksel dengan kanal warna RGB[cite: 11].
- **Distribusi Data**: Data digabungkan dan dibagi ulang menjadi set *train* (63.262 gambar), *validation* (13.556 gambar), dan *test* (13.557 gambar)[cite: 11].

## Data Preparation
Tahapan persiapan data meliputi:
- **Penggabungan Dataset**: Menyatukan folder *Training* dan *Test* bawaan dataset ke dalam satu direktori terpusat (*merged_dataset*).
- **Pembagian Ulang (*Stratified Split*)**: Membagi data secara proporsional menggunakan proporsi 70% untuk pelatihan, 15% untuk validasi, dan 15% untuk pengujian dengan menjaga keseimbangan distribusi kelas (*stratify*)[cite: 11].
- **Normalisasi Citra (*Rescaling*)**: Menerapkan *data generator* dengan membagi nilai piksel dengan 255 (`rescale=1./255`) agar berada pada rentang 0 hingga 1[cite: 11].

## Modeling
Model CNN dibangun menggunakan kerangka kerja TensorFlow/Keras dengan rincian arsitektur sebagai berikut:
1. **Conv2D (32 filter, kernel 3x3, ReLU) + MaxPooling2D**[cite: 11]
2. **Conv2D (64 filter, kernel 3x3, ReLU) + MaxPooling2D**[cite: 11]
3. **Conv2D (128 filter, kernel 3x3, ReLU) + MaxPooling2D**[cite: 11]
4. **Flatten Layer**[cite: 11]
5. **Dense Layer (128 neuron, ReLU)**[cite: 11]
6. **Output Dense Layer (131 neuron, Softmax)** untuk klasifikasi multi-kelas[cite: 11].

Model dikompilasi menggunakan *optimizer* `adam`, fungsi kerugian `categorical_crossentropy`, dan metrik `accuracy`[cite: 11]. Pelatihan dijalankan selama 10 *epoch*[cite: 11].

## Evaluation
Model dievaluasi menggunakan data pengujian (*test_gen*)[cite: 11]:
- **Akurasi Pengujian (Test Accuracy)**: Mencapai **99.96%**, yang menunjukkan bahwa model mampu mempelajari pola visual dari masing-masing kelas buah dengan sangat baik[cite: 11].
- **Grafik Pelatihan**: Visualisasi kurva *Accuracy* dan *Loss* menunjukkan konvergensi yang stabil antara data latih dan data validasi tanpa indikasi *overfitting* yang signifikan[cite: 11].

## Deployment & Export Model
Model akhir telah diekspor ke dalam tiga format utama untuk mendukung berbagai platform:
1. **SavedModel (`model_saved`)**: Format standar TensorFlow[cite: 11].
2. **TensorFlow Lite (`model.tflite`)**: Dioptimalkan untuk aplikasi perangkat seluler atau *edge device*[cite: 11].
3. **TensorFlow.js (`tfjs_model`)**: Dikonversi agar model dapat langsung berjalan di peramban web (*browser*)[cite: 11].

## Setup Environment & Cara Menjalankan
1. Pasang dependensi utama yang diperlukan:
   ```bash
   pip install tensorflow tensorflowjs tqdm scikit-learn matplotlib pandas
   ```
2. Jalankan sel kode secara berurutan di dalam lingkungan Jupyter Notebook atau Google Colab untuk mengunduh dataset, melatih model, hingga mengekspor berkas model.

## Kontributor
- Wiraldy Manggala Simanjuntak
