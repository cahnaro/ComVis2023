1. persiapan
- install sk learn module dengan cara ketik pip install -U scikit-learn pada windows powershell
- install pyton 3.10 (64bit)

2. run program
- buka pyton 3.10
- pilih file->open->pilih 04220061C-Identifikasi-citra-metode-SVM.py
- setelah file berhasil dibuka, pilih run->run module. atau bisa juga run dengan shortcut f5

3. penjelasan singkat kodingan
- Pertama-tama kita memuat dataset iris dari liblary Scikit-Learn dan memilih hanya dua fitur data sampel dataset tersebut (panjang & lebar).
- Selanjutnya dataset dibagi menjadi dua bagian yaitu training set dan testing set menggunakan train_test_split dari Scikit-Learn.
- Kemudian membuat model SVM dengan kernel linier menggunakan svm.SVC dari Scikit-Learn dan melatih model tersebut menggunakan training set.
- Setelah model terlatih, maka sistem melakukan prediksi menggunakan testing set dan menghitung akurasi menggunakan accuracy_score dari Scikit-Learn.

4. note
- menggunakan pyton 3.10 (64 bit)
- menggunakan sk learn module 1.2.2
- dataset iris adalah kumpulan data dari pengukuran kelopak bunga