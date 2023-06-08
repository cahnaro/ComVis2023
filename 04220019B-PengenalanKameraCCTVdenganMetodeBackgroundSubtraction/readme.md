Tugas MK Computer Vision 2023

Program ini adalah sebuah program yang digunakan untuk mengenali kamera CCTV dengan metode background subtraction. Metode ini memungkinkan pengguna untuk melakukan deteksi gerakan atau pelacakan objek pada suatu ruangan atau lokasi tertentu berdasarkan perbedaan antara frame pada video dengan background yang telah ditentukan.

Program ini menggunakan library OpenCV pada Python dan membuat objek dari class BackgroundSubtractorMOG2 untuk melakukan proses background subtraction. Program ini dapat membuka file video atau stream video dari kamera CCTV dan menampilkan hasil background subtraction pada setiap frame.

Requirements
Sebelum menjalankan program ini, pastikan bahwa komputer Anda telah terinstall Python dan library OpenCV.

Untuk menginstall OpenCV pada Python, jalankan perintah berikut di terminal:
- pip install opencv-python

Menjalankan Program
Download file programnya.
Untuk file vidio tidak diupload karena file terlalu besar, bisa diganti menggunakan vidio lain, dan diedit namanya pada bagian : 
- video = cv2.VideoCapture('Ketik_Nama_Vidio_Disini.ekstensi')
Buka terminal atau command prompt pada folder program.
Jalankan program dengan perintah:
- python PengenalanKameraCCTVdenganMetodeBackgroundSubtraction.py
Jika ingin menghentikan program, tekan tombol keyboard 'q'.