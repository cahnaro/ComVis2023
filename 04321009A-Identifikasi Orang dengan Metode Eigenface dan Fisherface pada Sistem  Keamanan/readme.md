Algoritma Eigenface
1. Hitung rata-rata mu_{i} dari setiap kolom dalam matriks, memberi kita nilai intensitas piksel rata-rata untuk setiap koordinat (x, y) dalam himpunan data gambar.
2. Kurangi mu_{i} dari setiap kolom c_{i} — ini disebut pemusatan rata-rata data dan merupakan langkah yang diperlukan saat melakukan PCA.
3. Sekarang matriks M kita telah berpusat pada rata-rata, hitung matriks kovarians.
4. Lakukan dekomposisi nilai eigen pada matriks kovarians untuk mendapatkan nilai eigen lambda_{i} dan vektor eigen mathbf{X_{i}}.
5. Urutkan mathbf{X_{i}} menurut |lambda_{i}|, terbesar hingga terkecil.
6. Ambil vektor eigen N teratas dengan besaran nilai eigen terbesar yang sesuai.
7. Ubah data input dengan memproyeksikan (yaitu, mengambil produk titik) ke ruang yang dibuat oleh vektor eigen N teratas - vektor eigen ini disebut eigenface kita.

Algoritma Fisherface
1. Biarkan X menjadi vektor acak dengan sampel yang diambil dari c.
2. Matriks sebar SB dan S_{W} dihitung.
3. Di mana μ adalah rata-rata total.
4. Dan μi adalah rata-rata kelas i∈{1,...,c}.
5. Algoritma klasik Fisher sekarang mencari proyeksi W, yang memaksimalkan kriteria pemisahan kelas.
6. Mengikuti solusi untuk masalah optimasi ini diberikan dengan memecahkan Masalah Eigenvalue Umum.
