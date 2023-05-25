1 Basa Citra
2 Konversi ruang warna RGB menjadi HSV
3 Thresholding pada masing-masing kanal H, S, dan V dengan nilai threshold 0.5		
4	Operasi X-OR antara background dengan frame	
5	Konversi citra menjadi grayscale	
6	Konversi citra menjadi biner	
7	Filtering citra menggunakan filter median 5 x 5
8 Menghilangkan objek yang memiliki luasan di bawah 1000	
9	Membuat masking berdasarkan hasil deteksi foreground	
10 Menampilkan hasil deteksi foreground pada frame	
11 Menampilkan cropping hasil deteksi foreground
