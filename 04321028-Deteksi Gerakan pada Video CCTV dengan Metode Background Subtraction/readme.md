Background subtraction (BS) adalah teknik yang umum dan banyak digunakan untuk menghasilkan topeng latar depan (yaitu, gambar biner yang berisi piksel milik objek bergerak di tempat kejadian) dengan menggunakan kamera statis.

Seperti namanya, BS menghitung topeng latar depan melakukan pengurangan antara bingkai saat ini dan model latar belakang, yang berisi bagian statis dari pemandangan, atau dengan kata lain bingkai pemandangan tanpa objek latar depan.

1. Absolute Background Subtraction (ABS)
Dalam metode Absolute Background Subtraction (ABS), bingkai statis dari pemandangan tanpa objek latar depan dibandingkan dengan bingkai yang masuk. Perbedaan absolut antara bingkai memisahkan latar depan dari latar belakang.

2. MOG2 (Mixture of Gaussian)
Dalam metode ini, campuran distribusi k Gaussians memodelkan setiap piksel latar belakang, dengan nilai k dalam 3 dan 5. Diasumsikan bahwa distribusi yang berbeda mewakili setiap warna latar belakang dan latar depan yang berbeda. Bobot masing-masing distribusi yang digunakan pada model sebanding dengan jumlah waktu setiap warna tetap berada pada piksel tersebut. Oleh karena itu, ketika bobot distribusi piksel rendah, piksel tersebut diklasifikasikan sebagai latar depan.

3. KNN (K- Nearest Neighbors)
Metode K Nearest Neighbor (KNN) menghitung jarak Euclidean dari setiap segmen dalam gambar segmentasi ke setiap wilayah pelatihan yang Anda tentukan. Jarak diukur dalam ruang n-dimensi, di mana n adalah jumlah atribut untuk wilayah pelatihan tersebut.

Dalam persamaan rekursif KNN digunakan untuk terus memperbarui parameter model campuran Gaussian dan secara bersamaan memilih jumlah komponen yang sesuai untuk masing-masing
piksel.
