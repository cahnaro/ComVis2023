Tugas MK Computer Vision 2023

Cara Kerja Program :

kami mengimpor library MoviePy dan SpeechRecognition. Kami kemudian mengambil audio dari video menggunakan objek VideoFileClip dari MoviePy dan mengekstrak audio ke file WAV menggunakan metode write_audiofile. 
Setelah itu, kami membuka file audio menggunakan objek AudioFile dari SpeechRecognition dan merekam audio menggunakan metode record. 
Akhirnya, kami menggunakan metode recognize_google dari SpeechRecognition untuk melakukan transkripsi teks dari audio menggunakan Google Speech Recognition API.

Langkah-langkah Running Program :

Buka IDLE Python
File -> Open -> Cari lokasi file Pengenalan Suara.py
Pilih menu Run -> Run Module
