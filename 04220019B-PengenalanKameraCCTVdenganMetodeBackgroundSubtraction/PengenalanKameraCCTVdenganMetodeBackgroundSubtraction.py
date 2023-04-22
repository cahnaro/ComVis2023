import cv2

# Buat objek background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Buka file video
video = cv2.VideoCapture('Marine_Snow.avi')

# Stream video dari kamera
# video = cv2.VideoCapture(0) # 0 menandakan kamera pertama yang terdeteksi oleh sistem

while True:
    # Baca setiap frame video
    ret, frame = video.read()

    # Berhenti jika video sudah selesai
    if not ret:
        break

    # Lakukan background subtraction pada frame
    fg_mask = background_subtractor.apply(frame)

    # Tampilkan hasil background subtraction
    cv2.imshow('Foreground Mask', fg_mask)

    # Tunggu tombol keyboard ditekan
    if cv2.waitKey(1) == ord('q'):
        break

# Membersihkan objek
video.release()
cv2.destroyAllWindows()