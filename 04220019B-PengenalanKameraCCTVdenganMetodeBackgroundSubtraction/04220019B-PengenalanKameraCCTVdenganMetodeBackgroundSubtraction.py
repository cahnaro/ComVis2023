import cv2

# Inisialisasi video capture
cap = cv2.VideoCapture(0)

# Inisialisasi metode Background Subtraction
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Baca frame dari video capture
    ret, frame = cap.read()
    
    # Terapkan metode Background Subtraction pada frame
    fgmask = fgbg.apply(frame)
    
    # Tampilkan hasil pengolahan pada frame
    cv2.imshow('frame', fgmask)
    
    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop video capture dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()