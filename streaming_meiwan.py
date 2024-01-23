import cv2

# Load pre-trained model untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # Gunakan 0 untuk kamera internal

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Ubah ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiscale(gray, scaleFactor=1.1, minNeighbors=5, minSize=30, 30))

    # Gambar kotak di sekitar wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y,), (x+w, y+h) (255, 0, 0), 2)

    # Tampilkan frame
    cv2.imshow('Face Detection', frame)

    # Jika tombol 's' ditekan, simpan gambar
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('captured_face.jpg', frame)
        print("Image saved!")

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()