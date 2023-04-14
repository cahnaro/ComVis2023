import moviepy.editor as mp
import speech_recognition as sr

# Inisialisasi recognizer
r = sr.Recognizer()

# Mengambil audio dari video
video_file = "nama_file_video.mp4"
audio_file = "nama_file_audio.wav"
clip = mp.VideoFileClip(video_file)
clip.audio.write_audiofile(audio_file)

# Membuka file audio
with sr.AudioFile(audio_file) as source:
    audio = r.record(source)

# Menggunakan Google Speech Recognition API untuk melakukan transkripsi
try:
    text = r.recognize_google(audio)
    print("Transkripsi: " + text)
except sr.UnknownValueError:
    print("Google Speech Recognition tidak dapat memahami audio")
except sr.RequestError as e:
    print("Permintaan ke Google Speech Recognition gagal; {0}".format(e))
