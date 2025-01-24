import threading
import tkinter as tk
from PIL import Image, ImageTk
from queue import Queue
import sounddevice as sd
import numpy as np
import torch
import webrtcvad
import librosa
import os
import wave
import time
from model.model import CNN
import sys
from tkinter import ttk

# Model setup
classes = sorted(os.listdir('data'))
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("/home/linden/Desktop/projects/cr_prediction/src/model/cnn_model.pt", map_location=device))
model.eval()
prediction_threshold = 0.7

# Audio and VAD setup
FRAME_DURATION_MS = 20
SAMPLE_RATE = 16000
BLOCK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
vad = webrtcvad.Vad(3)
audio_q = Queue()
ui_q = Queue()

# Clash royale logic
cards = []
images = {}
elixir_count = 5
cls_to_elixir = {
    'archers': 3,
    'arrows': 3,
    'fireball': 4,
    'giant': 5,
    'knight': 3,
    'mini_pekka': 4,
    'minions': 3,
    'musketeer': 4,
}

# Load each image and cache for efficient image swapping
for cls in classes + ['unknown']:
    if cls != '_silence':
        img = Image.open(f"images/{cls}.webp")
        images[cls] = img

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.tobytes())

def save_segment_to_wav(pcm_bytes, sample_rate, out_path):
    with wave.open(out_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

# Fetch opponent's hand and display in UI
def update_card_images():
    if len(cards) > 4:
        new_deck = cards[:-4]
        while len(new_deck) < 4:
            new_deck.append('unknown')
        card1.image = ImageTk.PhotoImage(images[new_deck[0]])
        card1.config(image=card1.image)

        card2.image = ImageTk.PhotoImage(images[new_deck[1]])
        card2.config(image=card2.image)

        card3.image = ImageTk.PhotoImage(images[new_deck[2]])
        card3.config(image=card3.image)

        card4.image = ImageTk.PhotoImage(images[new_deck[3]])
        card4.config(image=card4.image)

# Classify audio segment
def classify_segment(segment_bytes):
    global cards, elixir_count
    audio_int16 = np.frombuffer(segment_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    n_mfcc = 12
    mfcc = librosa.feature.mfcc(y=audio_float32, sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=1024)
    mfcc = mfcc.T

    # If audio is too short for CNN
    if mfcc.shape[0] < 4 or mfcc.shape[1] < 4:
        return

    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(mfcc_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    predicted_label = classes[pred_idx]
    if confidence > prediction_threshold:
        ui_q.put(f"{time.strftime('%X')} - {predicted_label} ({confidence:.2f})")
        # Update card position in queue
        if predicted_label in cards:
            cards.remove(predicted_label)
        cards.append(predicted_label)
        elixir_count = max(0, elixir_count - cls_to_elixir[predicted_label])
        update_card_images()

def audio_processing_loop():
    speech_buffer = bytearray()
    in_speech = False

    while True:
        block = audio_q.get()
        is_speech = vad.is_speech(block, SAMPLE_RATE)
        if is_speech and not in_speech:
            in_speech = True
            speech_buffer.clear()
            speech_buffer.extend(block)
        elif is_speech and in_speech:
            speech_buffer.extend(block)
        elif not is_speech and in_speech:
            in_speech = False
            final_segment = bytes(speech_buffer)
            speech_buffer.clear()
            if len(final_segment) > 640:
                classify_segment(final_segment)

def start_audio_stream():
    global running
    running = True
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=BLOCK_SIZE, callback=audio_callback):
        audio_processing_loop()

# Poll ui_q continuously to show latest prediction made
def update_prediction_text():
    while not ui_q.empty():
        result = ui_q.get()
        result_label.config(text=result)
    root.after(100, update_prediction_text)

def stop_audio_stream():
    global running
    running = False
    print("Audio stream stopped.")

def quit_program():
    stop_audio_stream()  # Stop the audio stream
    root.quit()  # Stop the Tkinter main loop
    root.destroy()  # Destroy the Tkinter window
    print("Program shut down successfully.")


def update_elixir_count():
    global elixir_count
    if elixir_count < 10:
        elixir_count += 1
        elixir_count_label.config(text=str(elixir_count))
        progress_bar["value"] = elixir_count
    root.after(2800, update_elixir_count)

# Tkinter UI
root = tk.Tk()
root.title("Clash Royale Prediction")
root.geometry("1200x600")

result_label = tk.Label(root, text="Waiting for input...", font=("Arial", 16))
result_label.grid(row=0, column=0, columnspan=4, pady=20)

# Ensure the images are used in the UI to prevent garbage collection
im1 = ImageTk.PhotoImage(images["unknown"])
card1 = tk.Label(root, image=im1)
card1.grid(row=2, column=0,pady=20)
card1.image = im1

im2 = ImageTk.PhotoImage(images["unknown"])
card2 = tk.Label(root, image=im2)
card2.grid(row=2, column=1,pady=20)
card2.image = im2

im3 = ImageTk.PhotoImage(images["unknown"])
card3 = tk.Label(root, image=im3)
card3.grid(row=2, column=2,pady=20)
card3.image = im3

im4 = ImageTk.PhotoImage(images["unknown"])
card4 = tk.Label(root, image=im4)
card4.grid(row=2, column=3,pady=20)
card4.image = im4
    
style = ttk.Style()
style.theme_use("default")  # Use the default theme for better customization
style.configure("Custom.Horizontal.TProgressbar",
                background="#AB1BA5",  # Color of the progress bar
                troughcolor="#655548",  # Color of the background
                thickness=20)

progress_bar = ttk.Progressbar(root, style="Custom.Horizontal.TProgressbar", orient="horizontal", length=1000, mode="determinate")
progress_bar["maximum"] = 10
progress_bar["value"] = 5
progress_bar.grid(row=3, column=0, columnspan=4, pady=10)


elixir_count_label = tk.Label(root, text=str(elixir_count))
elixir_count_label.grid(row=3, column=4, pady=10)

start_button = tk.Button(root, text="Start Audio", font=("Arial", 14), command=lambda: threading.Thread(target=start_audio_stream, daemon=True).start())
start_button.grid(row=4, column=0, pady=20)

quit_button = tk.Button(root, text="Quit", font=("Arial", 14), command=quit_program)
quit_button.grid(row=4, column=1, pady=10)


if __name__ == "__main__":
    # Start the UI loop
    root.after(100, update_prediction_text)
    root.after(2800, update_elixir_count)  # Elixir updates by 1 every 2.8 seconds
    root.mainloop()