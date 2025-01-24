import sounddevice as sd
import wave
import numpy as np
import os
import queue
import sys
import soundfile as sf
from playsound import playsound
import argparse

def record_audio(download_path):
    # Ensure the provided path is valid
    if not os.path.isdir(download_path):
        print(f"The provided path '{download_path}' does not exist. Exiting...")
        return
    
    # Define the file path for recording
    file_path = os.path.join(download_path, 'recording.wav')
    
    # Remove the file if it already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        # Make sure the file is opened before recording anything
        with sf.SoundFile(file_path, mode='x', samplerate=16000,
                          channels=1) as file:
            with sd.InputStream(samplerate=16000, device=sd.default.device,
                                channels=1, callback=callback):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording finished, playing it back:')
        playsound(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Recorder Script")
    parser.add_argument("download_path", type=str, help="Path where the recording will be saved")
    args = parser.parse_args()
    
    record_audio(args.download_path)
