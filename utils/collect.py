##################################################################
# Audio collection for wav data
# Select random class for user to say to diversify examples
##################################################################

import sounddevice as sd
import os
import queue
import sys
import soundfile as sf
from playsound import playsound
import random
import time

classes = os.listdir('data')
_class = random.choice(classes)
base_file_path = os.path.join('data', _class)
num_examples = len(os.listdir(base_file_path))
output_path = os.path.join(base_file_path, f"{num_examples}.wav")

print(f"Class to say: {_class}")
time.sleep(1)

q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


try:
    # Make sure the file is opened before recording anything:
    with sf.SoundFile(output_path, mode='x', samplerate=16000,
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
    playsound(output_path)

except Exception as e:
  print(e)
  pass
