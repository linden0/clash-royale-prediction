import pytest
import os
import numpy as np
from queue import Queue

def test_audio_callback():
    audio_q = Queue()
    test_data = (np.random.rand(32000) * 32767).astype(np.int16)  # Mock audio data

    def mock_callback(indata, frames, time_info, status):
        audio_q.put(indata.tobytes())

    # Call the mock callback
    mock_callback(test_data, len(test_data), None, None)

    # Check if data was added to the queue
    assert not audio_q.empty(), "Audio queue is empty after callback"
    assert len(audio_q.get()) > 0, "No data was added to the audio queue"


