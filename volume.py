import pyaudio
import numpy as np

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

def initialize():
    # Initialize PyAudio
    p = pyaudio.PyAudio()



# Function to change volume of each channel independently
def change_volume(data, left_volume, right_volume):
    # Convert byte data to numpy array
    audio_data = np.frombuffer(data, dtype=np.int16)
    
    # Reshape array to separate left and right channels
    audio_data = audio_data.reshape((-1, 2))
    
    # Apply volume changes
    audio_data[:, 0] = audio_data[:, 0] * left_volume
    audio_data[:, 1] = audio_data[:, 1] * right_volume
    
    # Ensure values are within int16 range
    audio_data = np.clip(audio_data, -32768, 32767)
    
    # Convert numpy array back to byte data
    return audio_data.astype(np.int16).tobytes()

# Callback function for audio stream
def callback(in_data, frame_count, time_info, status):
    left_volume = 0.5  # Example volume for left channel
    right_volume = 1.0  # Example volume for right channel
    out_data = change_volume(in_data, left_volume, right_volume)
    return (out_data, pyaudio.paContinue)

# Open input stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

# Start the stream
stream.start_stream()

print("Adjusting stereo volume per channel in real-time. Press Ctrl+C to stop.")

try:
    while stream.is_active():
        # You can update left_volume and right_volume here in real-time if needed
        pass
except KeyboardInterrupt:
    print("Stopping...")

# Stop and close the stream
stream.stop_stream()
stream.close()

# Terminate PyAudio
p.terminate()
