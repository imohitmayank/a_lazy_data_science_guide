Audio Intelligence Snippets
========================

Here are some code snippets that you might find useful when playing around with audio files :wink:

## Check WAV file's stats

- To check some basic stats of a wav file, we can use the following code, 
  
``` python linenums="1"
# import
import wave

# Open the wave file and extract some properties
file_path = f'{dataset_folder}/call_1.wav'
with wave.open(file_path, 'rb') as wav_file:
    n_channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    n_frames = wav_file.getnframes()
    comp_type = wav_file.getcomptype()
    comp_name = wav_file.getcompname()
# structure the required stats
wav_file_stats = {
    "Number of Channels": n_channels,
    "Sample Width": sample_width,
    "Frame Rate": frame_rate,
    "Number of Frames": n_frames,
    "Compression Type": comp_type,
    "Compression Name": comp_name
}
# print
print(wav_file_stats)
# Example output: 
# {'Number of Channels': 1,
#  'Sample Width': 2,
#  'Frame Rate': 22050,
#  'Number of Frames': 3821760,
#  'Compression Type': 'NONE',
#  'Compression Name': 'not compressed'}
```
## Get WAV file duration in seconds

``` python linenums="1"
# import 
import wave

def print_wav_duration(file_path):
    # Open the wave file
    with wave.open(file_path, 'rb') as wav_file:
        # Extract the frame rate and number of frames
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        # Calculate duration
        duration = n_frames / float(frame_rate)
        print(f"The duration of the file is: {duration} seconds.")

# Example usage with a placeholder file path
# You would replace 'path/to/file.wav' with the actual file path of your .wav file
print_wav_duration('path/to/file.wav')
```

## Convert Dual Channel WAV file to Mono

``` python linenums="1"
import wave
import numpy as np

def stereo_to_mono(file_path):
    # Open the stereo wave file
    with wave.open(file_path, 'rb') as stereo_wav:
        # Check if it's already mono
        if stereo_wav.getnchannels() != 2:
            return "The file is not a stereo file."
        
        # Read the stereo wave file data
        frames = stereo_wav.readframes(stereo_wav.getnframes())
        # Convert frames to numpy array
        frames = np.frombuffer(frames, dtype=np.int16)
        # Reshape the data to 2 columns for stereo
        frames = np.reshape(frames, (-1, 2))
        # Take the mean of the two channels to convert to mono
        mono_frames = frames.mean(axis=1, dtype=np.int16)
        
        # Get stereo file params to use for mono file
        params = stereo_wav.getparams()
        num_frames = len(mono_frames)

    # Set the file path for output
    mono_file_path = file_path.replace(".wav", "_mono.wav")

    # Create a new wave file for mono
    with wave.open(mono_file_path, 'wb') as mono_wav:
        # Set parameters for mono (nchannels=1)
        mono_wav.setparams((1, params.sampwidth, params.framerate, num_frames, params.comptype, params.compname))
        # Write frames for mono
        mono_wav.writeframes(mono_frames.tobytes())

    return f"Stereo file converted to mono: {mono_file_path}"

# Replace with an actual file path to a stereo wav file
# e.g., stereo_to_mono("path/to/stereo_file.wav")
stereo_to_mono(f'{dataset_folder}/call_5.wav')
```