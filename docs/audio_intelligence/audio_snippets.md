Audio Intelligence Snippets
========================

Here are some code snippets that you might find useful when playing around with audio files 😉

## Load WAV file and check file stats

- To check some basic stats of a wav file, we can use `wave` package
  
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
    # read the data 
    data = wav_file.readframes(n_frames)
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

- We can also `scipy` package for wav file loading and printing stats

``` python linenums="1"
# import
from scipy.io import wavfile

# let's define a function to print the stats
def print_wav_stats(sample_rate, data):
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Data type: {data.dtype}")
    print(f"Duration: {data.shape[0] / sample_rate} seconds")
    print(f"Number of samples: {data.shape[0]}")
    print(f"Value range: {data.min()} to {data.max()}")
    print(f"Channels: {data.shape}")

# Load the wav file
file_path = f'{dataset_folder}/call_1.wav'
sample_rate, data = wavfile.read(file_path)
# print stats, example below
print_wav_stats(sample_rate, data) 
# Sample rate: 48000 Hz
# Data type: int16
# Duration: 1.18 seconds
# Number of samples: 56640
# Value range: -1787 to 1835
# Channels: (56640, 2)
``` 

!!! Note
    `scipy` returns data in an array with shape `(n_samples, n_channels)`. Whereas `wave` package returns data in bytes format.


## Get WAV file duration in seconds

- You can check the above section or below is a readymade function.

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

- Let's first do this with `scipy` package

``` python linenums="1"
# import
from scipy.io import wavfile

def stereo_to_mono(file_path):
    # Load the stereo wave file
    sample_rate, data = wavfile.read(file_path)
    # Check if it's already mono
    if data.shape[1] != 2:
        return "The file is not a stereo file."
    # Convert to mono by taking the mean of the two channels
    mono_data = data.mean(axis=1) # <--- THIS IS THE ONLY IMPORTANT LINE
    # Set the file path for output
    mono_file_path = file_path.replace(".wav", "_mono.wav")
    # Save the mono file
    wavfile.write(mono_file_path, sample_rate, mono_data.astype(data.dtype))
    return f"Stereo file converted to mono: {mono_file_path}"
```

- Same can be done with `wave` and `numpy` packages

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

## Downsample WAV file

- Let's first do this with `scipy` package

``` python linenums="1"
# import
from scipy.io import wavfile
import scipy.signal as sps

# Define a function to downsample a wave file
def downsample_wav(file_path, target_sample_rate=16000):
    """Downsample a wave file to a target sample rate.

    Args:
        file_path (str): The path to the wave file.
        target_sample_rate (int): The target sample rate to downsample to.
    
    Returns:
        str: A message indicating the success of the operation.
    """
    # Load the wave file
    sample_rate, data = wavfile.read(file_path)
    # Check if the target sample rate is the same as the original
    if sample_rate == target_sample_rate:
        return "The file is already at the target sample rate."
    # Calculate the new number of samples
    new_num_samples = int(data.shape[0] * target_sample_rate / sample_rate)
    # Resample the data
    new_data = sps.resample(data, number_of_samples)
    # Set the file path for output
    downsampled_file_path = file_path.replace(".wav", f"_downsampled_{target_sample_rate}.wav")
    # Save the downsampled file
    wavfile.write(downsampled_file_path, target_sample_rate, new_data.astype(data.dtype))
    # return a message indicating the success of the operation
    return f"File downsampled to {target_sample_rate} Hz: {downsampled_file_path}"
```

!!! Warning
    Before saving `.wav` file using `scipy`, make sure the dtype is `int16`. 