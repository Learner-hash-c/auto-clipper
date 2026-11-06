import os
import subprocess
import numpy as np
from scipy.io import wavfile

def extract_temp_wav(video_path: str, wav_path: str = "temp/temp_audio.wav") -> str:
    """Uses FFmpeg to extract a lightweight 16kHz mono WAV file for fast math."""
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    
    command = [
        "ffmpeg", "-y",               # Overwrite temp file if it exists
        "-i", video_path,             # Input video path
        "-vn",                        # Skip video decoding (makes it 20x faster)
        "-ac", "1",                   # 1 audio channel (Mono)
        "-ar", "16000",               # 16,000 Hz sample rate (lightweight in RAM)
        wav_path
    ]
    # Run FFmpeg silently in the background
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return wav_path

def find_audio_spikes(video_path: str, top_n: int = 5, clip_duration: int = 20) -> list:
    """Calculates second-by-second volume energy and returns the top action windows."""
    wav_path = extract_temp_wav(video_path)
    
    # 1. Load audio: sample_rate is 16000, data is a giant NumPy array of sound waves
    sample_rate, data = wavfile.read(wav_path)
    
    # Convert to float32 so squaring numbers doesn't overflow memory
    data = data.astype(np.float32)
    
    # 2. Slice the giant array into 1-second buckets and calculate RMS (volume energy)
    total_seconds = len(data) // sample_rate
    rms_per_second = []
    
    for sec in range(total_seconds):
        chunk = data[sec * sample_rate : (sec + 1) * sample_rate]
        # Root Mean Square formula: square root of the average of squared wave values
        rms = np.sqrt(np.mean(chunk ** 2))
        rms_per_second.append(rms)
        
    # 3. Pick the top_n loudest peaks without overlapping clips
    spikes = []
    rms_copy = np.array(rms_per_second)
    
    for _ in range(top_n):
        peak_sec = int(np.argmax(rms_copy))  # Find the index (second) with highest volume
        peak_score = float(rms_copy[peak_sec])
        
        if peak_score == 0:
            break
            
        # Center the 20s clip around the peak (e.g., 5s before impact, 15s after)
        start_sec = max(0, peak_sec - 5)
        end_sec = min(total_seconds, start_sec + clip_duration)
        
        spikes.append({
            "peak_second": peak_sec,
            "start_time": f"{start_sec // 60:02d}:{start_sec % 60:02d}",
            "end_time": f"{end_sec // 60:02d}:{end_sec % 60:02d}",
            "energy_score": round(peak_score, 2)
        })
        
        # Zero out the surrounding 20 seconds so the next loop finds a DIFFERENT fight moment
        suppress_start = max(0, peak_sec - 15)
        suppress_end = min(total_seconds, peak_sec + 15)
        rms_copy[suppress_start:suppress_end] = 0
        
    return spikes

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Replace "test.mp4" with the path to any video on your PC
    test_video = "test.mp4" 
    
    if os.path.exists(test_video):
        print(f"Scanning audio energy for: {test_video}...")
        results = find_audio_spikes(test_video)
        for i, clip in enumerate(results, 1):
            print(f"Clip #{i}: {clip['start_time']} to {clip['end_time']} (Peak at {clip['peak_second']}s | Score: {clip['energy_score']})")
    else:
        print(f"Place a video named '{test_video}' in your auto-clipper folder to test!")