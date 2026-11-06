import os
import subprocess
import cv2
import numpy as np
from scipy.io import wavfile

def extract_temp_wav(video_path: str, wav_path: str = "temp/temp_audio.wav") -> str:
    """Uses FFmpeg to extract a lightweight 16kHz mono WAV file for fast math."""
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return wav_path

def calculate_audio_rms(video_path: str) -> np.ndarray:
    """Returns a 1D NumPy array of audio volume energy for each second."""
    wav_path = extract_temp_wav(video_path)
    sample_rate, data = wavfile.read(wav_path)
    data = data.astype(np.float32)
    
    total_seconds = len(data) // sample_rate
    rms_per_second = []
    
    for sec in range(total_seconds):
        chunk = data[sec * sample_rate : (sec + 1) * sample_rate]
        rms = np.sqrt(np.mean(chunk ** 2))
        rms_per_second.append(rms)
        
    return np.array(rms_per_second, dtype=np.float32)

def calculate_motion_per_second(video_path: str, sample_fps: int = 5) -> np.ndarray:
    """Scans video at 160x90 grayscale to measure second-by-second pixel velocity."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(native_fps / sample_fps)))
    
    motion_per_second = []
    current_sec_diffs = []
    prev_gray = None
    frame_idx = 0
    
    while True:
        # grab() is super fast because it advances the frame without decoding pixels
        ret = cap.grab()
        if not ret:
            break
            
        # Only decode the pixels every Nth frame (e.g., 5 times per second)
        if frame_idx % frame_step == 0:
            _, frame = cap.retrieve()
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Measure absolute pixel change between this frame and the last
                diff = cv2.absdiff(gray, prev_gray)
                current_sec_diffs.append(float(np.mean(diff)))
            prev_gray = gray
            
        frame_idx += 1
        
        # Every time we pass 1 full second of native frames, store the average motion
        if frame_idx % int(round(native_fps)) == 0:
            avg_motion = np.mean(current_sec_diffs) if current_sec_diffs else 0.0
            motion_per_second.append(avg_motion)
            current_sec_diffs = []
            
    cap.release()
    return np.array(motion_per_second, dtype=np.float32)

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Scales any array of numbers to sit cleanly between 0.0 and 1.0."""
    if len(arr) == 0 or np.max(arr) == np.min(arr):
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def find_action_spikes(
    video_path: str,
    top_n: int = 5,
    clip_duration: int = 20,
    audio_weight: float = 0.6,
    motion_weight: float = 0.4
) -> list:
    """Combines normalized Audio RMS + Visual Motion to find the best action clips."""
    print("1/2 Scanning audio energy...")
    audio_scores = calculate_audio_rms(video_path)
    
    print("2/2 Scanning visual pixel velocity...")
    motion_scores = calculate_motion_per_second(video_path)
    
    # Match lengths in case audio/video streams differ by 1 second
    total_seconds = min(len(audio_scores), len(motion_scores))
    audio_norm = normalize_array(audio_scores[:total_seconds])
    motion_norm = normalize_array(motion_scores[:total_seconds])
    
    # Hybrid Score Formula: (0.6 * Audio) + (0.4 * Motion)
    combined_scores = (audio_weight * audio_norm) + (motion_weight * motion_norm)
    
    spikes = []
    scores_copy = combined_scores.copy()
    
    for _ in range(top_n):
        peak_sec = int(np.argmax(scores_copy))
        peak_score = float(scores_copy[peak_sec])
        
        if peak_score <= 0:
            break
            
        start_sec = max(0, peak_sec - 5)
        end_sec = min(total_seconds, start_sec + clip_duration)
        
        spikes.append({
            "peak_second": peak_sec,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_time": f"{start_sec // 60:02d}:{start_sec % 60:02d}",
            "end_time": f"{end_sec // 60:02d}:{end_sec % 60:02d}",
            "hybrid_score": round(peak_score, 3),
            "audio_score": round(float(audio_norm[peak_sec]), 2),
            "motion_score": round(float(motion_norm[peak_sec]), 2)
        })
        
        # Zero out surrounding 30 seconds (-15s to +15s) to avoid duplicate overlaps
        suppress_start = max(0, peak_sec - 15)
        suppress_end = min(total_seconds, peak_sec + 15)
        scores_copy[suppress_start:suppress_end] = -1.0
        
    return spikes

# --- TEST BLOCK ---
if __name__ == "__main__":
    test_video = "test.mp4"
    
    if os.path.exists(test_video):
        results = find_action_spikes(test_video)
        print("\n--- TOP HYBRID ACTION CLIPS ---")
        for i, clip in enumerate(results, 1):
            print(
                f"Clip #{i}: {clip['start_time']} - {clip['end_time']} | "
                f"Peak: {clip['peak_second']}s | "
                f"Score: {clip['hybrid_score']} (Audio: {clip['audio_score']}, Motion: {clip['motion_score']})"
            )