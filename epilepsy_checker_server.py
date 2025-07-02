# epilepsy_checker_server.py
# This server uses Flask, yt-dlp, and OpenCV to analyze YouTube videos.
# VERSION 2: Includes more nuanced frequency analysis based on clinical data.
#
# To run this server:
# 1. Install necessary libraries:
#    pip install Flask flask-cors yt-dlp opencv-python numpy
#
# 2. Run the script from your terminal:
#    python epilepsy_checker_server.py
#
# The server will start on http://127.0.0.1:5000

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import yt_dlp
import os
import logging
import traceback

# --- Setup ---
app = Flask(__name__)
CORS(app) # Allow requests from the frontend HTML file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Analysis Parameters ---
# These values are based on established guidelines for photosensitive epilepsy triggers.
# Increased frame rate for more accurate frequency detection.
FRAME_RATE = 30
LUMINANCE_CHANGE_THRESHOLD = 20
# Frequency (Hz) based limits.
MODERATE_RISK_HZ_LOWER_BOUND = 3
HIGH_RISK_HZ_LOWER_BOUND = 16
RED_FLASH_SENSITIVITY = 1.2

# --- Analysis Logic ---

def analyze_video_url(video_url):
    """
    Main function to download and analyze a video from a URL.
    This is the robust version that handles errors and cleans up files.
    """
    ydl_opts = {
        'format': 'best[ext=mp4][height<=480]/best[height<=480]',
        'outtmpl': '%(id)s.%(ext)s',
        'quiet': True,
        'noplaylist': True,
    }

    filepath = None
    analysis_results = {}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info(f"Fetching video info for: {video_url}")
            info = ydl.extract_info(video_url, download=False)

            if not isinstance(info, dict):
                raise TypeError(f"yt-dlp did not return valid video metadata. Analysis cannot continue.")

            logging.info(f"Starting download for video ID: {info.get('id')}")
            ydl.download([video_url])
            
            filepath = ydl.prepare_filename(info)
            logging.info(f"Download finished. Analyzing file: {filepath}")

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Video file was not found at expected path: {filepath}")

            analysis_results = analyze_video_file(filepath)
            
    except Exception as e:
        logging.error(f"Error during video download or analysis: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logging.info(f"Cleaned up video file: {filepath}")
            except OSError as e:
                logging.error(f"Error removing file {filepath}: {e}")

    return analysis_results


def analyze_video_file(filepath):
    """
    Analyzes a local video file for photosensitive triggers with nuanced frequency detection.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return {"error": "Could not open video file for analysis."}

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0:
        source_fps = 30
        logging.warning("Video has 0 FPS metadata, assuming 30 FPS.")

    # We now analyze at a higher FRAME_RATE defined globally
    frame_skip = int(source_fps / FRAME_RATE) if source_fps > FRAME_RATE else 1
    
    frame_count = 0
    last_luminance = -1
    frame_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        current_time = frame_count / source_fps
        small_frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        current_luminance = np.mean(gray_frame)
        
        is_flash = False
        if last_luminance != -1 and abs(current_luminance - last_luminance) > LUMINANCE_CHANGE_THRESHOLD:
            is_flash = True
        
        is_red_flash = False
        if is_flash:
            b, g, r = cv2.split(small_frame)
            mean_r = np.mean(r)
            if mean_r > 128 and mean_r > np.mean(g) * RED_FLASH_SENSITIVITY and mean_r > np.mean(b) * RED_FLASH_SENSITIVITY:
                is_red_flash = True

        frame_data.append({
            "time": current_time,
            "flash": is_flash,
            "red_flash": is_red_flash
        })
        last_luminance = current_luminance

    duration = frame_count / source_fps
    cap.release()
    
    # --- NEW: More sophisticated post-analysis processing ---
    timestamps = []
    for i, frame in enumerate(frame_data):
        one_second_ago = frame["time"] - 1
        frames_in_last_second = [f for f in frame_data if f["time"] >= one_second_ago and f["time"] <= frame["time"]]
        flash_count_in_last_second = sum(1 for f in frames_in_last_second if f["flash"])

        # Check for High-Risk Frequency (e.g., 16-30 Hz)
        if flash_count_in_last_second > HIGH_RISK_HZ_LOWER_BOUND:
            if not any(t['time'] > one_second_ago for t in timestamps if "Flash Rate" in t['type']):
                 timestamps.append({"time": frame["time"], "type": f"High-Risk Flash Rate ({HIGH_RISK_HZ_LOWER_BOUND+1}-30 Hz)"})
        
        # Check for Moderate-Risk Frequency (e.g., 3-15 Hz)
        elif flash_count_in_last_second > MODERATE_RISK_HZ_LOWER_BOUND:
            if not any(t['time'] > one_second_ago for t in timestamps if "Flash Rate" in t['type']):
                 timestamps.append({"time": frame["time"], "type": f"Moderate Flash Rate ({MODERATE_RISK_HZ_LOWER_BOUND+1}-{HIGH_RISK_HZ_LOWER_BOUND} Hz)"})

        # Check for Red Flashes (as before)
        if frame["red_flash"]:
             if not any(abs(t['time'] - frame['time']) < 1 for t in timestamps if t['type'] == 'Red Flash Sequence'):
                timestamps.append({"time": frame["time"], "type": "Red Flash Sequence"})

    return {
        "timestamps": sorted(timestamps, key=lambda x: x['time']),
        "duration": duration
    }


# --- API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided."}), 400

    url = data['url']
    
    results = analyze_video_url(url)
    if "error" in results:
        return jsonify(results), 400

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
