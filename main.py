import cv2
import os
import numpy as np
import glob
from multiprocessing import Pool, cpu_count
import psutil
import time
from pedalboard.io import AudioFile
from pedalboard import *
import noisereduce as nr
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

"""
Credits: 
	1. https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
	2. http://richzhang.github.io/colorization/
	3. https://github.com/richzhang/colorization/

Downloaded the model files from: 
	1. colorization_deploy_v2.prototxt:    https://github.com/richzhang/colorization/tree/caffe/colorization/models
	2. pts_in_hull.npy:					   https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
	3. colorization_release_v2.caffemodel: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

"""

# Paths to load the model
PROTOTXT = r"Model/colorization_deploy_v2.prototxt"
POINTS = r"Model/pts_in_hull.npy"
MODEL = r"Model/colorization_release_v2.caffemodel"

def extract_audio(source_video, source_audio):
    # Load the video clip
    video_clip = VideoFileClip(source_video)

    # Extract the audio from the video clip
    audio_clip = video_clip.audio

    # Write the audio to a separate file
    audio_clip.write_audiofile(f"{source_audio}/Sample.mp3")

    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()

    print("Audio extraction successful!")

# Function to extract frames from a video until reaching the desired frame count
def extract_frames(video_file, output_images):
    cap = cv2.VideoCapture(video_file)
    # frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    if not os.path.exists(output_images):
        os.makedirs(output_images)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 1 == 0:
            output_file = f"{output_images}/frame_{frame_count}.jpg"
            cv2.imwrite(output_file, frame)
            print(f"Frame {frame_count} has been extracted and saved as {output_file}")
    
    cap.release()
    cv2.destroyAllWindows()

# Function to rename multiple files
def rename_frames(output_images):
    file_names = [int(filename[6:-4]) for filename in os.listdir(output_images)]
    max_num_digits = len(str(max(file_names)))
    one_digit_up_min = 10 ** max_num_digits

    for filename in os.listdir(output_images):
        new_file_name = int(filename[6:-4]) + one_digit_up_min
        my_source = os.path.join(output_images, filename)
        my_dest = os.path.join(output_images, f"frame_{new_file_name}.jpg")
        print(f"Renaming {my_source} to {my_dest}")
        os.rename(my_source, my_dest)

# Function to initialize and load the model
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

# Function to colorize a single frame
def colorize_frame(file_path, output_path):
    net = load_model()
    image = cv2.imread(file_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    output_file = os.path.join(output_path, os.path.basename(file_path))
    cv2.imwrite(output_file, colorized)
    print(f"Colorized frame saved as {output_file}")

# Function to control CPU usage
def limit_cpu_usage(max_cpu_usage):
    while psutil.cpu_percent(interval=1) > max_cpu_usage:
        time.sleep(0.1)

# Function to colorize frames in parallel
def color_frames(output_images, color_images):
    if not os.path.exists(color_images):
        os.makedirs(color_images)
    
    paths = glob.glob(os.path.join(output_images, "*.*"))
    
    with Pool(cpu_count()) as pool:
        pool.starmap(colorize_frame, [(path, color_images) for path in paths])
    
    '''
    with Pool(cpu_count()) as pool:
        for path in paths:
            pool.apply_async(colorize_frame, args=(path, color_images))
            limit_cpu_usage(95)  # Limit CPU usage to %
        pool.close()
        pool.join()
    '''
    
# Function to create a video from colorized frames
def join_color(color_images, color_video):
    video_filename = os.path.join(color_video, "SampleVideo.mp4")
    valid_images = sorted([img for img in os.listdir(color_images) if img.endswith((".jpg", ".jpeg", ".png"))])

    if not valid_images:
        print("No valid images found to create a video.")
        return

    first_image = cv2.imread(os.path.join(color_images, valid_images[0]))
    h, w, _ = first_image.shape
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_filename, codec, 25, (w, h))

    for img in valid_images:
        loaded_img = cv2.imread(os.path.join(color_images, img))
        vid_writer.write(loaded_img)
        print(f"Adding {img} to video.")

    vid_writer.release()
    print(f"Video saved as {video_filename}")

def enhance_audio(source_audio):
    sr=44100
    with AudioFile(f"{source_audio}/Sample.mp3").resampled_to(sr) as f:
        audio = f.read(f.frames)

    reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)

    board = Pedalboard([
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
        Compressor(threshold_db=-16, ratio=2.5),
        LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
        Gain(gain_db=10)
    ])

    effected = board(reduced_noise, sr)

    with AudioFile(f"{source_audio}/SampleAudio.mp3", 'w', sr, effected.shape[0]) as f:
        f.write(effected)

    print("Audio enhanced")

def output_video(color_video, source_audio):
    # Open the video and audio
    video_clip = VideoFileClip(f"{color_video}/SampleVideo.mp4")
    audio_clip = AudioFileClip(f"{source_audio}/SampleAudio.mp3")

    # Concatenate the video clip with the audio clip
    final_clip = video_clip.set_audio(audio_clip)

    # Export the final video with audio
    final_clip.write_videofile(f"{color_video}/OutputVideo.mp4")

    print("Task Complete")

if __name__ == "__main__":
    source_video = r"SourceVideo/SampleVideo.mp4"
    source_audio = r"SourceAudio/"
    output_images = r"OutputImages/"
    color_images = r"ColorImages/"
    color_video = r"OutputVideo/"

    extract_audio(source_video, source_audio)
    extract_frames(source_video, output_images)
    rename_frames(output_images)
    color_frames(output_images, color_images)
    join_color(color_images, output_video)
    enhance_audio(source_audio)
    output_video(color_video, source_audio)