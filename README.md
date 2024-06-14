# Video Colorization and Audio Enhancement Script

This Python script processes a video by extracting its audio, extracting and colorizing its frames, enhancing the audio, and then combining the processed video and audio into a final output.

![Screenshot (99)](https://github.com/Mamun1113/ComputerVision-ColorMovie/assets/66373332/75305d76-ab15-46b9-aa10-a25fc5608ff5)
![Screenshot (98)](https://github.com/Mamun1113/ComputerVision-ColorMovie/assets/66373332/3d2a5666-4272-4a6d-a7aa-557793434e46)
![Screenshot (100)](https://github.com/Mamun1113/ComputerVision-ColorMovie/assets/66373332/0259154e-f524-4721-8254-50329717c951)

## Demo: https://youtu.be/47sEN_ezAOY

## Features

1. **Audio Extraction**: Extracts audio from a video file and saves it separately.
2. **Frame Extraction**: Extracts frames from a video file.
3. **Frame Colorization**: Colorizes extracted frames using a deep learning model.
4. **Audio Enhancement**: Enhances the extracted audio using noise reduction and other audio effects.
5. **Video Creation**: Combines colorized frames into a video.
6. **Final Output**: Combines the processed video and enhanced audio into a final video file.

## Model Credits

- OpenCV Colorization Sample: [GitHub](https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py)
- Colorization Paper and Model: [Rich Zhang](http://richzhang.github.io/colorization/)
- Colorization GitHub Repository: [GitHub](https://github.com/richzhang/colorization/)

Model files downloaded from:
- `colorization_deploy_v2.prototxt`: [Link](https://github.com/richzhang/colorization/tree/caffe/colorization/models)
- `pts_in_hull.npy`: [Link](https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy)
- `colorization_release_v2.caffemodel`: [Dropbox](https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1)

## Requirements

- Python 3.x
- OpenCV
- Numpy
- MoviePy
- Pedalboard
- noisereduce
- psutil

## Installation

Install the required Python libraries using pip:

```sh
pip install opencv-python numpy moviepy pedalboard noisereduce psutil
```

## Usage
Place the source video in the SourceVideo directory.
Run the script:

```sh
python main.py
```

The final output video will be saved in the OutputVideo directory as OutputVideo.mp4.

## Folders/Files need to be created:
1. main.py (given)
2. Model
3. SourceVideo
4. SourceAudio
5. OutputImages
6. ColorImages
7. OutputVideo

![image](https://github.com/Mamun1113/ComputerVision-ColorMovie/assets/66373332/7f737886-0fe5-4136-8e48-4f5645bc2a3a)

## Notes
Ensure your model files (colorization_deploy_v2.prototxt, pts_in_hull.npy, colorization_release_v2.caffemodel) are placed in the Model directory.
Adjust the paths in the script as needed to match your directory structure.
