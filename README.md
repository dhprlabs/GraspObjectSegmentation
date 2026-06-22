# Grasp Object Segmentation

This repo contains a `inference.py` file which use the intelrealsense's pyrealsense2 lib along with pytorch, ultralytics, opencv in order
to output the inference of a trained `yolov8n-seg` model can be directly deployed on any edge devices.

## How to run

```bash
python3 -m venv /home/$USER/_anyname_   # 1. make virtual env

source /home/$USER/_anyname_   # 2. source the virtual env

git clone https://github.com/dhprlabs/GraspObjectSegmentation.git   # 3. clone the repo

cd /path/to/repository   # 4. navigate to the repo

pip install -r requirements.txt   # 5. install required libraries

```

## After connecting the intel realsense d435i

```bash
python3 inference.py   # 6. run the inference.py file
```
