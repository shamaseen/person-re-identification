# person-re-identification
person re-identification


https://github.com/shamaseen/person-re-identification/assets/94133296/0c53d496-26da-4a5a-b3be-3f0929ab8a1f

# # Introduction
This project aims to track people in the same video and re-identify them.


The framework used to accomplish this task relies on SMILEtrack to track and re-identify ID's of humans, respectively.
The tracking can be completed using YOLO Nas or YOlO Nas pose and ReID relies on SOLIDER-REID.
# # Notes
- I have implemented pose detection using YOLO-NAS to filter for individuals with near-complete body visibility. This is achieved by only accepting detections where at least 50% of the 17 keypoints have confidence scores exceeding 40%.
- in the SMILEtrack part I have disabled the inside Reid.

# # Installation
## #1 Anaconda
 - Download [Anaconda](https://www.anaconda.com/products/individual) if it is not installed on your machine

## #2 Clone the repository
```python
git clone https://github.com/shamaseen/person-re-identification.git
```
## #3 Create a project environment
```python
cd person-re-identification
conda create --name reid python=3.7.13
conda activate reid
```
## #4 Install dependencies

### 1-SOLIDER-REID
```python
pip install -r SOLIDER-REID/requirements.txt
```

```python
pip install -U openmim
mim install mmcv==2.1.0
mim install mmcv-full==1.7.0
```

### 2- SMILEtrack_Official
```python
pip install -r SMILEtrack_Official/requirements.txt
```

### 3-YOLO nas
```python
pip install -r requirements.txt
```
### 4- PyTorch 
- Install torch and torchvision based on the cuda version of your machine
```python
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

# # Download Pre-trained models.
- Download [SMILEtrack](https://github.com/WWangYuHsiang/SMILEtrack/tree/9bff163e757b2ab651f796bc1a096a9c87040818)
- Download [SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID/tree/8c08e1c3255e8e1e51e006bf189e52cc57b009ed)
- Download [YOLO-NAS](https://docs.deci.ai/super-gradients/latest/documentation/source/model_zoo.html#computer-vision-models-pretrained-checkpoints)
# # Run the code
##  Code Params
## ReID SOLIDER
- --soldier-weights: Path to the SOLIDER ReID model weights file.
- --soldier-config: Path to the SOLIDER ReID configuration file.
- --soldier-reid-thred: Threshold for ReID matching (higher values require stricter matches).
### Custom Parameters

- --frame-rate: Frame rate of the input video or camera.
- --frame-thred: Minimum number of frames an object must be tracked before being considered.
- --pose-conf: Confidence threshold for pose detection.
- --pose-point-conf: Confidence threshold for individual pose points.
- --frame-size: Tuple representing the frame size (width, height).
### Detection model Parameters
- --source: Source of the input (file/folder path or '0' for webcam).
- --model-name: Name of the object detection model to use.
- --model-weight: Path to the object detection model weights file.
- --conf-thres: Confidence threshold for object detection.
- --iou-thres: Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS).
- --device: CUDA device to use (e.g., '0', '0,1', or 'cpu').
- --classes-ids-fillter: Filter to include only specified object classes.
- --name: Name to use for saving results.
### Tracking Parameters
- --track_high_thresh: Confidence threshold for maintaining existing tracks.
--track_low_thresh: Lowest confidence for considering detections for tracking.
- --new_track_thresh: Threshold for creating new tracks.
- --track_buffer: Number of frames to keep lost tracks before removing them.
- --match_thresh: Matching threshold for associating detections with tracks.
- --aspect_ratio_thresh: Threshold for filtering out bounding boxes with extreme aspect ratios.
- --min-box-area: Threshold for filtering out very small bounding boxes.
- --fuse-score: Whether to fuse detection scores and IoU for track association.
you can run the code using the command
### ReID Parameters for SMILEtrack (disabled in the project)

- --with-reid: Whether to enable the ReID module.
- --fast-reid-config: Path to the FastReID configuration file.
- --fast-reid-weights: Path to the FastReID model weights file.
- --proximity_thresh: Threshold for rejecting ReID matches with low bounding box overlap.
- --appearance_thresh: Threshold for rejecting ReID matches with low appearance similarity.
## Run the code
```python
python3 main.py --soldier-weights PATH/MODEL.PTH --soldier-config CONFIG_PATH/FILE.CFG --model-name DETECTION_MODEL_NAME --model-weight MODEL_WEIGHT
--source VIDEO_PATH
```
example
```python
python3 main.py --soldier-weights ./SOLIDER-REID/swin_base_market.pth --soldier-config './SOLIDER-REID/configs/market/swin_base.yml' --model-name 'yolo_nas_pose_l' --model-weight 'coco_pose'
--source './Test_video/campus4-c0.mp4'
```

