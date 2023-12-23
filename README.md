# person-re-identification
person re-identification


# # Introduction
This project aims to track people in the same video and re-identify them.


The framework used to accomplish this task relies on SMILEtrack to track and re-identify ID's of humans, respectively.
The tracking can be completed using YOLO Nas or YOlO Nas pose and ReID relies on SOLIDER-REID.

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
conda create --name reid python=3.7
conda activate reid
```
## #4 Install dependencies

### 1- SMILEtrack_Official
```python
pip install -r SMILEtrack_Official/requirements.txt
```
### 2- PyTorch
- Install torch and torchvision based on the cuda version of your machine
```python
conda install pytorch torchvision cudatoolkit -c pytorch
```
### 3-YOLO nas
```python
pip install -r requirements.txt
```