# PlacesCNN Transfer Learning

The process includes 2 main steps:
1. Extract raw feature of placesCNN after processing through **avg_pooling** layer (vector dimension = 512).
2. Train a feedforward neural network to classify new classes based on pre-trained features.

**Requirements:**
- Python >= 3.7
- Torch & TorchVision >= 1.6.0
## 1. Installation
```
git clone https://github.com/nvtu/PlacesCNNTransferLearning
cd PlacesCNNTransferLearning
git clone https://github.com/nvtu/places365
pip3 install -r requirements.txt
```
## 2. Download SHREC'18 2D Scene dataset
```
wget http://orca.st.usm.edu/~bli/SceneIBR2018/SceneIBR2018_Dataset.zip
```
Unzip all file in the zip file.
## 3. Running Instruction
1. Configure data and output folder path in **config.ini** file, for instance:
- DATA_DIR = /mnt/DATA/nvtu/SceneIBR2018_Dataset
- PROCESSED_DATA_DIR = /home/nvtu/PlacesCNNTraining/PROCESSED_DATA
- MODEL_DIR = /home/nvtu/PlacesCNNTraining/MODEL
2. Extract placesCNN features:
```
python3 extract_SHREC_placeCNN_features.py
```
3. Prepare training data and create h5py file for fast data processing
```
python3 prepare_placefeat_data.py
```
4. Start training
```
python3 train.py
```
--> The best model is saved in MODEL_DIR

## STANDALONE PLACESCNN RUNNING
Please look at the file places365/run_placesCNN_unified.py in \_\_main\_\_ part to see example of standalone placesCNN running