import h5py
from extract_SHREC_placeCNN_features import *
import numpy as np
import configparser
from tqdm import tqdm


config = configparser.ConfigParser()
config.read('config.ini')
processed_data_path = config['PATH']['PROCESSED_DATA_DIR']
place_feat_path = osp.join(processed_data_path, 'Raw')

classes, images_in_class = parse_training_metadata()
X = []
y = []
# Load raw feature
print("Combining features into a feature matrix...")
for cls_index, cls in tqdm(enumerate(classes)):
    cls_place_feat_path = osp.join(place_feat_path, cls)
    for file in os.listdir(cls_place_feat_path):
        raw_place_feat_path = osp.join(cls_place_feat_path, file)
        place_feat = np.load(raw_place_feat_path)
        X.append(place_feat)
        y.append(cls_index)
X = np.vstack(np.array(X))
y = np.array(y)

num_training_items, feat_size = X.shape

file_name = osp.join(processed_data_path, 'train_data.h5')
with h5py.File(file_name, 'w') as out:
    out.create_dataset("X_train", data=X)
    out.create_dataset("y_train", data=y)