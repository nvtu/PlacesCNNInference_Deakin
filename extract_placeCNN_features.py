import os
import os.path as osp
from multiprocessing import Pool
import sys
current_dir = osp.dirname(osp.abspath(__file__))
if not current_dir in sys.path:
    sys.path.append(current_dir)
from typing import List, Tuple
from functools import partial
from places365.run_placesCNN_unified import ExtractPlaceCNNFeatureParams, extract_placeCNN_feature, load_model
from tqdm import tqdm
import configparser


def create_folder(folder_path):
    if not osp.exists(folder_path):
        os.makedirs(folder_path)

# Initialize some path paramenters
config = configparser.ConfigParser()
config.read('config.ini')
current_dir = os.getcwd()
data_dir = config['PATH']['DATA_DIR']

# Processed data folder path
processed_data_path = config['PATH']['PROCESSED_DATA_DIR']
create_folder(processed_data_path)

# Attribute feature and prediction folder
attribute_folder_path = osp.join(processed_data_path, 'Attributes')
attribute_feat_path = osp.join(attribute_folder_path, 'feat')
attribute_pred_path = osp.join(attribute_folder_path, 'pred')
create_folder(attribute_feat_path)
create_folder(attribute_pred_path)

# Category feature and prediction folder
category_folder_path = osp.join(processed_data_path, 'Categories')
category_feat_path = osp.join(category_folder_path, 'feat')
category_pred_path = osp.join(category_folder_path, 'pred')
create_folder(category_feat_path)
create_folder(category_pred_path)

# Raw feature folder
raw_feat_path = osp.join(processed_data_path, 'Raw')
create_folder(raw_feat_path)

# CAMs folder
CAMs_folder_path = osp.join(processed_data_path, 'CAMs')
create_folder(CAMs_folder_path)


def create_output_folder():
    user_ids = os.listdir(data_dir)
    for user_id in user_ids:
        user_folder_path = osp.join(data_dir, user_id)
        for date in os.listdir(user_folder_path):
            create_folder(osp.join(attribute_feat_path, user_id, date))
            create_folder(osp.join(attribute_pred_path, user_id, date))
            create_folder(osp.join(category_feat_path, user_id, date))
            create_folder(osp.join(category_pred_path, user_id, date))
            create_folder(osp.join(raw_feat_path, user_id, date))
            create_folder(osp.join(CAMs_folder_path, user_id, date))

 
def get_dataset_inputs(data_dir: str) -> List[str]:
    image_paths = []
    image_extensions = ['.jpeg', '.jpg', '.png']
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_extension = osp.splitext(file)[-1]
            if file_extension.lower() in image_extensions:
                image_path = osp.join(root, file)
                image_paths.append(image_path)
    return image_paths

    
def generate_input_params(data_dir: str, image_paths: List[str]) -> List[ExtractPlaceCNNFeatureParams]:
    feat_extension = '.npy'
    pred_extension = '.txt'
    input_params = []
    for image_path in image_paths:
        image_extension = osp.splitext(image_path)[-1]
        attr_feat_path = image_path.replace(data_dir, attribute_feat_path).replace(image_extension, feat_extension)
        attr_pred_path = image_path.replace(data_dir, attribute_pred_path).replace(image_extension, pred_extension)
        cate_feat_path = image_path.replace(data_dir, category_feat_path).replace(image_extension, feat_extension)
        cate_pred_path = image_path.replace(data_dir, category_pred_path).replace(image_extension, pred_extension)
        raw_feat_file_path = image_path.replace(data_dir, raw_feat_path).replace(image_extension, feat_extension)
        CAMs_path = image_path.replace(data_dir, CAMs_folder_path)

        params = ExtractPlaceCNNFeatureParams(
            image_file_path = image_path,
            raw_feat_output_path = raw_feat_file_path,
            output_attribute_feat = True, 
            attribute_feat_output_path = attr_feat_path,
            attribute_pred_output_path = attr_pred_path,
            output_category_feat = True,
            category_feat_output_path = cate_feat_path,
            category_pred_output_path = cate_pred_path,
            output_CAMs = True,
            CAMs_output_path = CAMs_path
        )
        input_params.append(params)
            
    return input_params


if __name__ == '__main__':
    create_output_folder()
    image_paths = get_dataset_inputs(data_dir)
    input_params = generate_input_params(data_dir, image_paths)
    total_training_images = len(input_params)

    # Start extracting features
    model = load_model()
    # for input_param in tqdm(input_params):
    #     extract_placeCNN_feature(input_param, model)
    func = partial(extract_placeCNN_feature, model=model)
    with Pool(3) as p:
        with tqdm(total=total_training_images) as pbar:
            for i, _ in enumerate(p.imap_unordered(func, input_params)):
                pbar.update()
            pass
