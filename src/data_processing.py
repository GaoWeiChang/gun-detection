import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class GunDataset(Dataset):
    def __init__(self, root:str, device:str="cpu"):
        self.image_path = os.path.join(root, "Images")
        self.label_path = os.path.join(root, "Labels")
        
        self.image_name = sorted(os.listdir(self.image_path))
        self.label_name = sorted(os.listdir(self.label_path))
        
        self.device = device
        logger.info("Data Processing Initialized...")

    def __getitem__(self, idx):
        try:
            logger.info(f"Loading Data for index {idx}")
            
            # load and process image
            image_path =  os.path.join(self.image_path , str(self.image_name[idx]))
            logger.info(f"Image Path : {image_path}")
            image = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)

            image_res = img_rgb/255 # normalize
            image_res = torch.as_tensor(image_res).permute(2,0,1) # convert from (H,W,C) to (C,H,W)

            # load label
            label_name = self.image_name[idx][:-4] + "txt" # ex. 113.jpeg -> 113.txt
            label_path = os.path.join(self.label_path , str(label_name))
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found : {label_path}")
            
            box = []
            with open(label_path,"r") as label_file:
                n_object = int(label_file.readline())
                for i in range(n_object):
                    box.append(list(map(int,label_file.readline().split())))  # bounding box position

            # store informations
            target = {} # store target info. eg. box, area, ...
            bbox_area = [] # area of bounding box
            bbox_label = [] # label of each bounding box

            for i in range(len(box)):
                res = (box[i][2]-box[i][0]) * (box[i][3]-box[i][1])
                bbox_area.append(res)
                bbox_label.append(1) # assign same label for all box

            # combine all information and store in target dict
            target["boxes"] = torch.as_tensor(box, dtype=torch.float32)
            target["area"] =  torch.as_tensor(bbox_area, dtype=torch.float32)
            target["image_id"] = torch.as_tensor([idx])
            target["labels"] = torch.as_tensor(bbox_label , dtype=torch.int64)

            # move to device
            image_res = image_res.to(self.device)
            for key in target:
                if isinstance(target[key], torch.Tensor):
                    target[key] = target[key].to(self.device)

            return image_res, target
        
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data ",e)

    def __len__(self):
        return len(self.image_name)

if __name__ == "__main__":
    root_path = "artifacts/raw"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = GunDataset(root=root_path, device=device)
    image,target = dataset[0]

    print("Image Shape : ", image.shape)
    print("Target Keys : " , target.keys())
    print("Bounding boxes : ",target["boxes"])
