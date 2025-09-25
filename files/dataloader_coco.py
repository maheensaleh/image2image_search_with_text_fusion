import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pycocotools.coco import COCO
import PIL
import os


# Define the image pre-processing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the desired size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])




# Custom dataset class
class CocoCustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_path,transform=None):
        self.coco = CocoDetection(root=image_dir, annFile=annotation_path, transform=transform)
        self.coco_annotation = COCO(annotation_file=annotation_path)
        self.image_dir = image_dir
        cat_ids = self.coco_annotation.getCatIds()
        self.cat_names =  self.coco_annotation.loadCats(cat_ids)


    def __getitem__(self, index):
        
        # get img for CNN
        img_vis, target = self.coco[index]
        
        #get img for captioning
        img_info = self.coco_annotation.loadImgs([index+1])[0]
        img_file_name = img_info["file_name"]
        img_cap = PIL.Image.open(os.path.join(self.image_dir, img_file_name)).convert("RGB")

        # get class labels        
        class_labels = [int(x['category_id']) for x in target]
        class_labels = set(class_labels)
        class_names = [self.cat_names[i]["name"] for i in class_labels ]
        return img_vis,img_cap, class_names, img_file_name

    def __len__(self):
        return len(self.coco)
    
    



def get_data_loader_train(train_dir,annotation_path_train,batch_size, shuffle):
    # Create the COCO custom dataset for training set
    coco_dataset_train = CocoCustomDataset(train_dir, annotation_path_train, transform=preprocess)
    data_loader_train = DataLoader(coco_dataset_train, batch_size=batch_size, shuffle=shuffle , collate_fn=lambda x: tuple(zip(*x)))
    return data_loader_train

def get_data_loader_val(valid_dir,annotation_path_val,batch_size, shuffle):
    # Create the COCO custom dataset for validation set
    coco_dataset_val = CocoCustomDataset(valid_dir, annotation_path_val, transform=preprocess)
    data_loader_val = DataLoader(coco_dataset_val, batch_size=batch_size, shuffle=shuffle , collate_fn=lambda x: tuple(zip(*x)))
    return data_loader_val


if __name__ == "__main__":
  
    # Define the path to the COCO dataset
    data_dir = '/content/drive/MyDrive/UdS-DSAI/HLCV/project/dataset'
    train_dir = f'{data_dir}/train/data'  # or 'val2017' for validation set
    valid_dir = f'{data_dir}/valid/data'  # or 'val2017' for validation set
    annotation_path_train = f'{data_dir}/train/labels.json'
    annotation_path_val = f'{data_dir}/valid/labels.json'


    data_loader_train = get_data_loader_train(train_dir, batch_size=32, shuffle=True)
    data_loader_val = get_data_loader_val(valid_dir,batch_size=32, shuffle=True)
    
    print("Number of batches in the training set:", len(data_loader_train))
    print("Number of batches in the validation set:", len(data_loader_val))

    # Iterate over the data loader for training set
    for imgs_vis, imgs_cap,targets,fp in data_loader_train:
        # Process the images and targets as needed
        print("Training batch shape:", len(imgs_vis))
        print("Training batch shape:", len(imgs_cap))
        print("Training batch shape:", len(fp))
        print(fp[0])
        print("Training batch shape:", len(targets))
        break

    # # Iterate over the data loader for validation set
    for imgs_vis, imgs_cap,targets,fp  in data_loader_val:
        # Process the images and targets as needed
        print("Validation batch shape:", len(imgs_vis))
        print("Validation batch shape:", len(imgs_cap))
        print("Validation batch shape:", len(targets))
        break