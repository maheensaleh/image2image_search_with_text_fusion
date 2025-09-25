import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import os


# Define the image pre-processing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the desired size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Custom dataset class
class ImageCustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Get the list of image file paths
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, image_path

    def __len__(self):
        return len(self.image_paths)



# Create the data loader
def get_data_loader(image_folder, batch_size, shuffle, preprocess=preprocess):
    image_dataset = ImageCustomDataset(image_folder, transform=preprocess)
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def get_query_image(image_path):
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)
    return image



if __name__ == "__main__":
    
    # Define the path to the image folder
    image_folder = 'dataset/hlcv_assg/model'

    # Create the data loader for training set
    data_loader  = get_data_loader(image_folder, batch_size=32, shuffle=True)

    # Iterate over the data loader
    for images, image_paths in data_loader:
        # Process the images and image_paths as needed
        print("Batch shape:", images.shape)
