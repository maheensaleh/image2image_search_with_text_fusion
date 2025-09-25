import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from dataloader_assg import get_data_loader
from dataloader_coco import get_data_loader_train, get_data_loader_val



  # Load pre-trained models
# vgg_model = models.vgg16(pretrained=True)
resnet_model = models.resnet50(pretrained=True)
# model = models.inception_v3(pretrained=True)
# model = models.mobilenet_v2(pretrained=True)
efficientnet_model = models.efficientnet_b0(pretrained=True)

# Function to obtain visual embeddings using different models
def get_visual_embedding(image_batch, model_name):
    # Load pre-trained models
    if model_name == "efficientnet":
        model = efficientnet_model
    elif model_name == "resnet50":
        model = resnet_model


    model = nn.Sequential(*list(model.children())[:-1])
    # Set the model to evaluation mode
    model.eval()

    # Generate visual embeddings
    with torch.no_grad():
        embeddings = model(image_batch).squeeze().numpy()

    return embeddings

if __name__ == "__main__":

    # print("for assg images")
    # model_images = 'dataset\hlcv_assg\model'
    # batch_size = 4
    # model_dataLoader = get_data_loader(model_images, batch_size, shuffle=False)
    # # get visual embedding
    # model_name = "resnet50"  # Choose the desired model name
    # for images, paths in model_dataLoader:
    #     embedding = get_visual_embedding(images, model_name)
    #     print(embedding.shape)
    #     break
    
    print("for coco images")
    train_data_loader = get_data_loader_train(batch_size=4, shuffle=False)
    model_name = "resnet50"  # Choose the desired model name
    for imgs_vis, imgs_cap, targets, fp in train_data_loader:
        images = torch.stack(list(imgs_vis), dim=0)
        embedding = get_visual_embedding(images, model_name)
        print(embedding.shape)
        break
    
    
    # 
    

