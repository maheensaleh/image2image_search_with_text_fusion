import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"



# Loading a fine-tuned image captioning Transformer Model

# ViT Encoder-Decoder Model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
# Corresponding ViT Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# Image processor
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


# Image inference
def get_caption(images):
    # Preprocessing the Image
    img = image_processor(images, return_tensors="pt").to(device)
    # Generating captions
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)
    return caption

def load_image(image_path):
    image = Image.open(image_path)
    return image

# We have used greedy decoding which is the default. Other options might include beam search or multinomial sampling. You can experiment with them and see the difference.

if __name__ == "__main__":
    
    from dataloader_assg import get_data_loader, get_query_image
    from dataloader_coco import get_data_loader_train, get_data_loader_val
    # img = load_image("dataset\hlcv_assg\model\obj91__0.png")
    # cap = get_caption(model, image_processor, tokenizer, img)
    # print(cap)
    
    # print("Batch inference for assg images")
    # data_loader = get_data_loader("dataset\hlcv_assg\query", batch_size=4, shuffle=False )
    # for imgs, img_paths in data_loader:
    #     imgs = []
    #     for img_path in img_paths:
    #         imgs.append(load_image(img_path))
    #     caps = get_caption(model, image_processor, tokenizer, imgs)
    #     print(img_paths)
    #     print(caps)
    #     break
    
    
    print("Batch inference for coco images")
    coco_data_loader = get_data_loader_train(batch_size=4, shuffle=False)
    for imgs_vis, imgs_cap, targets, fp in coco_data_loader:

        caps = get_caption(model, image_processor, tokenizer, imgs_cap)
        print(targets)
        print(caps)
        print(fp)
        break
        
    