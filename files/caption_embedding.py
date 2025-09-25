from sentence_transformers import SentenceTransformer
from infersent_files.models import InferSent
import torch
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nltk
nltk.download('punkt')

#for infersent
V = 2
infersent_model_path = '/content/drive/MyDrive/UdS-DSAI/HLCV/project/infersent_files/infersent2.pkl'
infersent_params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent_model = InferSent(infersent_params_model)
infersent_model.load_state_dict(torch.load(infersent_model_path))

W2V_PATH = '/content/drive/MyDrive/UdS-DSAI/HLCV/project/infersent_files/glove.840B.300d.txt'
infersent_model.set_w2v_path(W2V_PATH)


#for sentence bert
sent_embed_model_name = "all-MiniLM-L6-v2"
# sent_embed_model_name = "multi-qa-MiniLM-L6-cos-v1"
sentence_embed_model = SentenceTransformer(sent_embed_model_name)


# for universal sent encoder
univ_sent_enc_module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
univ_sent_enc_model = hub.load(univ_sent_enc_module_url)
print ("module %s loaded" % univ_sent_enc_module_url)


def sentenceEmbeddings(captions, model):

    if model == "all-MiniLM-L6-v2":
      return sentence_embed_model.encode(captions)[0]

    elif model == "infersent":
      return  infersent_model.encode(captions)

    elif model == "universal_sent_encoder":
      return univ_sent_enc_model(captions)[0]


if  __name__ == "__main__": 
    
    from dataloader_coco import get_data_loader_train, get_data_loader_val
    from image_captioning import get_caption

    model = SentenceTransformer('all-MiniLM-L6-v2')
    # multi-qa-MiniLM-L6-cos-v1

    #Our sentences we like to encode
    sentences = ['This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.']

    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    #Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
        
        
    # test coco image captions
    coco_data_loader = get_data_loader_train(batch_size=4, shuffle=False)
    for imgs_vis, imgs_cap, targets, fp in coco_data_loader:

        caps = get_caption(imgs_cap)
        print(targets)
        print(caps)
        print(fp)
        break
        
