'''
REFERENCE: 
[1] https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning 
- sampling the captions, without command line arguments
'''

import pickle
import argparse
from collections import Counter
import torch
import torch.utils.data as data
import os
import numpy as np
import nltk
nltk.download('punkt')
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from training import Vocabulary #it is needed for reading the vocab.pkl file. 


# [1], no command line arguments
def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

if __name__ == "__main__":

    # Normalize instructions:  https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([ 
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])
    
    
    encoder = EncoderCNN(512).eval()  # encoder does not see the image captions
    decoder = DecoderRNN(512, 1024, len(vocab), 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #models are trained -> down
    encoder.load_state_dict(torch.load('encoder-3-final.ckpt',map_location=lambda storage, loc: storage))
    decoder.load_state_dict(torch.load('decoder-3-final.ckpt',map_location=lambda storage, loc: storage))

    image = load_image('images/train2014_resized/COCO_train2014_000000000089.jpg', transform)
    print(image.shape)
    image_tensor = image.to(device)
    print(image_tensor.shape)


    feature = encoder(image_tensor) #extract feature vector from image
    sampled_ids = decoder.sample(feature) #decode to word indexes.
    sampled_ids = sampled_ids[0].cpu().numpy()
    sampled_caption = []

    # read vocabulary
    with open('opt/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)

    #create a sentence from word ids
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    print (sentence)

    # plot the corresponding image.
    image = Image.open('images/train2014_resized/COCO_train2014_000000000089.jpg')
    plt.imshow(np.asarray(image))











