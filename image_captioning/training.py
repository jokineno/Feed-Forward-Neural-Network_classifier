
'''
REFERENCES: 
[1] https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning 
- loading data, building encoder-decoder, training
- I'm not using command line arguments as in the source. 

[2] https://pytorch.org/docs/stable/torchvision/models.html
- Transformation for pretrained models

[3] https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py
- build vocabulary
'''

import os
import sys
import pickle
import argparse
from collections import Counter
import torch
import torch.utils.data as data
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



#Vocabulary, used pad, start,end
# based on [3] with modifications
class Vocabulary(object):
    def __init__(self):
        self.word2idx = dict()
        self.idx2word = dict()
        self.idx = 0
        self.padword = "<pad>"
        self.startword = "<start>"
        self.endword = "<end>"
        self.unknown = "<unk>"

        # add a new word and increase a size of index by 1. 
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    # return length with len(object)
    def __len__(self):
        return len(self.word2idx)


#[1]
#Create a dataset based on annotations
class CocoDataset(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    #return image and target pair while iterating over the data. -> see collate_fn
    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            # transformation based on Pytorch instructions. 
            image = self.transform(image)
            
        # words to list
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)

# source [1]
# collate_fn creates minibatches from a list of images and target and returns (images, targets, lengths)
def collate_fn(data):
    
    # Sort by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    #merge images into a stack
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


# source [1]
#creates a dataset from json file with coco class and turns is into a torch.utils.data.DataLoader. 
def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

# source [1]
#luo sanasto, parametrina json, joka on sana ja thresholdina sanan pituus
def build_vocab(json_file, threshold):
    coco = COCO(json_file)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        image_ann = coco.anns[id]
        caption = str(image_ann['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # add words longer than threshold
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Vocabulary and add words
    vocab = Vocabulary()
    vocab.add_word(vocab.padword)
    vocab.add_word(vocab.startword)
    vocab.add_word(vocab.endword)
    vocab.add_word(vocab.unknown)

    # add words to vocabulary
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


# source [1] different model
#pretrained CNN -> extract vector based on image, use pretrained resNet-34
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet34(pretrained=True) #34
        modules = list(resnet.children())[:-1]   # don't use the last fully connected layer. 
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        #extract feature vector from image
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# source [1]
#build LSTM DECODER. 
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.unit = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)   
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    #decode and create captions
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.unit(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    #generate captions and use a greedy search.
    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.unit(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


if __name__ == "__main__":
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('CUDA IS AVAILABLE')
    else:
        print("NO CUDA AVAILABLE")

    #transformation for images. 
    #[2] see :https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([ 
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])



    #build vocabulary
    vocab = build_vocab(json='opt/cocoapi/annotations/captions_train2014.json', threshold=int(4))
    vocab_path = 'opt/vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print("Saved to '{}'".format(vocab_path))
    print("Size: {}".format(len(vocab)))

    # try to read from a file
    try:
        with open('opt/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except: 
        vocab = vocab

    data_loader = get_loader('images/train2014_resized/', 'annotations/captions_train2014.json', vocab, 
                                transform, 128,
                                shuffle=False, num_workers=2) 

    encoder = EncoderCNN(512).to(device)
    decoder = DecoderRNN(512, 1024, len(vocab), 1).to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.SGD(params, lr=0.001)

    # source [1], with modifications. 
    ## TRAIN THE MODEL
    epochs = 3
    total_step = len(data_loader)
    for epoch in range(epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            print("ROUND",i,"out of ", len(data_loader))
            
            

            #zero the gradients
            decoder.zero_grad()
            encoder.zero_grad()

            #set train mode
            encoder.train()
            decoder.train()

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            # Print log info
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch, epochs, i, total_step, loss.item())) 
                    
            # Save the model checkpoints
            if (i+1) % 1500 == 0:
                torch.save(decoder.state_dict(), os.path.join(
                        'models/', 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                        'models/', 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

    #save also the last step
    torch.save(decoder.state_dict(), os.path.join('models/', 'decoder-3-final.ckpt'.format(epoch+1, i+1)))
    torch.save(encoder.state_dict(), os.path.join('models/', 'encoder-3-final.ckpt'.format(epoch+1, i+1)))