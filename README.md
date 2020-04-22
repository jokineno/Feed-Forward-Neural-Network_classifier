# 6 different Neural Networks projects
- These are projects in which I focused on learning PyTorch framework in the context of NLP. These are really helpful projects to get started with deep learning. After doing these projects PyTorch framework becomes really intuitive and it is easy to continue with more challenging tasks. 
## Project 1: Twitter Sentiment Classification 
- Tweets are classified on each of three classes: positive, neutral, negative
- Logistic Regression used as a classifier
- Data can be found from the root of this repository. 

## Project 2:  Twitter Sentiment Classification
- Similar to project 1 but instead of Logistic Regression Feed-Forward Neural Network is used. 
- Also, pretrained word embeddings are used to increase the understand about similarity of words

## Project 3: Language Identification
- 6 different Uralic languages is identified based on a word sample.
- FFNN used as a model. Softmax is computed over the vocabulary to extract probabilities of target classes. 
- In addition to projects 1 and 2 minibatch training is added 

## Project 4: Digit Recognition from MNIST data. 
- Data can be downloaded from MNIST with Pytorch function or straight from the source page. 
- 6 different models are trained
- The differences of models is in regularization and optimization techniques. 
- See images of models' figures of digits.

## Project 5: Convolutional Neural Networks - Classify Images in 10 categories
- CNN is trained to classify images from 10 different categories
- .sh file is a script for PUHTI which is a powerful cloud computing cluster in Espoo, Finland. See www.csc.fi
- What is interesting, the learned filters are also plotted. See images in the folder. 
- 
## Project 6: Recurrent Neural Networks: Language Identification
- 3 different RNNs are used to classify a language based on an input word.

## Final project: Image captioning
- Implemenent encoder-decoder network to create caption from image inputs. 
- Pretrained CNN is used to extract a feature vector from image. 
- Long-Short-Term-Memory module works as an encoder, which is used to predict a caption based on extracted feature vector by CNN. 
- The implementation is based on the article Show and Tell by Google Engineers. 
- The citing and references are included in .py files. Big thanks for all the previous work.

NOTE: The code skeletons are provided by course stuff at Introduction to Deep Learning at University of Helsinki but final implementations are my own work. 
