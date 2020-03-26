- Contents: 
	- The code is divided in three parts: preprocessing, training and sampling
	- .sh file was used for running the code in puhti. 
	- encoder and decoder files are the trained models from puhti. 3 epochs and learning rate 0.001.
	- .pdf final report in which i'm using the provided overleaf template.  
	- THERE ARE NO IMAGES AND ANNOTATIONS IN THIS FOLDER
		- if you want to run the project see cocodataset.com/#download



- Sources: 
- There are 3 main sources: 
- [1] https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
- [2] https://github.com/rammyram/image_captioning/blob/master/Image_Captioning.ipynb
- [3] https://pytorch.org/docs/stable/torchvision/models.html

I have put sources in to .py files but no in the final report because there is no need to cite them in the text. The most of the work is based on source [1] but preprocessing and dowloading data was done with the help of source [2]. 
Source [3] is used for performing normalizing the images. 


Dependencies: 
There are a bunch of external libraries but pycocotools is a special one. 
Before running the code install pycocotools with pip or then follow instructions from this source https://github.com/cocodataset/cocoapi. 



If you want to run the whole project run: 
1. Pre_process.py -> downloads the images and annotations
2. training.py -> build vocabulary and models and trains them -> use cloud clusters or a smaller dataset if you want to do it on your local machine. 
3. Run sample.py -> select an image and see how the generated caption describes the corresponding image. 

Also if you're citing this work, give credits to original authors that I have cited already in this work. 