'''
REFERENCE: 
[1] https://github.com/rammyram/image_captioning/blob/master/Image_Captioning.ipynb 
- Download images and extract zips 

[2] https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/resize.py
- resize images


'''

import argparse
import os
from PIL import Image
from pycocotools.coco import COCO
import sys
import urllib
import zipfile 

# given an image resize it based on parameter "size"
#

# [2] resize based on source! 
def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    # resize images in image_dir and save to output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))
    print("RESIZING COMPLETED")





if __name__ == "__main__":

    '''
    NOTE! This is a combination of both sources: [1] for downloading the data and extracting .zips and [2] for resizing images. 
    '''


    #0 Create a project folder

    #1 DOWNLOAD 2014 train images [83K/13GB] from http://cocodataset.org/#download

    #create images folder into project folder
    os.makedirs('images' , exist_ok=True)

    #download images from coco
    train2014 = 'http://images.cocodataset.org/zips/train2014.zip'
    urllib.request.urlretrieve(train2014 , 'train2014' )

    #[1] Extract file to images folder. 
    with zipfile.ZipFile( 'train2014' , 'r' ) as zip_ref:
        zip_ref.extractall('images')

    #remove zip file from project root folder
    try:
        os.remove('train2014')
        print('zip removed')
    except:
        None

     
    #2 CREATE train2014_resized folder and run the code below.
    os.chdir("/images")
    os.makedirs('train2014_resized' , exist_ok=True) 
    #back to root
    os.chdir("../")

    # -> new resized images will be saved there. 
    #if there's a .ds-store problem -> just delete the file because it may interupt the process
    # save images to train2014_resized directory
    image_dir = 'images/train2014/'
    output_dir = 'images/train2014_resized/'
    image_size = [256,256] #new dimensions
    resize_images(image_dir, output_dir, image_size) #resize all images and save them to output_dir

    #3 DOWNLOAD 2014 train/val annotations [241MB] from http://cocodataset.org/#download
    train_anns = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    urllib.request.urlretrieve(train_anns , filename = 'annotations_trainval2014.zip' )

    #Extract .zip file to a root folder -> it creates a new directory "annotations"
    with zipfile.ZipFile('annotations_trainval2014.zip' , 'r') as zip_ref:
        zip_ref.extractall( '.'  )   #extract in this folder

    #remove zip.
    try:
        os.remove( 'annotations_trainval2014.zip' )
        print('zip removed')
    except:
        None

    
    


    ### BUILD VOCABULARY: 
    