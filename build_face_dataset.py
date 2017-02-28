from __future__ import print_function
from multiprocessing import Pool
from PIL import Image
import numpy as np
import animeface
import sys
import os


# im from PIL.Image.open, face_pos position object, margin
def faceCrop(im,face_pos,m):
    """
    m is the relative margin added to the face image
    """
    x,y,w,h = face_pos.x, face_pos.y, face_pos.width, face_pos.height
    sizeX, sizeY = im.size
    new_x, new_y = max(0,x-m*w), max(0,y-m*h)
    new_w = w + 2*m*w if sizeX > (new_x + w + 2*m*w) else sizeX - new_x
    new_h = h + 2*m*h if sizeY > (new_y + h + 2*m*h) else sizeY - new_y
    new_x,new_y,new_w,new_h = int(new_x),int(new_y),int(new_w),int(new_h)
    return im.crop((new_x,new_y,new_x+new_w,new_y+new_h))
    
def min_resize_crop(im, min_side):
    sizeX,sizeY = im.size
    if sizeX > sizeY:
        im = im.resize((min_side*sizeX/sizeY, min_side), Image.ANTIALIAS)
    else:
        im = im.resize((min_side, sizeY*min_side/sizeX), Image.ANTIALIAS)
    return im.crop((0,0,min_side,min_side))
    #return im

def load_detect(img_path):
    """Read original image file, return the cropped face image in the size 96x96

    Input: A string indicates the image path
    Output: Detected face image in the size 96x96

    Note that there might be multiple faces in one image, 
    the output crossponding to the face with highest probability
    """
    im = Image.open(img_path)
    faces = animeface.detect(im)
    prob_list = []
    len_f = len(faces)
    if len_f == 0:
        return 0
    for i in range(len_f):
        prob_list.append(faces[i].likelihood)
    prob_array = np.array(prob_list)
    idx = np.argmax(prob_array)
    face_pos = faces[idx].face.pos
    im = faceCrop(im, face_pos, 0.5)
    return min_resize_crop(im, 96)

def process_img(img_path):
    """
    The face images are stored in {${pwd} + faces} 
    """
    tmp = img_path.split('/')
    cls_name,img_name = tmp[len(tmp)-2], tmp[len(tmp)-1]
    new_dir_path = os.path.join('faces',cls_name)
    try:
        os.makedirs(new_dir_path)
    except OSError as err:
        print("OS error: {0}".format(err))

    new_img_path = os.path.join(new_dir_path, img_name)
    if os.path.exists(new_img_path):
        return 0
    im = load_detect(img_path)
    # no faces in this image
    if im == 0:
        return 0
    im.save(new_img_path, 'JPEG')

def try_process_img(img_path):
    try:
        process_img(img_path)
    except:
        e = sys.exc_info()[0]
        print('Err: %s \n' % e)

# multiprocessing version
def multi_construct_face_dataset(base_dir):
    cls_dirs = [f for f in os.listdir(base_dir)]
    imgs = []
    for i in xrange(len(cls_dirs)):
        sub_dir = os.path.join(base_dir, cls_dirs[i])
        imgs_tmp = [os.path.join(sub_dir,f) for f in os.listdir(sub_dir) if f.endswith(('.jpg', '.png'))]
        imgs = imgs + imgs_tmp
    print('There are %d classes, %d images in total. \n' % (len(cls_dirs), len(imgs)))
    pool = Pool(12) # 12 workers
    pool.map(try_process_img, imgs)


base_dir = '/home/jielei/gallery-dl/danbooru'
multi_construct_face_dataset(base_dir)