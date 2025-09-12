from align import align_ncc, pyramid_align
import numpy as np
import skimage as sk
import skimage.io as skio
import os

jpg = ["cathedral.jpg", "monastery.jpg", "tobolsk.jpg"]
# emir uses edge for better performance
tif_part1 = ["church.tif", "emir.tif", "harvesters.tif", "icon.tif", "italil.tif"] 
# lastochikino and lugano uses edge for better performance
tif_part2 = ["lastochikino.tif", "lugano.tif", "melons.tif", "self_portrait.tif", "siren.tif", "three_generations.tif"]
extra = ["detali_sobora.tif", "locomotive.tif", "monrepos_park.tif", "village_dagestan.tif"]
fix = ["lastochikino.tif", "lugano.tif", "emir.tif"]

def picture_dimensions(pic_list):
    for pic in pic_list:
        write(pic)
        imname = '1/proj1_data/'+pic
        im = skio.imread(imname)
        im = sk.img_as_float(im)
        height = np.floor(im.shape[0] / 3.0).astype(int)

        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        row, col = b.shape
        b_cropped = b[int(col*0.1): int(col - 0.1*col), int(0.1*row):int(row - 0.1*row)] 
        g_cropped = g[int(col*0.1): int(col - 0.1*col), int(0.1*row):int(row - 0.1*row)] 
        r_cropped = r[int(col*0.1): int(col - 0.1*col), int(0.1*row):int(row - 0.1*row)] 
        
        ag = align_ncc(b_cropped, g_cropped, 0, 0) # use for naive align (low-res)

        # ag = pyramid_align(b_cropped, g_cropped, 5, False) # uncomment for tif 
        # specific images that don't work with regular pyramid_ncc_align need to use edge mapping instead -> change False to True
        write("green shift: " + str(ag))
        g_shifted = np.roll(np.roll(g, ag[1], axis=0), ag[0], axis=1)

        ar = align_ncc(b_cropped, r_cropped, 0, 0) # use for naive align (low-res)

        # ar = pyramid_align(b_cropped, r_cropped, 5, False) # uncomment for tif
        # specific images that don't work with regular pyramid_ncc_align need to use edge mapping instead -> change False to True
        write("red shift: " + str(ar))
        r_shifted = np.roll(np.roll(r, ar[1], axis=0), ar[0], axis=1)

        # create a color image (stack images on top of each other)
        im_out = np.dstack([r_shifted, g_shifted, b])
        save_image(pic, im_out)

"""
====================
Helper Functions
====================
"""

def save_image(name, image):
    name, _ = os.path.splitext(name)
    im_out_uint8 = sk.img_as_ubyte(image)
    fname = '1/final_result/'+name+'.jpg'
    skio.imsave(fname, im_out_uint8)


def write(info):
    with open("1/results.txt", "a") as f: 
        f.write(info+'\n')

picture_dimensions(jpg)

