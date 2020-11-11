#<args:img_dirpath><args: annot_path><save_path>
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from pycocotools import mask
from os import listdir
import cv2
import json
import argparse
import os,random
import argparse


parser = argparse.ArgumentParser(
    description='Pasonatron deafult trainer')

parser.add_argument('img_dir', type=str,
                    help="path to image directory")

parser.add_argument('coco_ann_path', type=str,
                    help="path to json file")
parser.add_argument('save_path', type=str,
                    help="path to save coco preview")


args = parser.parse_args()





def loadImage(path):
    n_imgList = []
    imgList = listdir(path)
    imgList.sort()
    for x in imgList:
        print(x)
        if '.' in x:
            print(x)
            x_name, x_ext = x.split(".")
            if x_ext in ("png", "jpg", "JPEG"):  # add new image extention as required
                n_imgList.append(x)
    return n_imgList


#coco- annotation json file
# conf = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/output_time_check/coco_data/coco_annotations.json"

#base path
# base_path = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/output_time_check/coco_data/"
#directory
# save_path = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/output_time_check/"

# Read coco_annotations config
with open(os.path.join(args.img_dir, args.coco_ann_path)) as f:
    annotations = json.load(f)
    categories = annotations['categories']
    annotations = annotations['annotations']
img_list = loadImage(args.img_dir)
imgList = []
for file_name in img_list:
    f_name, ext = file_name.split(".")
    f_name2, img_id = f_name.split("_")
    # print(img_id)
    image_idx = int(img_id)
    # print(type(image_idx))
    # int(image_idx)
    im_path = os.path.join(args.img_dir, f"rgb_{img_id}.png")
    if os.path.exists(im_path):
        im = Image.open(im_path)
    else:
        im = Image.open(im_path.replace('png', 'jpg'))

    def get_category(_id):
        category = [category["name"]
                    for category in categories if category["id"] == _id]
        if len(category) != 0:
            return category[0]
        else:
            raise Exception("Category {} is not defined in {}".format(
                _id, os.path.join(args.img_dir, args.conf)))

    font = ImageFont.load_default()
    # Add bounding boxes and masks
    for idx, annotation in enumerate(annotations):
        if annotation["image_id"] == image_idx:
            draw = ImageDraw.Draw(im)
            bb = annotation['bbox']
            draw.rectangle(
                ((bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3])), fill=None, outline="red")
            draw.text((bb[0] + 2, bb[1] + 2),
                      get_category(annotation["category_id"]), font=font)
            if annotation["iscrowd"]:
                im.putalpha(255)
                an_sg = annotation["segmentation"]
                item = mask.decode(mask.frPyObjects(
                    an_sg, im.size[1], im.size[0])).astype(np.uint8) * 255
                item = Image.fromarray(item, mode='L')
                overlay = Image.new('RGBA', im.size)
                draw_ov = ImageDraw.Draw(overlay)
                draw_ov.bitmap((0, 0), item, fill=(255, 0, 0, 128))
                im = Image.alpha_composite(im, overlay)
            else:
                for item in annotation["segmentation"]:
                    # item = annotation["segmentation"][0]
                    poly = Image.new('RGBA', im.size)
                    pdraw = ImageDraw.Draw(poly)
                    pdraw.polygon(item, fill=(255, 255, 255, 127),
                                  outline=(0, 255, 0, 255))
                    im.paste(poly, mask=poly)
    im.save(os.path.join(args.save_path, 'coco_annotated_{}.png'.format(image_idx)), "PNG")
    opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    imgList.append(opencvImage)
    type(im)
    height,width,c = opencvImage.shape
    cv2.imshow("Test", opencvImage)
    cv2.waitKey(100)
    print(f"Image saved {image_idx}")
    # im.show()


vid_path = os.path.join(args.save_path, "coco-preview.avi")
out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))
print("Creating a preview video")
for i in range(len(imgList)):

    out.write(imgList[i])
    # cv2.imshow("test", imgList[i])
    # cv2.waitKey(1000)

out.release()
print("Coco preview video is created.")
print(f"Coco preview path: {vid_path}")
