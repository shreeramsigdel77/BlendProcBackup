from annotation_utils.coco.structs import COCO_Dataset
from common_utils.path_utils import get_script_dir
import argparse

parser = argparse.ArgumentParser(description='Fix coco json file for pasonatron package')

parser.add_argument('img_dir', type=str,
                    help="path to image directory")
parser.add_argument('coco_ann_path', type=str,
                    help="path to json file")
parser.add_argument('coco_ann_output', type=str,
                    help="path to save a fixed json file")

args = parser.parse_args()






# img_dir = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/output3/coco_data/"
# coco_ann_path = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/output3/coco_data/coco_annotations.json"
# coco_ann_output ="/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/output3/coco_data/coco_annotations1.json"



# dataset = COCO_Dataset.load_from_path(coco_ann_path, strict=False, img_dir=get_script_dir())
dataset = COCO_Dataset.load_from_path(args.coco_ann_path, strict=False, img_dir=args.img_dir)
dataset.save_to_path(args.coco_ann_output, strict=True, overwrite=True)
print("Coco Json Fixed Successfully ")
