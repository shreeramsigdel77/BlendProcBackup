#<args:img_dir><args:fixed_json_file><args:module_save_path>
from common_utils.path_utils import get_script_dir
from annotation_utils.coco.structs import COCO_Dataset
from pasonatron.det2.lib.train.bbox import Detectron2BBoxTrainer
from pasonatron.det2.lib.trainer import COCO_BBox_Trainer
import argparse



parser = argparse.ArgumentParser(
    description='Pasonatron deafult trainer')

parser.add_argument('img_dir', type=str,
                    help="path to train image directory")

parser.add_argument('coco_ann_path', type=str,
                    help="path to fixed json file")

parser.add_argument('module_save_path', type=str,
                    help="path to save a weight files")

args = parser.parse_args()



# img_dir = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/output3/coco_data"
# coco_ann_path = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/output3/coco_data/coco_annotations1.json"



trainer = Detectron2BBoxTrainer(
    coco_ann_path=args.coco_ann_path,
    img_dir=args.img_dir,
    model_name='faster_rcnn_R_50_FPN_3x',
    # trainer_constructor= DefaultTrainer,
    # trainer_constructor=COCO_BBox_Trainer, # aug included
    images_per_batch=1,
    max_iter=150,
    base_lr=0.003,
    batch_size_per_image=512,
    # output_dir='test_crescent2_output3',
    output_dir=args.module_save_path,
    min_size_train=1024,
    max_size_train=1024,
    checkpoint_period=5000
    
)
trainer.train(resume=False)


##### ap calculation


