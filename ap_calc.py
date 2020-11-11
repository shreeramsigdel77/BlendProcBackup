import os
from pycocotools.coco import COCO
from cocoeval import COCOeval
# from pycocotools.cocoeval import COCOeval
import numpy as npdef run(path, gt_path=None):
# gt_path = os.path.abspath(f'{path}/../../json/gt.json')
gt_dataset = COCO(annotation_file=gt_path)
dt_dataset = gt_dataset.loadRes(path)    # evaluator = COCOeval(cocoGt=gt_dataset, cocoDt=dt_dataset, iouType='keypoints')
# evaluator = COCOeval(cocoGt=gt_dataset, cocoDt=dt_dataset, iouType='keypoints')
# evaluator.params.useSegm = None
# evaluator.params.kpt_oks_sigmas = np.array([1.0]*12)/12    evaluator = COCOeval(cocoGt=gt_dataset, cocoDt=dt_dataset, iouType='bbox')
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()def main():
path = '/home/jitesh/3d/data/coco_data/hook_real2/img_07_09_13_Detection_h6_500_0099999_0.1_s1500_vis_infer_output_50_1x/infered_hook.json'
path = '/home/jitesh/3d/data/coco_data/hook_real2/img_07_12_23_Keypoints_h8_500_0099999_0.1_s1500_vis_infer_output_50_1x/infered_hook.json'
path = '/home/jitesh/3d/data/coco_data/hook_real2/img_07_15_10_Keypoints_h8_500_0049999_0.1_s1500_vis_infer_output_50_1x/infered_hook.json'
path = '/home/jitesh/3d/data/test_data/hexagon_bolts/img_b6_1_1x_0029999.pth_thres0.2_11_04_09_test/result.json'
# path = '/home/jitesh/3d/data/test_data/hexagon_bolts/img_bolt_0_1x_0019999.pth_thres0.2_10_27_16_test/result.json'
# path = '/home/jitesh/3d/data/test_data/hexagon_bolts/img_b4_5_1x_0019999.pth_thres0.2_10_27_16_test/result.json'
gt_path = os.path.abspath(f'{path}/../../json/gt.json')





run(path, gt_path)if __name__ == "__main__":
main()