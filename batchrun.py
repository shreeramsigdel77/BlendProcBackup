
import os
import time

def new_dir(path:str,directoryname:str):
    try:
        os.makedirs(path+"/"+directoryname)
    except OSError as e:
        print(f"Folder name with {directoryname} do exist.")
        # if e.errno != errno.EEXIST:
        #     raise
    return (path+"/"+directoryname)
    


#blendProc
blendProc_config_path = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/config2.yaml"
blend_file_path = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/crescent.blend"
# blend_file_path = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/cube.blend"
blend_mesh_name = "crescent.002"
# blend_mesh_name = "Cube"
base_prj_dir = "/home/pasonatech/Blender2Detectron/BlenderProc-master/projectCrescent/crescent_test2/output_time_check"



#create new directory if doesnt exisst
train_img_dir = new_dir(base_prj_dir, "train")
val_img_dir = new_dir(base_prj_dir, "val")











no_of_sample_image = 1 #generates 1 image at a time
no_of_train_img = 5 #generates (no of sample_img * no_of_train_img)
no_of_val_img = 4 #generates (no of sample_img * no_of_val_img)

#turn blenderPorc on off
train_dataset = ""
val_dataset = ""
#comment out true if you do not want to create dataset
train_dataset = "True"
val_dataset = "True"


#set true if you need a  coco_preview data
coco_preview = ""
coco_preview = "True"

#set true if neede for pasonatron
pasonatron = ""
pasonatron ="True"


start = time.time()
#train data
if train_dataset:
    for i in range(no_of_train_img):
    #blender proc
        os.system(
            f"python run.py {blendProc_config_path} {blend_file_path} {blend_mesh_name} {train_img_dir} {no_of_sample_image}"
            )
blendProc_time = time.time() - start
print(blendProc_time)

#val data
if val_dataset:
    for i in range(no_of_val_img):
    #blender proc
        os.system(
            f"python run.py {blendProc_config_path} {blend_file_path} {blend_mesh_name} {val_img_dir} {no_of_sample_image}"
            )

#No need to change;





#fix train coco
print("START")
train_img_dir_coco =train_img_dir+"/coco_data/"
val_img_dir_coco=val_img_dir+"/coco_data/"
print(train_img_dir_coco)

org_coco_anno = train_img_dir_coco+"coco_annotations.json"
new_coco_anno = train_img_dir_coco +"fixed_coco_annotations.json"


#fix val coco
val_coco_anno = val_img_dir_coco+"coco_annotations.json"
n_val_coco_anno = val_img_dir_coco+"fixed_coco_annotations.json"

print("*********Fixing annotation*********")

#fix train coco
os.system(
    f"python /home/pasonatech/Blender2Detectron/pasonatron/test/ram/fix_coco_json.py {train_img_dir_coco} {org_coco_anno} {new_coco_anno}"
)

#fix val coco
os.system(
    f"python /home/pasonatech/Blender2Detectron/pasonatron/test/ram/fix_coco_json.py {val_img_dir_coco} {val_coco_anno} {n_val_coco_anno}"
)
fixCoco_time = time.time() - blendProc_time


print("*********coco annotation*********")

#create new directory for coco-preview
train_img_preview = new_dir(train_img_dir, "coco_preview_dir")
val_img_preview = new_dir(val_img_dir, "coco_preview_dir")


#coco preview
print(train_img_dir_coco)
if coco_preview:
    os.system(
        f"python /home/pasonatech/Blender2Detectron/BlenderProc-master/preview_coco.py {train_img_dir_coco} {org_coco_anno} {train_img_preview}"
    )
print("Test")
#coco preview
if coco_preview:
    os.system(
        f"python /home/pasonatech/Blender2Detectron/BlenderProc-master/preview_coco.py {val_img_dir_coco} {val_coco_anno} {val_img_preview}"
    )

#pasonatron
det_weight_dir = base_prj_dir+"/detectron_weight_dir1"


#pasonatron coco
if pasonatron:
    os.system(
        f"python /home/pasonatech/Blender2Detectron/pasonatron/test/ram/bbox.py {train_img_dir_coco} {new_coco_anno} {det_weight_dir}"
    )
    train_time = time.time() - fixCoco_time

    print(f"Weight saved directory: {det_weight_dir}")




print("Process time")





# print(f"BlendProc data creation time {blendProc_time}")
# print(f"fix_coco {fixCoco_time}")
# print(f"train Time {train_time}")



# print(time.time()-start)




####### ap calulate
inference_dir = new_dir(base_prj_dir, "inference_results")
print(new_coco_anno)
print(val_img_dir_coco)
print(val_coco_anno)
print(det_weight_dir)
os.system(
    f"python /home/pasonatech/Blender2Detectron/jaitool/ap_cal.py {new_coco_anno} {val_img_dir_coco} {n_val_coco_anno} {det_weight_dir} {inference_dir}"
)


