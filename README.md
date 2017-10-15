## Installing **CNTK** 
---
### Download CNTK from https://github.com/Microsoft/CNTK/releases/

* Extract zip file 
* Navigate to `Scripts\install\windows`
* Run `install.bat`
<br/>
<br/>

## Test Installation 
---

* Navigate to `Example\Image\Detection\FastRCNN`
* Run `install_data_and_model.py`
* Then Run `run_fast_rcnn.py`
<br/>
<br/>
## Tagging Custom Dataset
---
### Installing VOTT
* Download https://github.com/CatalystCode/VOTT/releases
* Extract and run the exe file 

### Tagging images

* Follow the Instruction from https://github.com/CatalystCode/VoTT#tagging-an-image-directory
<br/>
<br/>
## Train on Custom Dataset
---
* Store the output folder from the VOTT in `Example\Image\Dataset`
* Edit `Examples\Image\Detection\utils\annotations\annotations_helper.py`
> change the `data_set_path` variable to your VOTT output folder as show below  

```python
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.abspath(__file__))
    #data_set_path = os.path.join(abs_path, "../../../DataSets/Grocery")
    data_set_path = os.path.join(abs_path, "../../../DataSets/<YourDataset>")

    class_dict = create_class_dict(data_set_path)
    create_map_files(data_set_path, class_dict, training_set=True)
    create_map_files(data_set_path, class_dict, training_set=False)
```
* Run `python Examples\Image\Detection\utils\annotations\annotations_helper.py`

* Create `MyDataSet_config.py` file in `Examples\Image\Detection\utils\configs`

> Paste the following code in the file <br> Make Sure to change the value of `__C.DATA.NUM_TRAIN_IMAGES` and `__C.DATA.NUM_TEST_IMAGES` according to your number of train and test images in YourDataset folder


``` python

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
__C.DATA = edict()
cfg = __C

# data set config
__C.DATA.DATASET = "YourDataset" 
__C.DATA.MAP_FILE_PATH = "../../DataSets/<YourDataSet>"
__C.DATA.CLASS_MAP_FILE = "class_map.txt"
__C.DATA.TRAIN_MAP_FILE = "train_img_file.txt"
__C.DATA.TRAIN_ROI_FILE = "train_roi_file.txt"
__C.DATA.TEST_MAP_FILE = "test_img_file.txt"
__C.DATA.TEST_ROI_FILE = "test_roi_file.txt"
__C.DATA.NUM_TRAIN_IMAGES = 20
__C.DATA.NUM_TEST_IMAGES = 5
__C.DATA.PROPOSAL_LAYER_SCALES = [4, 8, 12]

# overwriting proposal parameters for Fast R-CNN
# minimum relative width/height of an ROI
__C.roi_min_side_rel = 0.04
# maximum relative width/height of an ROI
__C.roi_max_side_rel = 0.4
# minimum relative area of an ROI
__C.roi_min_area_rel = 2 * __C.roi_min_side_rel * __C.roi_min_side_rel
# maximum relative area of an ROI
__C.roi_max_area_rel = 0.33 * __C.roi_max_side_rel * __C.roi_max_side_rel
# maximum aspect ratio of an ROI vertically and horizontally
__C.roi_max_aspect_ratio = 4.0

# For this data set use the following lr factor for Fast R-CNN:
# __C.CNTK.LR_FACTOR = 10.0

```

* Edit 

* set `__C.RESULTS_NMS_THRESHOLD = 0.0`
* set `__C.VISUALIZE_RESULTS = TRUE`
* set `__C.IMAGE_WIDTH = 850`
* set `__C.IMAGE_HEIGHT = 850`

* Edit the following code in `visualize_detections` function in `plot_helper.py`

* change scale variable to:   `scale = 1080.0 / max(imgWidth, imgHeight)`
* change rect_scale variable `rect_scale = 1080 / pad_width`

> Add write to text file code as seen below make sure to change the .txt path 

``` python                  
            if iter == 0 and draw_negative_rois:
                drawRectangles(result_img, [rect], color=color, thickness=thickness)
            elif iter==1 and label > 0:
                thickness = 4
                file = open("C:/Users/vmadmin/Desktop/tem.txt","a")
                file.write(str(rect))
                file.write("\n")
                file.close() 
                drawRectangles(result_img, [rect], color=color, thickness=thickness)
            elif iter == 2 and label > 0:
                try:
                    font = ImageFont.truetype(available_font, 18)
                except:
                    font = ImageFont.load_default()
                text = classes[label]
                if roi_scores is not None:
                # =================== Add this =========================
                    file = open("C:/Users/vmadmin/Desktop/tem.txt","a")
                    file.write("[")
                    file.write(str(text))
                    file.write(" ")
                    file.write(str(round(score, 2)))
                    file.write("]")
                    file.write("\n")
                    file.close()    
                    text += "(" + str(round(score, 2)) + ")"
                # ====================================================== 
                result_img = drawText(result_img, (rect[0],rect[1]), text, color = (255,255,255), font = font, colorBackground=color)
    return result_img
```

* Add the below code to `plot_test_set_results` function in `plot_helper.py`

```python
        # apply regression and nms to bbox coordinates
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)
        nmsKeepIndices = apply_nms_to_single_image_results(regressed_rois, labels, scores,
                                                           use_gpu_nms=cfg.USE_GPU_NMS,
                                                           device_id=cfg.GPU_ID,
                                                           nms_threshold=cfg.RESULTS_NMS_THRESHOLD,
                                                           conf_threshold=cfg.RESULTS_NMS_CONF_THRESHOLD)

        filtered_bboxes = regressed_rois[nmsKeepIndices]
        filtered_labels = labels[nmsKeepIndices]
        filtered_scores = scores[nmsKeepIndices]

#=========== Add this ============================
        file = open("C:/Users/vmadmin/Desktop/tem.txt","w")
        file.write("")
        file.close() 
#=================================================
``` 

* After finishing editing the file make sure you activate the cntk environment by  running `cntkpy35.bat` in `cntk\Scripts`

* Then run the `run_faster_rcnn.py` file 
