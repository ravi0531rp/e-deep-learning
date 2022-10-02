from yolact import Yolact
from data import cfg, MEANS, STD, set_cfg, mask_type
import torch.nn.functional as F
import torch
import cv2
from layers.output_utils import postprocess
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from loguru import logger
from glob import glob
import argparse

# python generate_overlayed_images_script.py --config yolact_plus_base_config --padding 50 --trained_model /home/ubuntu/yolact/datasets/yolact_plus.pth --files /home/ubuntu/yolact/datasets/train_data --out_files  /home/ubuntu/yolact/datasets/train_data_segmented2

class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()

        self.mean = torch.Tensor(MEANS).float()
        self.std  = torch.Tensor( STD ).float()

        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std  = self.std.cuda()
        self.mean = self.mean[None, :, None, None]
        self.std = self.std[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)
        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, (cfg.max_size, cfg.max_size), mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        
        img = img[:, (2, 1, 0), :, :].contiguous()

        # Return value is in channel order [n, c, h, w] and RGB
        return img


def evalimage(image_id:int, net:Yolact, path:str):
    img =cv2.imread(path)
    h,w,_ = img.shape
    frame = torch.from_numpy(img).float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    batch = batch.cuda()
    preds = net(batch)

    classes, scores, boxes, masks = postprocess(preds, w, h, crop_masks=True, score_threshold=0)

    classes = list(classes.cpu().detach().numpy().astype(int))
    if isinstance(scores, list):
        box_scores = list(scores[0].cpu().detach().numpy().astype(float))
        mask_scores = list(scores[1].cpu().detach().numpy().astype(float))
    else:
        scores = list(scores.detach().cpu().detach().numpy().astype(float))
        box_scores = scores
        mask_scores = scores
    masks = masks.view(-1, h*w)
    boxes = boxes.cpu().detach().numpy()
    masks = masks.view(-1, h, w).cpu().detach().numpy()

    facet_bbox_list =[]
    mask_list=[]
    scores_list = []
    classes_list = []
    for i in range(masks.shape[0]):
        # Make sure that the bounding box actually makes sense and a mask was produced
        if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:

            if mask_scores[i]>0.4:
                bbox = boxes[i,:]
                bbox = [round(float(x)*10)/10 for x in bbox]
                facet_bbox_list.append(bbox)
                scores_list.append(mask_scores[i])
                classes_list.append(classes[i])
                mask_list.append(masks[i,:,:].astype(np.uint8))

    return facet_bbox_list, mask_list, classes_list, scores_list

def generate_building_masks(image_path, net):
    _, mask_list, classes_list, _ = evalimage(1,net,image_path)
    logger.info(classes_list)
    
    build_masks = []
    for index in range(len(classes_list)):
        if classes_list[index] == 0:
            build_masks.append(mask_list[index])
    logger.debug(f"Number of building masks are .. {len(build_masks)}")
    return build_masks

def create_overlayed_image(image_path, build_masks, padding = 50, save_dir = None):
    if not len(build_masks):
        logger.warning(f"{image_path} generated no buildings.. ") 

    else:
        try:
            os.makedirs(save_dir)
        except:
            pass
        
        image = cv2.imread(image_path)
        save_name = image_path.split("/")[-1].split(".")[0] 
        length, width, _ = image.shape

        best_cont_areas = []
        best_conts = []
        for mask in build_masks:
            mask_copy = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.int16)
            mask_copy[:, :, 0] = mask
            mask_copy = cv2.convertScaleAbs(mask_copy)
            contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours):
                contour = max(contours, key=cv2.contourArea)
                best_cont_areas.append(cv2.contourArea(contour))
                best_conts.append(contour)

        logger.info(f"Num of masks is {len(build_masks)} ; Num of conts in lst is {len(best_cont_areas)}")
        best_cont_idx = best_cont_areas.index(max(best_cont_areas))
        best_cont = best_conts[best_cont_idx]

        contour_image = np.zeros_like(image)

        contour_image = cv2.drawContours(contour_image, [best_cont], 0, (255,255,255), -1)
        overlayed = cv2.bitwise_and(image, image, mask = contour_image[:,:,0])
        
        x,y,w,h = cv2.boundingRect(best_cont)

        left_x = max(0, x - padding)
        left_y = max(0, y - padding)
        right_x = min(width, x + w +  padding)
        right_y = min(length, y + h + padding)

        cropped_padded = overlayed[left_y : right_y , left_x : right_x]

        cv2.imwrite(os.path.join(save_dir, save_name + "_cropped.jpg") , cropped_padded )


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("-c", "--config", type=str, default="yolact_plus_base_config")
    my_parser.add_argument("-p", "--padding", type=int, default=50)
    my_parser.add_argument("-m", "--trained_model", type=str, required=True)
    my_parser.add_argument("-f", "--files", type=str, default="/home/ubuntu/yolact/datasets/train_data")
    my_parser.add_argument("-o", "--out_files", type=str, default="/home/ubuntu/yolact/datasets/train_data_segmented")

    args = vars(my_parser.parse_args())

    config = args["config"]
    padding = args["padding"]
    trained_model = args["trained_model"]
    files = args["files"]
    out_files = args["out_files"]

    set_cfg(config)
    cfg.mask_proto_debug = False
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = Yolact()

    net.load_weights(trained_model)
    net =net.cuda()
    net.eval()
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False

    files = files + "/*.jpg"
    files = glob(files)

    try:
        os.makedirs(out_files)
    except:
        pass
    
    for idx, file in enumerate(files):
        try:
            build_masks = generate_building_masks(file, net)
            create_overlayed_image(file, build_masks, padding = padding, save_dir = out_files)   
        except Exception as e:
            logger.error(e)
        logger.success(f"Successfully written for total images = {idx+1}")