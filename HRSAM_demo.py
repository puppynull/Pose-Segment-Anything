import numpy as np
from segment_anything.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from SimpleHRNet import SimpleHRNet
import requests
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("HR-Segment-Anything Demo", add_help=True)
    # parser.add_argument(
    #     "--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", required=True, help="path to checkpoint file"
    # )
    # parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    # parser.add_argument(
    #     "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    # )

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg

    # config_file = args.config  # change the path of the model config file
    # grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    # sam_checkpoint = args.sam_checkpoint
    # image_path = args.input_image
    # text_prompt = args.text_prompt
    # output_dir = args.output_dir
    # box_threshold = args.box_threshold
    # text_threshold = args.text_threshold
    # device = args.device

    # image = cv2.imread(image_path)
    image = cv2.imread('img/boxing.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = SimpleHRNet(48, 17, 'pose_hrnet_w48_384x288.pth')
    joints = model.predict(image)
    dic_joint = {}
    list_joint = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
                  "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    print(joints.shape)
    joint_value = joints[0]
    for i in range(joint_value.shape[0]):
        dic_joint[list_joint[i]] = np.round(joint_value[i, :-1]).astype(int)
    print(dic_joint)
    # joint_li = ["nose", "left_eye", "right_eye"]
    joint_li = ["left_shoulder", "left_elbow", "left_wrist"]
    np_promt = np.zeros((len(joint_li), 2))
    for i in range(len(joint_li)):
        np_promt[i, :] = dic_joint[joint_li[i]].reshape(1, 2)
        # np_promt = np.append(np_promt, dic_joint[i].reshape(1,dic_joint[i].shape[0]),axis=0)

    np_promt = np_promt.astype(int)
    np_promt[:, [0, 1]] = np_promt[:, [1, 0]]
    point_label = np.array([1] * np_promt.shape[0])
    sam = sam_model_registry["vit_h"](checkpoint='sam_vit_h_4b8939.pth')
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(point_coords=np_promt,
                                              point_labels=point_label,
                                              multimask_output=True)
    mask = masks[0]
    output_dir = 'outputs'
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    show_mask(mask, plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "HR_sam_output_boxingarm.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
