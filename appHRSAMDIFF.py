import copy
import sys

import numpy as np
import gradio as gr
from segment_anything.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from SimpleHRNet import SimpleHRNet
import requests
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline, DiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# mask_generator = SamAutomaticMaskGenerator(sam)

def detectbyHRNet(img,pth,joint_li):
    model = SimpleHRNet(48, 17, pth)
    joints = model.predict(img)
    dic_joint = {}
    list_joint = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
                  "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    print(joints.shape)
    joint_value = joints[0]
    for i in range(joint_value.shape[0]):
        dic_joint[list_joint[i]] = np.round(joint_value[i, :-1]).astype(int)
    print(dic_joint)
    name_li = []
    for joint_li_index in joint_li:
        if(joint_li_index == 'head'):
            name_li += ["nose", "left_eye", "right_eye"]
        elif(joint_li_index == 'left_arm'):
            name_li += ["left_shoulder", "left_elbow", "left_wrist"]
        elif(joint_li_index == 'right_arm'):
            name_li += ["right_shoulder", "right_elbow", "right_wrist"]
        elif(joint_li_index == 'left_leg'):
            name_li += ["left_hip", "left_knee", "left_ankle"]
        elif(joint_li_index == 'right_leg'):
            name_li += ["right_hip", "right_knee", "right_ankle"]
    np_promt = np.zeros((len(name_li), 2))
    for i in range(len(name_li)):
        np_promt[i, :] = dic_joint[name_li[i]].reshape(1, 2)
        # np_promt = np.append(np_promt, dic_joint[i].reshape(1,dic_joint[i].shape[0]),axis=0)

    np_promt = np_promt.astype(int)
    np_promt[:, [0, 1]] = np_promt[:, [1, 0]]
    return np_promt

def SegWithSAM(img, pth):
    sam = sam_model_registry["vit_h"](checkpoint=pth)
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    print('load')
    print(img.shape)
    mask_decode = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]))
    for i in range(len(masks)):
        temp_mask = masks[i]['segmentation'].astype(np.int)
        mask_decode += (i + 1) * temp_mask
    ratio = 255/np.max(mask_decode)
    mask_decode = np.round(mask_decode * ratio).astype(np.int)
    mask_show = np.zeros((mask_decode.shape[0], mask_decode.shape[1], 3))
    mask_show[:,:,0] = mask_decode
    mask_show[:,:,1] = mask_decode
    mask_show[:,:,2] = mask_decode
    mask_show = np.round(mask_show).astype(np.int)
    # np.save()
    print(mask_show)
    print(np.max(mask_show))
    merge_show = np.round(0.6*img + 0.4*mask_show)
    merge_show = merge_show.astype(np.int)
    print(merge_show)
    print(np.max(merge_show))
    # img = cv2.resize(img, (1024, 512),
    #                  interpolation=cv2.INTER_NEAREST)
    # print(img.shape)
    # img_data = torch.from_numpy(img/255).permute(2,0,1).unsqueeze(0).float()
    # print(img_data.shape)
    # mask = np.round(F.softmax(model(img_data),dim=1).squeeze(0).detach().numpy())
    # print(mask.shape)
    # mask_show = np.zeros((3, mask.shape[1], mask.shape[2]))
    # for i in range(mask.shape[0]):
    #     mask_show[0] += (i+1)*mask[i]
    # ratio = 255/np.max(mask_show)
    # mask_show = np.round(mask_show*ratio).astype(np.int)
    # mask_show[1] = mask_show[0]
    # mask_show[2] = mask_show[0]
    # mask_show = np.transpose((mask_show),(1,2,0))
    # index = np.where(mask_show == 0)
    # index2 = np.where(mask_show > 0)
    # merge_show = np.zeros((mask.shape[1], mask.shape[2], 3))
    # merge_show[index] = img[index]
    # merge_show[index2] = np.round(0.6*img[index2] + 0.4*mask_show[index2])
    # merge_show = merge_show.astype(np.int)
    # img[index] = 0
    # print(merge_show)
    return mask_show, merge_show, img
def Segwithpointprompt(img, pth_sam, pth_HR, name_li):
    li_prompt = []
    print(name_li)
    sam = sam_model_registry["vit_h"](checkpoint= pth_sam)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    np_prompt = detectbyHRNet(img, pth_HR, name_li)
    point_label = np.array([1]*np_prompt.shape[0])
    # point_label = [1]*len(li_prompt)
    # np_point = np.array(li_prompt)
    point_label = np.array(point_label)
    masks, scores, logits = predictor.predict(point_coords=np_prompt,
                                              point_labels=point_label,
                                              multimask_output=True)
    result_mask = (masks[0].astype(int)*255).astype(int)
    print(result_mask)
    print(np.max(result_mask))
    mask_img = np.zeros((result_mask.shape[0], result_mask.shape[1], 3))
    mask_img[:,:,0] = result_mask
    mask_img[:,:,1] = result_mask
    mask_img[:,:,2] = result_mask
    mask_img = mask_img.astype(int)
    index = np.where(mask_img[:,:,0] == 255)
    index_seg = np.where(mask_img == 0)
    seg_show = copy.deepcopy(img)
    seg_show[index_seg] = 0
    merge_show = copy.deepcopy(img)
    merge_show[:,:,0][index] += 50
    print(np.max(mask_img))

    return mask_img, merge_show, seg_show
def inpainting(img, masks, prompt):
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img
    mask = masks[:,:,0]
    w, h = mask.shape
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    image_pil = Image.fromarray(image)
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image_pil = image_pil.resize((512, 512))
    mask_pil = mask_pil.resize((512, 512))
    image = pipe(prompt=prompt, image=image_pil, mask_image=mask_pil).images[0]
    print(type(image))
    image = image.resize((h, w))
    return image
def save_img(img, path_save):
    print(img.shape)
    print(type(img))
    print(path_save)

    cv2.imwrite(path_save,img)
    return
def sizecontrol(img_seg, img_merge, img_diff):
    return img_seg, img_merge, img_diff

with gr.Blocks() as demo:
    gr.Markdown("Auto seg with HR--SAM.")
    with gr.Tab("Auto labeling by point"):
        with gr.Row():
            with gr.Column():
                pth_SAM = gr.Dropdown(['sam_vit_h_4b8939.pth'],
                                    label='SAM pth', info='choose SAM pth')
                pth_HRNet = gr.Dropdown(['pose_hrnet_w48_384x288.pth'],
                                        label='HRNet pth', info='choose HRNet pth')
            # pth_save = gr.Textbox('imgsavepth')
            #     joint_li = gr.Dropdown(["nose", "left_eye", "right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip",
            #   "right_hip","left_knee","right_knee","left_ankle","right_ankle"], label='choose the joint you want', info='choose point')
            #     joint_li = gr.CheckboxGroup(["nose", "left_eye", "right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip",
            #   "right_hip","left_knee","right_knee","left_ankle","right_ankle"], label='choose the joint you want', info='choose point')
                joint_li = gr.CheckboxGroup(
                    ["head", "left_arm", "right_arm", "left_leg", "right_leg"],
                    label='choose the joint you want', info='choose point')
                promp_text = gr.Textbox(label='prompt for stablediffusion')
                image_input_seg = gr.Image()
                with gr.Column():
                    with gr.Row():
                        button_segall = gr.Button('seg entire img')
                        button_segpoint = gr.Button('seg with HR point')
                    button_inpaint = gr.Button('diffusion with prompt')
                    button_nouse = gr.Button('control size(never mind)')
            with gr.Column():
                mask_output = gr.Image()
                seg_output = gr.Image()
                merge_output = gr.Image()
                inpaint_output = gr.Image()
    button_segall.click(SegWithSAM, inputs=[image_input_seg, pth_SAM], outputs=[mask_output,merge_output,seg_output])
    button_segpoint.click(Segwithpointprompt, inputs=[image_input_seg, pth_SAM, pth_HRNet, joint_li], outputs=[mask_output,merge_output,seg_output])
    button_inpaint.click(inpainting, inputs=[image_input_seg, mask_output, promp_text], outputs=[inpaint_output])
    button_nouse.click(sizecontrol, inputs=[seg_output,merge_output,inpaint_output], outputs=[seg_output,merge_output,inpaint_output])

    # text_button_1.click(reverse_text, inputs=text_input_1, outputs=text_output_1)

demo.launch()