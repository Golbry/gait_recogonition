# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:26:01 2021

@author: 86150
"""
# encoding: utf-8

import os
import sys

root_path = os.path.abspath(os.path.join('..'))
sys.path.append(root_path)
import _init_paths
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
# from PIL import Image
import cv2
from libs.networks.vgg_refinedet import VGGRefineDet
from libs.networks.resnet_refinedet import ResNetRefineDet
from libs.utils.config import coco512, MEANS
from libs.data_layers.transform import base_transform
from matplotlib import pyplot as plt

import pdb

is_gpu = False
if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	is_gpu = True

# for VOC
# class_names = ['__background__',
#                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#                'bus', 'car', 'cat', 'chair', 'cow',
#                'diningtable', 'dog', 'horse', 'motorbike', 'person',
#                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# class_names = ['__background__',
#                'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
#                'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
#                'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie',
#                'suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
#                'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
#                'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
#                'cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
#                'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
#                'clock','vase','scissors','teddy bear','hair drier','toothbrush']
class_names = ['__background__',
			   'person']
num_classes = len(class_names)

if __name__ == '__main__':
	# construct networks based on VGG16.
	cfg = coco512
	base_network = 'resnet101'
	model_path = '../output/resnet101_refinedet512_coco_400000.pth'
	print('Construct {}_refinedet network.'.format(base_network))
	refinedet = ResNetRefineDet(cfg['num_classes'], cfg)
	refinedet.create_architecture()
	# for CPU
	net = refinedet
	# for GPU
	if is_gpu:
		net = refinedet.cuda()
		cudnn.benchmark = True
	# load weights
	net.load_weights(model_path)
	net.eval()
	
	video_path='20210616140120.avi'
	cap=cv2.VideoCapture(video_path)


	fourcc=cv2.VideoWriter_fourcc(*'XVID')
	fps=cap.get(cv2.CAP_PROP_FPS)
	width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	out=cv2.VideoWriter('final_resnet.avi',fourcc,fps,(width,height))
	

	# image
	while cap.isOpened():
		ret, image = cap.read()
		if ret:
			pass
		else:
			print('no')
			break

	# preprocess
	# norm_image = base_transform(image, (320, 320), MEANS)
		norm_image = cv2.resize(image, (512, 512)).astype(np.float32)
		norm_image -= MEANS
		norm_image = norm_image.astype(np.float32)
		norm_image = torch.from_numpy(norm_image).permute(2, 0, 1)

	# forward
		input_var = Variable(norm_image.unsqueeze(0))  # wrap tensor in Variable
		if torch.cuda.is_available():
			input_var = input_var.cuda()
		detection = net(input_var)


	# scale each detection back up to the image,
	# scale = (width, height, width, height)
		scale = torch.Tensor(image.shape[1::-1]).repeat(2)
		threshold = 0.5
		num_top = detection.size(2)
		colors = (plt.cm.hsv(np.linspace(0, 1, num_classes)) * 255).tolist()
		for i in range(1, num_classes):
			for j in range(num_top):
				score = detection[0, i, j, 0]
				if score < threshold:
					continue
				label_name = class_names[i]
				display_txt = '%s: %.2f' % (label_name, score)
				pts = (detection[0, i, j, 1:] * scale).cpu().numpy().astype(np.int32)
				pts = tuple(pts)
				cv2.rectangle(image, pts[:2], pts[2:], colors[i], 2)
				cv2.putText(image, display_txt,
					pts[:2],
					cv2.FONT_HERSHEY_COMPLEX_SMALL,
					0.8, colors[i])
	# pdb.set_trace()
		# name, ext = os.path.splitext(img_path)
		# cv2.imwrite(name + '_result1' + ext, image)
		image = cv2.resize(image, (width, height))
		out.write(image)
	cap.release()
	out.release()
	cv2.destroyAllWindows()





