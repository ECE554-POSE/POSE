import tkinter as tk
from functools import partial
#from jetcam.csi_camera import CSICamera
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg
from PIL import Image
from PIL import ImageTk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
import cv2
import torchvision.transforms as transforms
import PIL.Image
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects


#Change this flag for using USB camera
USBCam = 1

class POSE_GUI:
	def __init__(self):

		with open('../trt_pose/tasks/human_pose/human_pose.json', 'r') as f:
			human_pose = json.load(f)

		self.topology = trt_pose.coco.coco_category_to_topology(human_pose)

		self.num_parts = len(human_pose['keypoints'])
		self.num_links = len(human_pose['skeleton'])

		self.WIDTH = 224
		self.HEIGHT = 224

		self.data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).cuda()

		#Creation of model
		#model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

		self.OPTIMIZED_MODEL = '../trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

		#torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

		self.model_trt = TRTModule()
		self.model_trt.load_state_dict(torch.load(self.OPTIMIZED_MODEL))

		self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
		self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
		self.device = torch.device('cuda')

	
		self.parse_objects = ParseObjects(self.topology)
		self.draw_objects = DrawObjects(self.topology)

		self.WIDTH = 224
		self.HEIGHT = 224
		self.root = tk.Tk()
		self.root.title('POSE')
		self.root.geometry(str(self.WIDTH)+"x"+str(self.HEIGHT + 50))




		if USBCam:
			self.camera = USBCamera(width=self.WIDTH, height=self.HEIGHT, capture_fps=30)
		else:
			self.camera = CSICamera(width=self.WIDTH, height=self.HEIGHT, capture_fps=30)




		#self.camera.running = True
		self.start_button = tk.Button(self.root, text= 'Start Game', command=self.game_start)
		self.start_button.pack(side=tk.TOP, padx=5, pady=5)
		self.exit_button = tk.Button(self.root, text= 'Quit', command=self.exit_app)
		self.exit_button.pack(side=tk.TOP, padx=5, pady=5)
		#Button for starting pose estimation
		#self.capture_button = tk.Button(self.root,text= 'Capture', command=self.process_POSE)
		#self.capture_button.pack(side=tk.TOP, padx=5, pady=5)
		self.lmain = tk.Label(self.root)
		self.lmain.pack()
	
	#Camera Displays Here
	#Need to add button that cals pose estimation routine
	def main_loop(self):
		img = self.camera.read()
		data = self.preprocess(img)
		cmap, paf = self.model_trt(data)
		cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
		counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
		self.draw_objects(img, counts, objects, peaks)
		#imgdraw = cv2.cvtColor(data ,cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=img)
		self.lmain.imgtk = imgtk
		self.lmain.configure(image=imgtk)
		self.root.after(10, self.main_loop)

	def game_start(self):
		print("Game time started")
		# Need to either start a Tkinter screen with the camera feed or use a matplotlib for it
		self.main_loop() # Function that repeats itself to continously query the camera for a new image every 10 ms

	def preprocess(self, img):
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		image = PIL.Image.fromarray(image)
		image = transforms.functional.to_tensor(image).to(self.device)
		image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
		return image[None, ...]
			
	# def process_POSE(self, img):
	# 	data = self.preprocess(img)
	# 	cmap, paf = self.model_trt(data)
	# 	cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
	# 	counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
	# 	self.draw_objects(data, counts, objects, peaks)
	# 	imgdraw = cv2.cvtColor(data ,cv2.COLOR_BGR2RGB)
	# 	return imgdraw
				

	def exit_app(self):
		self.camera.running = False
		self.camera.unobserve_all()
		self.root.destroy()

		

if __name__ == '__main__':
	POSE = POSE_GUI()
	POSE.root.mainloop() # Tk main loop to keep window up
