import tkinter as tk
from functools import partial
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
USBCam = 0

if USBCam:
	from jetcam.usb_camera import USBCamera
else:
	from jetcam.csi_camera import CSICamera


class POSE_GUI:
	def state_handler(self):
		print("State Handler")
		self.pose_estimate()

	def countdown_handler(self):

		lbl = "Time Remainging {}".format(self.mtick)
		print(lbl)
		self.timer_label['text'] = lbl
		self.mtick = self.mtick - 1
		if self.mtick == 0:
			self.state_handler
			self.mtick = self.mdelay_sec
			self.state_handler()
		
		if self.running:
			self.root.after(1000, self.countdown_handler)


	def __init__(self):

		with open('./tasks/human_pose/human_pose.json', 'r') as f:
			human_pose = json.load(f)

		self.topology = trt_pose.coco.coco_category_to_topology(human_pose)

		self.num_parts = len(human_pose['keypoints'])
		self.num_links = len(human_pose['skeleton'])

		self.WIDTH = 224
		self.HEIGHT = 224

		self.data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).cuda()


		self.mdelay_sec = 10
		self.mtick = self.mdelay_sec

		#Creation of modelor
		#model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

		self.OPTIMIZED_MODEL = './tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

		#torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

		self.model_trt = TRTModule()
		self.model_trt.load_state_dict(torch.load(self.OPTIMIZED_MODEL))

		self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
		self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
		self.device = torch.device('cuda')

	
		self.parse_objects = ParseObjects(self.topology)
		self.draw_objects = DrawObjects(self.topology)

		self.root = tk.Tk()
		self.root.title('POSE')
		self.root.geometry(str(self.WIDTH*2+100)+"x"+str(self.HEIGHT + 150))

		if USBCam:
			self.camera = USBCamera(width=self.WIDTH, height=self.HEIGHT, capture_fps=30)
		else:
			self.camera = CSICamera(width=self.WIDTH, height=self.HEIGHT, capture_fps=30)

		# Organizing GUI
		# Rows
		self.im_row = tk.Frame(self.root)
		self.but_row = tk.Frame(self.root)
		self.im_row.pack(side=tk.TOP, fill=tk.Y, padx = 5, pady = 5)
		self.but_row.pack(side=tk.BOTTOM, fill=tk.Y, padx = 5, pady = 5)

		# Image row
		self.lmain = tk.Label(self.im_row)
		self.lmain.pack(side=tk.LEFT, padx=0, pady=0)
		self.rmain = tk.Label(self.im_row)
		self.rmain.pack(side=tk.RIGHT, padx=0, pady=0)

		# Button row
		# Create and pack (show) the starting buttons
		self.start_button = tk.Button(self.but_row, text= 'Start Game', command=self.game_start)
		self.start_button.pack(side=tk.TOP, padx=5, pady=0)
		self.exit_button = tk.Button(self.but_row, text= 'Quit', command=self.exit_app)
		self.exit_button.pack(side=tk.BOTTOM, padx=5, pady=0)

		self.timer_label = tk.Label(self.but_row, text='Countdown Timer')
		
		# Create other buttons but leave them hidden
		self.pose_button = tk.Button(self.but_row, text= 'Estimate Pose', command=self.pose_estimate)
		self.stop_button = tk.Button(self.but_row, text= 'Stop Game', command=self.game_stop)

		self.running = False
		
		self.objx = []
		self.objy = []


		


			

	#Camera Displays Here
	#Need to add button that cals pose estimation routine
	def main_loop(self):
		img = self.camera.read()
		self.img = img
		# data = self.preprocess(img)
		# cmap, paf = self.model_trt(data)
		# cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
		# counts, objects, peaks = self.padrse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
		# self.draw_objects(img, counts, objects, peaks)
		#imgdraw = cv2.cvtColor(data ,cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=img)
		self.lmain.imgtk = imgtk
		self.lmain.configure(image=imgtk)

		if self.running:		
			self.root.after(10, self.main_loop)

	def pose_estimate(self):
		img = self.img
		data = self.preprocess(img)
		cmap, paf = self.model_trt(data)
		cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
		counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)

		self.draw_objects(img, counts, objects, peaks)



		topology = self.topology
		height = img.shape[0]
		width = img.shape[1]
		objcnt = 0
		K = topology.shape[0]
		count = int(counts[0])
		for i in range(count):
			color = (0,255,0)
			obj = objects[0][i]
			C = obj.shape[0]
			for j in range(C):
				k = int(obj[j])
				if k>= 0:
					peak = peaks[0][j][k]
					x = round(float(peak[1]) * width)
					y = round(float(peak[0]) * height)
					self.objx.insert(objcnt, x)
					self.objy.insert(objcnt, y)
					objcnt = objcnt+1
					print("OBJ #",objcnt)
					print("OBJx = ",x)
					print("OBJy = ",y)


		img = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=img)
		self.rmain.imgtk = imgtk
		self.rmain.configure(image=imgtk)

	def game_start(self):
		print("Game time started")
		# Hide the start button and place the estimate pose button and stop buttons instead
		self.start_button.pack_forget()
		self.pose_button.pack(side=tk.TOP, padx=5, pady=0)
		self.stop_button.pack(side=tk.TOP, padx=5, pady=0)
		self.running = True # Context flag to let the loop code know to repeat itself
		#self.root.after(self.mdelay_sec*1000, self.state_handler)
		self.countdown_handler()
		self.main_loop() # Function that repeats itself to continously query the camera for a new image every 10 ms

	def game_stop(self):
		print("Game time stopped")
		self.mtimerflag = 0
		# Hide the stop and estimate buttons and show start button again
		self.stop_button.pack_forget()
		self.start_button.pack(side=tk.TOP, padx=5, pady=0)
		self.pose_button.pack_forget()
		self.running = False # Set context flag to false to have code stop repeating itself

	def preprocess(self, img):
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		image = PIL.Image.fromarray(image)
		image = transforms.functional.to_tensor(image).to(self.device)
		image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
		return image[None, ...]				

	def exit_app(self):
		self.root.destroy()

		

if __name__ == '__main__':
	POSE = POSE_GUI()
	POSE.root.mainloop() # Tk main loop to keep window up
