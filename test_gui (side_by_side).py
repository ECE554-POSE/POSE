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
import os
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

if USBCam:
	from jetcam.usb_camera import USBCamera
else:
	from jetcam.csi_camera import CSICamera


class POSE_GUI:
	def __init__(self):

		# Context variable declarations and loading		
		self.running = False
		self.WIDTH = 224
		self.HEIGHT = 224
		self.thresh = 127
		self.iteration = 0
		self.minimum_joints = 1
		self.path = './images/'
		
		# Load all image filenames in image directory into a list
		self.levels = []
		for r, d, f in os.walk(self.path):
			for file in f:
				if '.jpg' in file or '.png' in file:
					self.levels.append(self.path + file)
		print(self.levels)
		
		self.mdelay_sec = 10
		self.mtick = self.mdelay_sec
		
		#Loading model and model data
		with open('./tasks/human_pose/human_pose.json', 'r') as f:
			human_pose = json.load(f)

		self.topology = trt_pose.coco.coco_category_to_topology(human_pose)

		self.num_parts = len(human_pose['keypoints'])
		self.num_links = len(human_pose['skeleton'])
		
		self.data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).cuda()

		self.OPTIMIZED_MODEL = './tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
		self.model_trt = TRTModule()
		self.model_trt.load_state_dict(torch.load(self.OPTIMIZED_MODEL))

		self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
		self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
		self.device = torch.device('cuda')
	
		self.parse_objects = ParseObjects(self.topology)
		self.draw_objects = DrawObjects(self.topology)

		# Start camera
		if USBCam:
			self.camera = USBCamera(width=self.WIDTH, height=self.HEIGHT, capture_fps=30)
		else:
			self.camera = CSICamera(width=self.WIDTH, height=self.HEIGHT, capture_fps=30)

		# Creating main GUI
		self.root = tk.Tk()
		self.root.title('POSE')
		self.root.geometry(str(self.WIDTH*3+100)+"x"+str(self.HEIGHT + 150))

		# Organizing GUI
		# Rows
		self.top_row = tk.Frame(self.root)
		self.im_row = tk.Frame(self.root)
		self.but_row = tk.Frame(self.root)
		self.top_row.pack(side=tk.TOP, fill=tk.Y, padx = 5, pady = 5)
		self.im_row.pack(side=tk.TOP, fill=tk.Y, padx = 5, pady = 5)
		self.but_row.pack(side=tk.BOTTOM, fill=tk.Y, padx = 5, pady = 5)

		# Top row
		self.timer_label = tk.Label(self.top_row, text='Countdown Timer')
		self.timer_label.pack(side=tk.TOP, padx=5, pady=0)

		# Image row
		self.lmain = tk.Label(self.im_row)
		self.lmain.pack(side=tk.LEFT, padx=0, pady=0)
		self.mmain = tk.Label(self.im_row)
		self.mmain.pack(side=tk.RIGHT, padx=0, pady=0)
		self.rmain = tk.Label(self.im_row)
		self.rmain.pack(side=tk.RIGHT, padx=0, pady=0)

		# Button row
		# Create and pack (show) the starting buttons
		self.start_button = tk.Button(self.but_row, text= 'Start Game', command=self.game_start)
		self.start_button.pack(side=tk.TOP, padx=5, pady=0)
		self.exit_button = tk.Button(self.but_row, text= 'Quit', command=self.exit_app)
		self.exit_button.pack(side=tk.BOTTOM, padx=5, pady=0)

		# Create other buttons but leave them hidden
		self.pose_button = tk.Button(self.but_row, text= 'Estimate Pose', command=self.pose_estimate)
		self.stop_button = tk.Button(self.but_row, text= 'Stop Game', command=self.game_stop)

	#Camera Displays Here
	#Need to add button that cals pose estimation routine
	def main_loop(self):
		if self.running:
			img = self.camera.read()
			self.img = img # Load image into context variable
			img = Image.fromarray(img)
			imgtk = ImageTk.PhotoImage(image=img)
			self.lmain.imgtk = imgtk
			self.lmain.configure(image=imgtk)		
			self.root.after(10, self.main_loop)

	def pose_estimate(self):
		# Run pose estimation and display overlay
		img = self.img
		data = self.preprocess(img)
		cmap, paf = self.model_trt(data)
		cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
		counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
		self.draw_objects(img, counts, objects, peaks)

		# Extract point coordinates
		topology = self.topology
		height = img.shape[0]
		width = img.shape[1]
		objcnt = 0
		count = int(counts[0])
		points = []
		for i in range(count):
			obj = objects[0][i]
			C = obj.shape[0]
			for j in range(C):
				k = int(obj[j])
				if k>= 0:
					peak = peaks[0][j][k]
					x = round(float(peak[1]) * width)
					y = round(float(peak[0]) * height)
					points.insert(objcnt, [x, y])
					objcnt = objcnt+1


		img = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=img)
		self.rmain.imgtk = imgtk
		self.rmain.configure(image=imgtk)

		return points

	def pose_score(self, points, mask):
		if len(points) < self.minimum_joints:
			return None # Return a score of 0 if no pose detected
		
		correct = 0
		# Locate points in mask and mark if point is over mask or not
		for point in points:
			xi = point[0]
			yi = point[1]
			point_val = mask[yi, xi]
			print("Point: "+str(xi)+", "+str(yi))
			# print("OBJx = ",xi)
			# print("OBJy = ",yi)
			# print(point_val)
			if point_val >= 255:
				print ('Correct!')
				correct = correct + 1
			else:
				print ('Wrong!')
		score = (correct / len(points)) * 100
		img2 = Image.fromarray(mask)
		#self.draw_objects(img2, counts, objects, peaks)
		imgtk2 = ImageTk.PhotoImage(image=img2)
		self.mmain.imgtk = imgtk2
		self.mmain.configure(image=imgtk2)
		return score

	def state_handler(self):
		print("State Handler")

		# Load mask desired for current iteration
		level = self.levels[self.iteration]
		print(level)
		mask_img=cv2.imread(level, cv2.IMREAD_GRAYSCALE)
		mask=cv2.threshold(mask_img, self.thresh, 255, cv2.THRESH_BINARY)[1]

		# Run pose estimation and return list of identified coordinates
		calc_points = self.pose_estimate()
		
		# Get a score based on correct points over mask
		# TODO: Retrieve a score into a variable
		score = self.pose_score(calc_points, mask)

		if score is not None:		
			print ("You scored " + str(score))
		else:
			print("Didn't detect a pose from player.")
		
		# Change iteration counter to move to next image for next time it is called
		if self.iteration < (len(self.levels) - 1):
			self.iteration = self.iteration + 1
		else:
			self.iteration = 0

	def countdown_handler(self):
		if self.running:
			lbl = "Time Remaining {}".format(self.mtick)
			#print(lbl)
			self.timer_label['text'] = lbl
			self.mtick = self.mtick - 1
			if self.mtick == 0:
				self.state_handler
				self.mtick = self.mdelay_sec
				self.state_handler()
			self.root.after(1000, self.countdown_handler)

	def game_start(self):
		print("Game time started")
		# Hide the start button and place the estimate pose button and stop buttons instead
		self.start_button.pack_forget()
		self.pose_button.pack(side=tk.TOP, padx=5, pady=0)
		self.stop_button.pack(side=tk.TOP, padx=5, pady=0)
		self.running = True # Context flag to let the loop code know to repeat itself
		self.countdown_handler() # Function that loops to control pose estimation loop
		self.main_loop() # Function that repeats itself to continously query the camera for a new image every 10 ms

	def game_stop(self):
		print("Game time stopped")
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
