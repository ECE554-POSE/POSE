import tkinter as Tk
from tkinter import messagebox
from button_panel import ButtonPanel
from functools import partial
from jetcam.utils import bgr8_to_jpeg
from PIL import Image
from PIL import ImageTk
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

class MainGUI:

	def __init__(self, root):

		# Context variable declarations and loading		
		self.running = False
		self.WIDTH = 224
		self.HEIGHT = 224
		self.thresh = 127
		self.iteration = 0
		self.minimum_joints = 1
		self.path = './images/'		
		self.mdelay_sec = 10
		self.mtick = self.mdelay_sec
		self.mask = None
		
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
		
		self.frame=Tk.Frame(root)
		self.root=root
		# Create editable title 
		self.titleVar = Tk.StringVar()
		self.title= Tk.Label(root, textvariable=self.titleVar, font="Verdana 36")
		self.titleVar.set("Pose Estimation Game")
		self.title.pack(side=Tk.TOP)
		self.frame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)	
		
		# Create image capture figure
		# Set as Frame with three possible images (live feed, mask/pose to make, image captured)
		# Image row
		self.im_row = Tk.Frame(self.frame)
		self.feed_label = Tk.Label(self.im_row)
		self.feed_label.pack(side=Tk.LEFT)
		self.mask_label = Tk.Label(self.im_row)
		self.pose_label = Tk.Label(self.im_row)
		
		# Create editable description label
		self.desTextVar = "Please select an option from the right"
		self.desText = Tk.Label(self.frame, text=self.desTextVar, font="Verdana 12")
		


		#Create Combobox for selection (Steps are currently in comments)
		#Grab maps from repository
		#Parse map names to develop choices
		#group map names into array
		self.levels = []
		for r, d, f in os.walk(self.path):
			for file in f:
				if '.jpg' in file or '.png' in file:
					self.levels.append(self.path + file)
		print(self.levels)
		#choices = {'These', 'are', 'temporary', 'choices'}
		#Put map names in combo box	
		self.ddVar = Tk.StringVar()
		self.ddVar.set('Select a Choice')
		self.dropDown = Tk.OptionMenu(self.frame, self.ddVar, *self.levels)
		self.ddVar.trace('w', self.mask_select)


		
		# Create inital button panel
		self.buttonPanel = ButtonPanel(root)


		self.im_row.pack()
		self.desText.pack()
		self.buttonPanel.pack()
		self.root.after(10, self.camera_loop)
		MainGUI.updateToTitle(self)

	def updateToTitle(self):
		# unpack unused widgets
		self.dropDown.pack_forget()
		self.mask_label.pack_forget()
		self.pose_label.pack_forget()

		# build title frame
		self.titleVar.set("Pose Estimation Game")
		self.desText.configure(text ="Please select an option from the right")
		

		# set button commands
		self.buttonPanel.button1.configure(text="Pose Now!", command=lambda:MainGUI.updateToSelect(self))
		self.buttonPanel.button2.configure(text="Exit",command=self.root.destroy)



	def updateToSelect(self):
		# unpack unused widgets
		self.pose_label.pack_forget()

		# Reset mask object
		self.mask_label.configure(image='')
		self.mask = None

		# Build select frame
		self.titleVar.set("Select Your Pose")
		self.desText.configure(text ="Select an Option from Below")
		self.dropDown.pack()

		# Show mask (or representative image) from selected choice
		self.mask_label.pack(side=Tk.LEFT)

		# set button commands
		self.buttonPanel.button1.configure(text="Select", command=lambda:MainGUI.updateToPose(self))
		self.buttonPanel.button2.configure(text="Main Menu", command=lambda:MainGUI.updateToTitle(self))



	def updateToPose(self):
		# Check a mask was selected first. If not, display error and stay at previous state
		if self.mask is None:
			messagebox.showerror("No mask loaded", "Please select a course on the dropdown menu")
			return
		
		# unpack unused widgets
		self.dropDown.pack_forget()


		self.titleVar.set("Pose Now")
		self.buttonPanel.button1.configure(text="Main Menu", command=MainGUI.blankCommand)
		self.buttonPanel.button2.configure(text=" ", command=MainGUI.blankCommand)
		timer = 10
		self.desText.configure(text = "Time to Evaluation: " + str(timer) + "s")
		self.root.after(1000, lambda:MainGUI.countDown(self, timer))
		
	def pose_estimate(self):
		# Run pose estimation and display overlay
		img = self.img
		data = self.preprocess(img)
		cmap, paf = self.model_trt(data)
		cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
		counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
		self.draw_objects(img, counts, objects, peaks)

		# Extract point coordinates
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
		self.pose_label.imgtk = imgtk
		self.pose_label.configure(image=imgtk)

		return points

	def pose_score(self, points):
		if len(points) < self.minimum_joints:
			return None # Return a score of 0 if no pose detected
		
		correct = 0
		# Locate points in mask and mark if point is over mask or not
		for point in points:
			xi = point[0]
			yi = point[1]
			point_val = self.mask[yi, xi]
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
		return score
	
	def updateToEval(self):
		# Show image from pose estimation
		self.pose_label.pack(side=Tk.LEFT)

		# Run pose estimation and return list of identified coordinates
		calc_points = self.pose_estimate()
		
		# Get a score based on correct points over mask
		score = self.pose_score(calc_points)
		
		if score is not None:		
			self.desText.configure(text="Pose Accuracy: " + str(score)+ "%")
		else:
			self.desText.configure(text="Didn't detect a pose from player.")
		
		self.titleVar.set("Pose Evaluation")
		self.buttonPanel.button1.configure(text="Main Menu", command=lambda:MainGUI.updateToTitle(self))
		self.buttonPanel.button2.configure(text="Exit",command=self.root.quit)

	def countDown(self, timer):
		self.desText.configure(text = "Time to Evaluation: " + str(timer) + "s")
		timer = timer - 1
		if(timer >= 0):
			self.root.after(1000, lambda:MainGUI.countDown(self, timer))
		else:
			MainGUI.updateToEval(self)

	#Camera Displays Here
	def camera_loop(self):
		img = self.camera.read()
		self.img = img # Load image into context variable
		img = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=img)
		self.feed_label.imgtk = imgtk
		self.feed_label.configure(image=imgtk)		
		self.root.after(10, self.camera_loop)

	def mask_select(self, *args):
		mask_img=cv2.imread(self.ddVar.get(), cv2.IMREAD_GRAYSCALE)
		self.mask=cv2.threshold(mask_img, self.thresh, 255, cv2.THRESH_BINARY)[1]
		img = Image.fromarray(self.mask)
		imgtk = ImageTk.PhotoImage(image=img)
		self.mask_label.imgtk = imgtk
		self.mask_label.configure(image=imgtk)		
		
	def preprocess(self, img):
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		image = PIL.Image.fromarray(image)
		image = transforms.functional.to_tensor(image).to(self.device)
		image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
		return image[None, ...]	

	def blankCommand():
		print("Error: Button should not be pushed")



