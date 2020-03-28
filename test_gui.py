import tkinter as tk
from functools import partial
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg
from PIL import Image
from PIL import ImageTk

class POSE_GUI:
	def __init__(self):
		self.WIDTH = 224
		self.HEIGHT = 224
		self.root = tk.Tk()
		self.root.title('POSE')
		self.root.geometry(str(self.WIDTH)+"x"+str(self.HEIGHT + 50))
		self.camera = USBCamera(width=self.WIDTH, height=self.HEIGHT, capture_fps=30)
		#self.camera.running = True
		self.start_button = tk.Button(self.root, text= 'Start Game', command=self.game_start)
		self.start_button.pack(side=tk.TOP, padx=5, pady=5)
		self.exit_button = tk.Button(self.root, text= 'Quit', command=self.exit_app)
		self.exit_button.pack(side=tk.TOP, padx=5, pady=5)
		self.lmain = tk.Label(self.root)
		self.lmain.pack()
	
	def main_loop(self):
		img = self.camera.read()
		# data = preprocess(image)
		# cmap, paf = model_trt(data)
		# cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
		# counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
		# draw_objects(image, counts, objects, peaks)
		# imgdraw = bgr8_to_jpeg(image[:, ::-1, :])
		img=Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=img)
		self.lmain.imgtk = imgtk
		self.lmain.configure(image=imgtk)
		self.root.after(10, self.main_loop)

	def game_start(self):
		print("Game time started")
		# Need to either start a Tkinter screen with the camera feed or use a matplotlib for it
		self.main_loop() # Function that repeats itself to continously query the camera for a new image every 10 ms

	def exit_app(self):
		self.camera.running = False
		self.root.destroy()

		

if __name__ == '__main__':
	POSE = POSE_GUI()
	POSE.root.mainloop() # Tk main loop to keep window up
