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
		self.camera = USBCamera(width=self.WIDTH, height=self.HEIGHT, capture_fps=30)
		self.camera.running = True
		self.start_button = tk.Button(self.root, text= 'Start Game', command=self.game_start)
		self.start_button.pack(side=tk.TOP, padx=5, pady=5)
		self.exit_button = tk.Button(self.root, text= 'Quit', command=self.exit_app)
		self.exit_button.pack(side=tk.TOP, padx=5, pady=5)
		self.lmain = tk.Label(self.root)
		self.lmain.pack()
		
	def execute(self, change):
		img = change['new']
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


	def game_start(self):
		print("Game time started")
		# Need to either start a Tkinter screen with the camera feed or use a matplotlib for it
		self.camera.observe(self.execute, names='value')

	def exit_app(self):
		self.camera.unobserve_all()
		self.camera.running = False
		self.root.quit()



if __name__ == '__main__':
	POSE = POSE_GUI()
	POSE.root.mainloop()
