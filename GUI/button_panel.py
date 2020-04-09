import tkinter as Tk

# This class contains the buttons whcih allow the user to interact with each screen
# The class is responsible for initializing the buttons which will be used and
# positioning them. The relabelling and reconfiguring should be done in the view or controller

class ButtonPanel():
	def __init__(self, root):
		self.frame2 = Tk.Frame(root)
		self.button1 = Tk.Button(self.frame2, text="Option 1")
		self.button1.pack(side="top", fill=Tk.BOTH)
		self.button2 = Tk.Button(self.frame2, text="Option 2")
		self.button2.pack(side="top", fill=Tk.BOTH)
	
	def pack(self):
		self.frame2.pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=1)
		
