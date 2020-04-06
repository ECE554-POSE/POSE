import tkinter as Tk
from button_panel import ButtonPanel


class MainGUI:

	def __init__(self, root):

		self.frame=Tk.Frame(root)
		self.root=root
		# Create editable title 
		self.titleVar = Tk.StringVar()
		self.title= Tk.Label(root, textvariable=self.titleVar, font="Verdana 36")
		self.title.pack(side=Tk.TOP)
		self.titleVar.set("Pose Estimation Game")

		# Create description labels
		self.frame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)		
		self.intro_text1 = "This is where the capture from the camera will be going"
		self.intro_text2 = "Please select an option from the right"
		self.label1 = Tk.Label(self.frame, text=self.intro_text1, font="Verdana 12")
		self.label1.pack()
		self.label2 = Tk.Label(self.frame, text=self.intro_text2, font="Verdana 12")
		self.label2.pack()

		# Create inital button panel
		self.buttonPanel = ButtonPanel(root)

		MainGUI.updateToTitle(self)

	def updateToTitle(self):
		self.titleVar.set("Pose Estimation Game")
		self.buttonPanel.button1.configure(text="Pose Now!", command=lambda:MainGUI.updateToPose(self))
		self.buttonPanel.button2.configure(text="Exit",command=self.root.quit)

	def updateToSelect(self):
		self.titleVar.set("Select Your Pose")

	def updateToPose(self):
		self.titleVar.set("Pose Now")
		self.buttonPanel.button1.configure(text="Evaluate", command=lambda:MainGUI.updateToEval(self))
		self.buttonPanel.button2.configure(text="Main Menu", command=lambda:MainGUI.updateToTitle(self))

	def updateToEval(self):
		self.titleVar.set("Pose Evaluation")
		self.buttonPanel.button1.configure(text="Main Menu", command=lambda:MainGUI.updateToTitle(self))
		self.buttonPanel.button2.configure(text="Exit",command=self.root.quit)
