import tkinter as Tk
from button_panel import ButtonPanel


class MainGUI:

	def __init__(self, root):

		self.frame=Tk.Frame(root)
		self.root=root
		# Create editable title 
		self.titleVar = Tk.StringVar()
		self.title= Tk.Label(root, textvariable=self.titleVar, font="Verdana 36")
		self.titleVar.set("Pose Estimation Game")
		self.title.pack(side=Tk.TOP)
		self.frame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)	
		
		# Create image capture figure
		self.intro_text1 = "This is where the capture from the camera will be going"
		self.label1 = Tk.Label(self.frame, text=self.intro_text1, font="Verdana 12")
		
		# Create editable description label
		self.desTextVar = "Please select an option from the right"
		self.desText = Tk.Label(self.frame, text=self.desTextVar, font="Verdana 12")
		


		#Create Combobox for selection (Steps are currently in comments)
		#Grab maps from repository
		#Parse map names to develop choices
		#group map names into array
		choices = {'These', 'are', 'temporary', 'choices'}
		#Put map names in combo box	
		ddVar = Tk.StringVar()
		ddVar.set('Select a Choice')
		self.dropDown = Tk.OptionMenu(self.frame, ddVar, *choices)


		
		# Create inital button panel
		self.buttonPanel = ButtonPanel(root)


		self.label1.pack()
		self.desText.pack()
		self.buttonPanel.pack()
		MainGUI.updateToTitle(self)

	def updateToTitle(self):
		# unpack unused widgets
		self.dropDown.pack_forget()

		# build title frame
		self.titleVar.set("Pose Estimation Game")
		self.desText.configure(text ="Please select an option from the right")
		

		# set button commands
		self.buttonPanel.button1.configure(text="Pose Now!", command=lambda:MainGUI.updateToSelect(self))
		self.buttonPanel.button2.configure(text="Exit",command=self.root.quit)



	def updateToSelect(self):
		# unpack unused widgets


		# Build select frame
		self.titleVar.set("Select Your Pose")
		self.desText.configure(text ="Select an Option from Below")
		self.dropDown.pack()

		# set button commands
		self.buttonPanel.button1.configure(text="Select", command=lambda:MainGUI.updateToPose(self))
		self.buttonPanel.button2.configure(text="Main Menu", command=lambda:MainGUI.updateToTitle(self))



	def updateToPose(self):
		# unpack unused widgets
		self.dropDown.pack_forget()


		self.titleVar.set("Pose Now")
		self.buttonPanel.button1.configure(text="Main Menu", command=MainGUI.blankCommand)
		self.buttonPanel.button2.configure(text=" ", command=MainGUI.blankCommand)
		timer = 10
		self.desText.configure(text = "Time to Evaluation: " + str(timer) + "s")
		self.root.after(1000, lambda:MainGUI.countDown(self, timer))
		

	def updateToEval(self):
		self.titleVar.set("Pose Evaluation")
		self.desText.configure(text="Pose Accuracy: XX%")
		self.buttonPanel.button1.configure(text="Main Menu", command=lambda:MainGUI.updateToTitle(self))
		self.buttonPanel.button2.configure(text="Exit",command=self.root.quit)

	def countDown(self, timer):
		self.desText.configure(text = "Time to Evaluation: " + str(timer) + "s")
		timer = timer - 1
		if(timer >= 0):
			self.root.after(1000, lambda:MainGUI.countDown(self, timer))
		else:
			MainGUI.updateToEval(self)

	def blankCommand():
		print("Error: Button should not be pushed")



