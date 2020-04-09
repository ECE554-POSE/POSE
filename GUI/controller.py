import tkinter as Tk

from title_screen import MainGUI


class Controller:
    def __init__(self):
        self.root = Tk.Tk()
        self.view = MainGUI(self.root)

    def run(self):
        self.root.title("POSE")
        self.root.mainloop()
