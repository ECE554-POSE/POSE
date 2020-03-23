import tkinter as tk

def game_start():
    print("Game time started")
    # Need to either start a Tkinter screen with the camera feed or use a matplotlib for it
    


if __name__ == '__main__':
    root = tk.Tk()
    root.title('POSE')
    start_button = tk.Button(root, text= 'Start Game', command=game_start)
    start_button.pack(side=tk.LEFT, padx=5, pady=5)
    exit_button = tk.Button(root, text= 'Quit', command=root.quit)
    exit_button.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()