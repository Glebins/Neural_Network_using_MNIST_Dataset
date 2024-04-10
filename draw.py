import numpy as np
import math

from tkinter import *

from array_functions import *


class Draw:
    def __init__(self, root):

        self.infographic = None
        self.result = None
        self.canvas_width = 280
        self.canvas_height = 280

        # Defining title and Size of the Tkinter Window GUI
        self.pixels = np.zeros((self.canvas_width, self.canvas_height))
        self.root = root
        root.geometry("1000x600+550+200")
        self.root.title("Painter")
        self.root.configure(background="white")
        # self.root.resizable(0,0)

        # variables for pointer and Eraser
        self.pointer = "black"

        # Reset Button to clear the entire screen
        self.btn_clear_screen = Button(self.root, text="Clear Screen", bd=4, bg='white',
                                       command=self.clear_canvas, width=9, relief=RIDGE)
        self.btn_clear_screen.place(x=0, y=30)

        self.btn_print_image = Button(self.root, text="Print an array", bd=4, bg='white',
                                      command=self.print_array_of_pixels, width=9, relief=RIDGE)
        self.btn_print_image.place(x=0, y=70)

        #self.btn_guess = Button(self.root, text="Guess the number", bd=4, bg='white',
        #                        command=self.guess_number, width=10, relief=RIDGE)
        #self.btn_guess.place(x=0, y=110)

        # Creating a Scale for pointer and eraser size
        self.pointer_frame = LabelFrame(self.root, text='size', bd=5, bg='white', font=('arial', 15, 'bold'),
                                        relief=RIDGE)
        self.pointer_frame.place(x=0, y=320, height=200, width=70)

        self.pointer_size = Scale(self.pointer_frame, orient=VERTICAL, from_=50, to=0, length=168)
        self.pointer_size.set(10)
        self.pointer_size.grid(row=0, column=1, padx=15)

        # Defining a background color for the Canvas
        # self.background = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=470, width=680)
        self.background = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=self.canvas_height,
                                 width=self.canvas_width)
        self.background.place(x=80, y=40)

        # Bind the background Canvas with mouse click
        self.background.bind("<B1-Motion>", self.paint)
        self.root.bind("<Key>", lambda event: root.destroy() if event.char == "z" else None)

    def clear_canvas(self):
        self.background.delete('all')
        self.pixels = np.zeros((self.canvas_width, self.canvas_height))

    def paint(self, event):
        radius = self.pointer_size.get()
        x1, y1 = (event.x - radius), (event.y - radius)
        x2, y2 = (event.x + radius), (event.y + radius)

        self.background.create_oval(x1, y1, x2, y2, fill=self.pointer, outline=self.pointer)

        for i in range(int(event.x - radius) + 1, int(event.x + radius) + 1):
            dy = math.sqrt(radius ** 2 - (event.x - i) ** 2)

            for j in range(int(event.y - dy), int(event.y + dy + 1)):
                if i < 0 or i > self.canvas_width - 1 or j < 0 or j > self.canvas_height - 1:
                    continue
                self.pixels[i, j] = 1

    def print_array_of_pixels(self):
        pixels = compress_array(self.pixels)

        for i in range(len(pixels)):
            for j in range(len((pixels[0]))):
                print(int(pixels[j][i]), end='\t')
            print()
        print("\n\n")

    def draw_infographic(self, predict):
        self.infographic = Canvas(self.root, width=550, height=320)
        self.infographic.place(x=400, y=40)

        for i in range(1, 11):
            self.infographic.create_text(20, 30 * i, text=str(i - 1), fill="black", font="Helvetica 15 bold")

        max_len = 300
        alpha = max_len / np.max(predict)

        for i in range(0, 10):
            self.infographic.create_line(50, 30 * (i + 1), alpha * predict[i] + 50, 30 * (i + 1), fill="green", width=0)

        self.result = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=40, width=40)
        self.result.place(x=500, y=400)
        self.result.create_text(30, 30, text=str(np.argmax(predict)), fill="black", font='Helvetica 15 bold')
