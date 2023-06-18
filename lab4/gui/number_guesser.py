import numpy as np
import pickle
import PIL.Image
import PIL.ImageDraw
import PIL.ImageTk
import random

from tkinter import *

from cnn import *
from reader import *
from result_frame import *

font_family = 'Comic Sans'

def choose_result_frame():
    global result_displayer
    if result_displayer is not None:
        # result_displayer.pack_forget()
        result_displayer.destroy()
    match result_displ_var.get():
        case 0:
            result_displayer = NumeralFrame(window)
        case 1:
            result_displayer = CertantyInNumbersFrame(window)
        case 2:
            result_displayer = CertantyInPorcentageFrame(window)
        case _:
            pass
    result_displayer.grid(row=1, column=0, columnspan=3)
    result_displayer.show_result(result)

def start_drawing(event):
    global start_x
    global start_y

    start_x = event.x
    start_y = event.y

def draw_line(event):
    if (start_x is not None) & (start_y is not None):
        canvas.create_line(start_x, start_y, event.x, event.y, width=line_width, smoth='true')
        image_draw.line([start_x, start_y, event.x, event.y], 0, width=line_width)
    start_drawing(event)

def submit():
    if result_displayer is not None:
        bitmap = (1 - np.asarray(image.resize((input_side,input_side))) / 255)\
            .reshape((1, input_side, input_side))
        global result
        result = cnn.forward_prop(bitmap)
        result_displayer.show_result(result)

def save():
    image.save(filename_entry.get())

def clear():
    canvas.delete('all')
    clear_image()

    global result
    result = None

    if result_displayer is not None:
        result_displayer.clear()

def clear_image():
    image_draw.rectangle((0,0,side,side),fill=255)

def load_cnn_data(filename):
    with open(filename, 'rb') as nn_file:
        cnn = pickle.load(nn_file)

    return cnn

def load_image(x):
    clear()

    global image
    global image_draw
    global photo_image
    image_bitmap = x.reshape((input_side,input_side)).astype(np.uint8)
    image = PIL.Image.fromarray(image_bitmap)
    image = image.resize((side, side))
    photo_image = PIL.ImageTk.PhotoImage(image)
    image_draw = PIL.ImageDraw.Draw(image)

    canvas.create_image((side // 2,side // 2),image=photo_image)

    if show_res_on_load_var.get():
        submit()

def load_train():
    load_image(reader.get_random_train())

def load_test():
    load_image(reader.get_random_test())

cnn = load_cnn_data('pickle/nn.pickle')

window = Tk()
window.geometry("720x640")
window.title("Number guesser!")

result_displayer = None
result = None

result_displ_var = IntVar()
result_displayers = {0: 'Just a result',
                     1: 'Certanty in numbers in [0,1]', 2: 'Certanty in procentage'}

for i, name in result_displayers.items():
    result_displayer_rb = Radiobutton(window,
                                      text=name,
                                      variable=result_displ_var,
                                      value=i,
                                      command=choose_result_frame)\
        .grid(row=0, column=i)

Label(window,
      text = 'Draw a numeral! I\'am sure, i can guess it!',
      font=(font_family, 14, 'bold'))\
        .grid(row=0, column=len(result_displayers))

start_x = None
start_y = None

input_side = 28
side = 280
line_width = 15

image = PIL.Image.new("L", (side, side), 255)
photo_image = None
image_draw = PIL.ImageDraw.Draw(image)

canvas = Canvas(window, bg='white', width=side, height=side)
canvas.grid(row=1, column=len(result_displayers), columnspan=2)
canvas.bind('<Button-1>', start_drawing)
canvas.bind('<B1-Motion>', draw_line)

Button(window, text='submit', font=(font_family,14), command=submit)\
    .grid(row=2, column=len(result_displayers))
Button(window, text='clear', font=(font_family,14), command=clear)\
    .grid(row=2, column=len(result_displayers) + 1)

Button(window, text='load train', font=(font_family,14), command=load_train)\
    .grid(row=3, column=len(result_displayers))
Button(window, text='load test', font=(font_family,14), command=load_test)\
    .grid(row=3, column=len(result_displayers) + 1)

filename_entry = Entry(window, font=(font_family,14))
filename_entry.grid(row=4, column=len(result_displayers))
Button(window, text='save', font=(font_family,14), command=save)\
    .grid(row=4, column=len(result_displayers) + 1)

show_res_on_load_var = BooleanVar()
show_res_on_load_check_box = Checkbutton(window,
                                         text='Automatically show result after load',
                                         variable=show_res_on_load_var,
                                         onvalue=True,
                                         offvalue=False,
                                         font=(font_family,12))
show_res_on_load_check_box.grid(row=5, column=len(result_displayers), columnspan=2)
show_res_on_load_check_box.select()

choose_result_frame()

window.mainloop()
