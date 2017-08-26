import tkinter as TK
import numpy as np
import mnist as mnist
import sys, threading
import multiprocessing
from multiprocessing.queues import Queue
from queue import Empty
from tkinter import messagebox
from matplotlib import pyplot as plt


btn1pressed = False
newline = True
trained = False
the_canvas = None
mat_size = 28
canvas_size = 500
pixels = np.zeros((mat_size, mat_size))
net_thread = None
toplevel = None
top_popup_opening = False

class StdoutRedirector(object):
    def __init__(self,text_widget):
        self.text_space = text_widget

    def write(self,string):
        self.text_space.insert('end', string)
        self.text_space.see('end')
        
class StdoutQueue(Queue):
    def __init__(self, *args,**kwargs):
        Queue.__init__(self,ctx = multiprocessing.get_context(), *args,**kwargs)

    def write(self,msg):
        self.put(msg)

    def flush(self):
        sys.__stdout__.flush()

class ThreadSafeText(TK.Text):
    def __init__(self, master, **options):
        TK.Text.__init__(self, master, **options)
        self.queue = StdoutQueue()
        sys.stdout = self.queue
        self.update_me()
    def write(self, line):
        self.queue.put(line)
    def clear(self):
        self.queue.put(None)
    def update_me(self):
        try:
            while 1:
                line = self.queue.get_nowait()
                if line is None:
                    self.delete(1.0, TK.END)
                else:
                    self.insert(TK.END, str(line))
                self.see(TK.END)
                self.update_idletasks()
        except Empty:
            pass
        self.after(100, self.update_me)


def main():
    root = TK.Tk()
    root.title("Simple mnist net")
    root.resizable(0,0)
    global the_canvas
    the_canvas = TK.Canvas(root, width = canvas_size, height = canvas_size, background='white')
    the_canvas.config(highlightbackground='black')
    the_canvas.config(highlightthickness=1)
    train_button = TK.Button(root, text="Train", command=train_clicked, height = 2, width = 10)
    go_button = TK.Button(root, text="Go", command=go_clicked, height = 2, width = 10)
    clear_button = TK.Button(root, text="Clear", command=clear_clicked, height = 2, width = 10)
    show_button = TK.Button(root, text="Show", command=show_clicked, height = 2, width = 10)
    the_canvas.grid(row=0, column=0, columnspan=4, padx=(10, 10), pady=(10, 10))
    train_button.grid(row=1, column=0, padx=(10, 10), pady=(10, 10)) 
    go_button.grid(row=1, column=1, padx=(10, 10), pady=(10, 10)) 
    clear_button.grid(row=1, column=2, padx=(10, 10), pady=(10, 10))
    show_button.grid(row=1, column=3, padx=(10, 10), pady=(10, 10))
    the_canvas.bind("<Motion>", mousemove)
    the_canvas.bind("<ButtonPress-1>", mouse1press)
    the_canvas.bind("<ButtonRelease-1>", mouse1release)
    root.mainloop()
    
def go_clicked():
    if not trained:
        TK.messagebox.showinfo("Info", "Please train first")
    else:
        TK.messagebox.showinfo("Info", str(mnist.recognize(np.ravel(pixels))))

def clear_clicked():
    global pixels, the_canvas
    pixels = np.zeros((mat_size, mat_size))
    the_canvas.delete("all")
    
def show_clicked():
    plt.imshow(pixels, interpolation='nearest')
    plt.show()
    
def train_clicked():
    global trained, net_thread, toplevel, top_popup_opening
    if not top_popup_opening:
        top_popup_opening = True 
        toplevel = TK.Toplevel()
        text_box = ThreadSafeText(toplevel, wrap='word', height=11, width=60)
        text_box.pack()
        if not trained:
            net_thread = threading.Thread(target = mnist.train)
            net_thread.daemon = True
            net_thread.start()
            trained = True
        toplevel.protocol("WM_DELETE_WINDOW", on_closing)
        toplevel.focus_force()

def on_closing():
    global net_thread, top_popup_opening, toplevel
    top_popup_opening = False
    toplevel.destroy()
    
def mouse1press(event):
    global btn1pressed
    btn1pressed = True
def mouse1release(event):
    global btn1pressed, newline
    btn1pressed = False
    newline = True
def mousemove(event):
    if btn1pressed == True:
        global xorig, yorig, newline
        if newline == False:
            event.widget.create_line(xorig,yorig,event.x,event.y,
                                     smooth=TK.TRUE)
            image_matrix(event.x,event.y)
        newline = False
        xorig = event.x
        yorig = event.y
        
def image_matrix(x, y):
    pixel_size = canvas_size / mat_size
    if x < canvas_size and y < canvas_size:
        pixels[int(y/pixel_size)][int(x/pixel_size)] = 0.9
        '''
        try:
            if pixels[int(y/pixel_size)-1][int(x/pixel_size)] == 0: 
                pixels[int(y/pixel_size)-1][int(x/pixel_size)] = 0.5
            if pixels[int(y/pixel_size)][int(x/pixel_size)-1] == 0:
                pixels[int(y/pixel_size)][int(x/pixel_size)-1] = 0.5
            if pixels[int(y/pixel_size)][int(x/pixel_size)+1] == 0:
                pixels[int(y/pixel_size)][int(x/pixel_size)+1] = 0.5
            if pixels[int(y/pixel_size)+1][int(x/pixel_size)] == 0:
                pixels[int(y/pixel_size)+1][int(x/pixel_size)]= 0.5
        except IndexError:
            pass
        '''
        

if __name__ == "__main__":
    main()
#End