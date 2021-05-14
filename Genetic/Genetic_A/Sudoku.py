import random
import time
import json
import numpy as np
from tkinter import *
from tkinter.ttk import *
import GA_Sudoku_Solver as gss
import Logic
random.seed(time.time())

class SudokuGUI(Frame):

    def __init__(self, master):

        Frame.__init__(self, master)
        if master:
            master.title("Sudoku with Genetic Algorithm")

        self.grid = [[0 for x in range(9)] for y in range(9)]
        self.make_grid()
        self.bframe = Frame(self)
        # generate new game
        self.ng = Button(self.bframe, text='Generate New Game', width=20, command=self.new_game).pack(anchor=S)
        # solver
        self.sg = Button(self.bframe, text='Solver', width=20, command=self.solver).pack(anchor=S)

        self.bframe.pack(side='bottom', fill='x', expand='1')
        #self.new_game()
        self.pack()

    def rgb(self, red, green, blue):
        return "#%02x%02x%02x" % (red, green, blue)


    def new_game(self):

        self.given = Logic.new_game()
        self.grid = np.array(self.given).reshape((9,9)).astype(int)
        self.sync_board_and_canvas()

    def solver(self):
        s = gss.Sudoku()
        s.load(self.grid)
        start_time = time.time()
        generation, solution = s.solve()
        if (solution):
            if generation == -1:
                print("Invalid inputs")
                str_print = "Invalid input, please try to generate new game"
            elif generation == -2:
                print("No solution found")
                str_print = "No solution found, please try again"
            else:
                self.grid_2 = solution.values
                self.sync_board_and_canvas_2()
                time_elapsed = '{0:6.2f}'.format(time.time()-start_time)
                str_print = "Solution found at generation: " + str(generation) + \
                        "\n" + "Time elapsed: " + str(time_elapsed) + "s"
            Label(self.bframe, text=str_print, relief="solid", justify=LEFT).pack()
            self.bframe.pack()

    def make_grid(self):
        (w, h) = (450, 450)
        c = Canvas(self, bg=self.rgb(128, 128, 128), width=2 * w, height=h)
        c.pack(side='top', fill='both', expand='1')

        self.rects = [[None for x in range(18)] for y in range(9)]
        self.handles = [[None for x in range(18)] for y in range(9)]
        rsize = w/9
        guidesize = h/3

        for y in range(9):
            for x in range(18):
                (xr, yr) = (x * guidesize, y * guidesize)
                if x < 3:
                    if x == 0 and y == 0:
                        self.rects[y][x] = c.create_rectangle(xr+4, yr+4, xr+guidesize,
                                                      yr + guidesize, width=4, fill='yellow')
                    elif x == 0:
                        self.rects[y][x] = c.create_rectangle(xr+4, yr, xr+guidesize,
                                                      yr + guidesize, width=4, fill='yellow')
                    elif y == 0:
                        self.rects[y][x] = c.create_rectangle(xr, yr+4, xr+guidesize,
                                                      yr + guidesize, width=4, fill='yellow')
                    else:
                        self.rects[y][x] = c.create_rectangle(xr, yr, xr+guidesize,
                                                      yr + guidesize, width=4, fill='yellow')
                else:
                    if y == 0:
                        self.rects[y][x] = c.create_rectangle(xr, yr+4, xr+guidesize,
                                                      yr + guidesize, width=4, fill='gray')
                    else:
                        self.rects[y][x] = c.create_rectangle(xr, yr, xr+guidesize,
                                                      yr + guidesize, width=4, fill='gray')
                (xr, yr) = (x * rsize, y * rsize)
                r = c.create_rectangle(xr, yr, xr + rsize, yr + rsize)
                t = c.create_text(xr + rsize / 2, yr + rsize / 2)
                self.handles[y][x] = (r, t)

        self.canvas = c
        self.sync_board_and_canvas()

    def sync_board_and_canvas(self):
        g = self.grid
        for y in range(9):
            for x in range(9):
                if g[y][x] != 0:
                    self.canvas.itemconfig(self.handles[y][x][1],
                                           text=str(g[y][x]))
                else:
                    self.canvas.itemconfig(self.handles[y][x][1],
                                           text='')
    def sync_board_and_canvas_2(self):
        g = self.grid_2
        for y in range(9):
            for x in range(9):
                self.canvas.itemconfig(self.handles[y][x+9][1],
                                       text=str(g[y][x]))
tk = Tk()
gui = SudokuGUI(tk)
gui.mainloop()