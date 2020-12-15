from tkinter import Tk as Screen, Canvas, Button
import numpy as np

class FlipButton :
    def __init__(self, screen) :
        self.button = Button(screen, command = self.event)
        self.flag   = False
        self.event()
        self.button.pack()
    
    def event(self) :
        self.flag = not self.flag
        if self.flag :
            self.button['text'] = "비트 플립 모드 활성화"
            self.button['bg']   = 'yellow'
        else :
            self.button['text'] = "비트 플립 모드 비활성화"
            self.button['bg']   = 'red'

class Board(FlipButton) :
    def __init__(self, conf) :
        # setup the screen
        screen, canvas = Screen(), Canvas(width=conf['screen']['canvas']['width'], height=conf['screen']['canvas']['height'])

        
        # create points
        points = np.zeros(shape=(conf['finger']['number'], conf['finger']['joint'], 4), dtype=np.float)
        points[:, :, 3] = 1.0

        # drawing shapes
        shapes = {'ovals':[], 'lines':[]}
        for i in range(0, points.shape[0], 1) :
            for j in range(1, points.shape[1], 1) :
                shapes['lines'].append(canvas.create_line(0,0,1,1))
            for j in range(0, points.shape[1], 1) :
                shapes['ovals'].append(canvas.create_oval(0,0,1,1,fill='#AAFFFF'))
            

        # set button
        super().__init__(screen)

        self.conf, self.screen, self.canvas, self.points, self.shapes = conf, screen, canvas, points, shapes

    def mainloop(self) :
        self.screen.mainloop()

    def pack(self) :
        self.canvas.pack()
        self.screen.update()

    def get_components(self) :
        return self.conf, self.screen, self.canvas, self.points, self.shapes

class Mouse :
    def __init__(self, board, world) :
        self.x, self.y, self.mdh, self.wdh = 0, 0, np.zeros(4), world.wdh
        self.cw, self.ch = board.conf['screen']['canvas']['width'], board.conf['screen']['canvas']['height']
        self.binding(board.canvas)

    def binding(self, canvas) :
        canvas.bind("<ButtonPress-1>", self.press1)
        canvas.bind("<B1-Motion>", self.motion1)
        canvas.bind("<ButtonPress-3>", self.press3)
        canvas.bind("<B3-Motion>", self.motion3)

    def press1(self, event) :
        self.x, self.y = event.x, event.y
        self.mdh[0:2] = self.wdh[0:2]

    def motion1(self, event) :
        self.wdh[0] = self.mdh[0] + ((event.x - self.x) / self.cw  * 45)
        self.wdh[1] = self.mdh[1] + ((self.y - event.y) / self.ch * 45)

    def press3(self, event) :
        self.x, self.y = event.x, event.y
        self.mdh[2:4] = self.wdh[2:4]

    def motion3(self, event) :
        self.wdh[2] = self.mdh[2] + ((event.x - self.x) / self.cw  * 200)
        self.wdh[3] = self.mdh[3] + ((event.y - self.y) / self.ch * 200)

class Dynamics :
    def __init__(self, board, transformer, world, fingers) :
        self.conf, self.screen, self.canvas, self.points, self.shapes = board.get_components()
        self.tf, self.world, self.fingers = transformer, world, fingers
        
    def draw(self) :
        for i in range(0, self.points.shape[0], 1) :
            for j in range(1, self.points.shape[1], 1) :
                self.canvas.coords(
                    self.shapes['lines'][i*(self.points.shape[1]-1)+j-1],
                    self.points[i,j-1,2],
                    self.points[i,j-1,0],
                    self.points[i,  j,2],
                    self.points[i,  j,0]
                )
                lw = (self.points[i,j-1,1]+200)/10
                self.canvas.itemconfig(self.shapes['lines'][i*(self.points.shape[1]-1)+j-1], width=max([lw, 0.1]))
            for j in range(0, self.points.shape[1], 1) :
                rad = max([self.conf['screen']['point']['size']+(self.points[i][j][1]/20), 0.1])
                self.canvas.coords(
                    self.shapes['ovals'][i*self.points.shape[1]+j],
                    self.points[i,  j,2]-rad,
                    self.points[i,  j,0]-rad,
                    self.points[i,  j,2]+rad,
                    self.points[i,  j,0]+rad
                )

        self.canvas.after(self.conf['anim']['delay']['draw'], self.draw)

    def implement(self) :
        for i in range(self.points.shape[0]) :
            for j in range(self.points.shape[1]) :
                self.points[i][j] = self.tf.get_coordinates([self.tf.dh[i][k] for k in range(j, -1, -1)]+[self.world.wdh], self.world.wvec)
        # for i in range(self.points.shape[0]) :
        #     self.points[i][0] = self.tf.get_coordinates([self.tf.dh[i][0], self.world.wdh], self.world.wvec)
        # for i in range(self.points.shape[0]) :
        #     self.points[i][1] = self.tf.get_coordinates([self.tf.dh[i][1], self.tf.dh[i][0], self.world.wdh], self.world.wvec)
        # for i in range(self.points.shape[0]) :
        #     self.points[i][2] = self.tf.get_coordinates([self.tf.dh[i][2], self.tf.dh[i][1], self.tf.dh[i][0], self.world.wdh], self.world.wvec)                
        self.canvas.after(self.conf['anim']['delay']['implement'], self.implement)

    def theting(self, figure) :
        figure.theting()
        self.canvas.after(self.conf['anim']['delay']['theting'], lambda :self.theting(figure))

