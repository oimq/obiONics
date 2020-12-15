import numpy as np

class World() :
    def __init__(self) :
        # wvec : absolute frame
        # bvec, jvec : for visualize the rotate world frame
        # wdh : for graphic effect,  we need another world dh
        self.wvec, self.bvec, self.jvec, self.wdh = (np.zeros(4) for i in range(4))
        self.wvec[3], self.bvec[3] = 1, 1

    def set_wdh(self, th, al, di, ai) :
        self.wdh = np.array([th, al, di, ai])