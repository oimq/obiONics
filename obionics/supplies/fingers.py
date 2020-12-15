import numpy as np
from .powers import Thew

class Fingers() :
    def __init__(self, conf, transformer) :
        self.thetas = np.zeros(shape=(conf['finger']['number'], conf['finger']['joint']))
        self.dh, self.conf, self.thews, self.powers = transformer.dh, conf, list(), list()
    
    def theting(self) :
        self.dh[:,1:,0] = self.thetas[:,1:]

    def initedThews(self, nact, masks) :
        return [
            [
                Thew(nact, masks[i*(self.conf['finger']['joint']-1)+j]) 
                for j in range(self.conf['finger']['joint']-1)
            ] 
            for i in range(self.conf['finger']['number'])
        ]

    def initedPowers(self, nact, masks) :
        return [Thew(nact, masks[i]) for i in range(len(masks))]

    def move(self, pwr_idx_bits, len_idx_bits) :
        if self.thews and self.powers :
            pwr_bits = pwr_idx_bits >> len_idx_bits
            idx_bits = pwr_idx_bits - (pwr_bits << len_idx_bits)

            pwrlv = -1
            # first, get the power level
            for i, power in enumerate(self.powers) :
                if power.isact(pwr_bits) :
                    pwrlv = i; break;

            # second, apply power to the thews
            if pwrlv > 0 :
                for i in range(len(self.thews)) :
                    for j in range(len(self.thews[i]))  :
                        if self.thews[i][j].isact(idx_bits) :
                            self.thetas[i][j+1] = pwrlv * -10