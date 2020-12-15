class Thew :
    def __init__(self, nact, mask =0b0):
        self.mask, self.nact = mask, nact

    def count(self, bits):
        cnt = 0
        while bits > 0 : cnt, bits = cnt+(bits&0b1), bits>>1
        return cnt

    def isact(self, bits):
        return self.count(self.mask & bits) >= self.nact

