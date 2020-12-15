import numpy as np
from math import cos, sin, radians
from functools import reduce

class Rotate() :
    def __init__(self) :
        pass

    def getAll(self, th, di, al, ai) :
        return [
            self.getRotTheta(al), 
            self.getPrisDist(ai), 
            self.getRotAlpha(di), 
            self.getPrisAist(th)
        ]

    def getRotTheta(self, theta) :
        rt = radians(theta)
        return np.array([
            [cos(rt), -sin(rt),       0,       0],
            [sin(rt),  cos(rt),       0,       0],
            [      0,        0,       1,       0],
            [      0,        0,       0,       1]
        ])

    def getRotAlpha(self, alpha) :
        ra = radians(alpha)
        return np.array([
            [      1,        0,       0,       0],
            [      0,  cos(ra),-sin(ra),       0],
            [      0,  sin(ra), cos(ra),       0],
            [      0,        0,       0,       1]
        ])

    def getPrisDist(self, dist) :
        return np.array([
            [      1,        0,       0,       0],
            [      0,        1,       0,       0],
            [      0,        0,       1,    dist],
            [      0,        0,       0,       1]
        ])

    def getPrisAist(self, aist) :
        return np.array([
            [      1,        0,       0,    aist],
            [      0,        1,       0,       0],
            [      0,        0,       1,       0],
            [      0,        0,       0,       1]
        ])
    
    def getRevoluteJointMatrix(self, theta, dista, alpha, aista) :
        radth = radians(theta)
        radal = radians(alpha)
        return np.array([
            [   cos(radth), -1*cos(radal)*sin(radth),    sin(radal)*sin(radth),    aista*cos(radth)],
            [   sin(radth),    cos(radal)*cos(radth), -1*sin(radal)*cos(radth),    aista*sin(radth)],
            [          0.0,               sin(radal),               cos(radal),           1.0*dista],
            [          0.0,                      0.0,                      0.0,                 1.0]
        ])

    def get_coordinates(self, dmat, vec) :
        mat = reduce(lambda x,y: np.matmul(self.getRevoluteJointMatrix(*y),x),[np.identity(4)]+dmat)
        return np.matmul(mat, vec)
         
class Transformer(Rotate) :
    def __init__(self, num_finger=5, num_joint=3, link_length=150, world_pris=500) :
        self.dh = self.create_dh(num_finger, num_joint, link_length, world_pris)
    
    def create_dh(self, nf, nj, ll, wp) :
        dh = np.zeros(shape=(nf, nj, 4), dtype=np.float)
        mcp = wp/5
        for i in range(nf) :
            dh[i,0,1] = i*mcp+mcp
            dh[i,0,3] = wp
            for j in range(1, nj) : dh[i,j,3] -= ll
        # adjust thumb finger
        dh[-1,0,2], dh[-1,0,0], dh[ 4,0,3] = -90.0, 20.0, dh[ 4,0,3] + 200.0
        return dh
    
    def get_transposition_matrix(self, th, di, al, ai) :
        return reduce(lambda x,y:np.matmul(y,x), self.getAll(th, di, al, ai))