from random import randint

class BERGen() :
    def __init__(self, filp_number =0) :
        self.fn = filp_number
    
    def setFn(self, num) :
        self.fn = num

    def flip(self, bits, num_sigbits) :
        if self.fn > 0 :
            str_sig = list(("{0:0"+str(num_sigbits*2)+"b}").format(bits))
            flip_log = set()
            while len(flip_log) < self.fn :
                flip_idx = randint(0, len(str_sig)-1)
                if flip_idx in flip_log : continue
                else : flip_log.add(flip_idx)
                str_sig[flip_idx] = '1' if str_sig[flip_idx]=='0' else '1'
            return int("".join(str_sig), 2) 
        else :
            return bits

    def gen(self, pwr_signals, idx_signals, num_sigbits) :
        while True :
            for i in range(0, len(idx_signals), 1) :
                for p in range(0, len(idx_signals), 1) :
                    yield self.flip((pwr_signals[p]<<num_sigbits)+idx_signals[i], num_sigbits)
            for i in range(0, len(idx_signals), 1) :
                for p in range(0, len(idx_signals), 1) :
                    yield self.flip((pwr_signals[p]<<num_sigbits)+idx_signals[i], num_sigbits)