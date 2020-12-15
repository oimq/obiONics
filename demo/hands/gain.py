from obionics.generators import FanoGen

NUM_SIGBITS   = 16
NUM_ACVIVATED = 3
IDX_MASKS = [
    0b1111000000000000,
    0b0001111000000000,
    0b0000001111000000,
    0b1000000001110000,
    0b1000001000001001,
    0b0001000001000110,
    0b0000100000011010,
    0b0000010000100101,
    0b0100000010001100,
    0b0010000100000011,
]
PWR_MASKS = [
    0b1111000000000000,
    0b0001111000000000,
    0b0000001111000000,
    0b1000000001110000,
    0b1000001000001001,
    0b0001000001000110,
    0b0000100000011010,
    0b0000010000100101,
    0b0100000010001100,
    0b0010000100000011,
]

fg = FanoGen(10)
fg_gen = fg.gen(PWR_MASKS, IDX_MASKS, NUM_SIGBITS)

for i in range(10) :
    print(next(fg_gen))