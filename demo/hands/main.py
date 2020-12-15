from jSona import load
from obionics.coordinates   import Transformer, World
from obionics.tk            import Board, Dynamics, Mouse
from obionics.supplies      import Fingers
from obionics.generators    import BERGen
import numpy as np

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

canvas_timer = None

def gen8transfer(gen, fingers) :
    signal = next(gen)
    fingers.move(signal, NUM_SIGBITS)
    canvas_timer.after(200, lambda : gen8transfer(gen, fingers))

if __name__=="__main__" :
    # get configurations
    conf = load('./hands_default_configs.json')

    # for get coordinates
    tf = Transformer(
        num_finger=conf['finger']['number'],
        num_joint=conf['finger']['joint'],
        link_length=conf['screen']['line']['length'],
        world_pris=conf['screen']['world']['pris']
    )

    world = World()
    world.set_wdh(-20, 100, -30, -180) # visually good!

    board = Board(conf)
    board.pack()
    canvas_timer = board.canvas
    mouse = Mouse(board, world)

    fingers = Fingers(conf, tf)
    fingers.thews  = fingers.initedThews(NUM_ACVIVATED, IDX_MASKS)
    fingers.powers = fingers.initedPowers(NUM_ACVIVATED, PWR_MASKS)

    dynamics = Dynamics(board, tf, world, fingers)
    dynamics.draw()
    dynamics.implement()
    dynamics.theting(fingers)

    # Signal generating and Apply to finger
    fg = BERGen(0)
    gen = fg.gen(PWR_MASKS, IDX_MASKS, NUM_SIGBITS)
    gen8transfer(gen, fingers)    

    board.mainloop()