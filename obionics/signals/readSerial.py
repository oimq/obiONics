import serial
from datetime import datetime
import time

uno_port = 'COM20'
baudrate = 115200

ser = serial.Serial(uno_port, baudrate)

openfile = open('sEMG_sense.txt', 'w')

try :
    while True :
        if ser.readable() :
            t = datetime.now()
            res = ser.readline()
            openfile.seek(0)
            openfile.write("{}{}{}{}:{}\n".format(
                t.hour, t.minute, t.second, t.microsecond, res.decode(encoding='utf-8')[:len(res)-1]))
except KeyboardInterrupt as ki :
    print("Keyboard Interrupt Occurs.")
finally:
    openfile.close()
