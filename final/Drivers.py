
import importlib.util
import time
if not importlib.util.find_spec('RPi'):
    import sys
    import fake_rpi
    sys.modules['RPi'] = fake_rpi.RPi     # Fake RPi
    sys.modules['RPi.GPIO'] = fake_rpi.RPi.GPIO # Fake GPIO
    sys.modules['smbus'] = fake_rpi.smbus # Fake smbus (I2C)

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)


pattern = [(0, 1, 0, 1), (0, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0), (1, 0, 1, 0), (1, 0, 0, 0), (1, 0, 0, 1),(0, 0, 0, 1)]  # halfstep pattern
patternfull = [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 1, 0), (1, 0, 0, 1)]
delay = .05
 
 #NOTE: everything here is hardware specific 

class Motor1:

    def __init__(self, pin1, pin2, pin3, pin4):
        self.A1 = pin1
        self.A2 = pin2
        self.B1 = pin3
        self.B2 = pin4
        GPIO.setup(self.A1, GPIO.OUT)
        GPIO.setup(self.A2, GPIO.OUT)
        GPIO.setup(self.B1, GPIO.OUT)
        GPIO.setup(self.B2, GPIO.OUT)
        self.state = 0
        self.move(pattern[self.state])

    def inc(self):
        self.state = (self.state + 1) % len(pattern)
        self.move(pattern[self.state])

    def dec(self):
        self.state = (self.state - 1) % len(pattern)
        self.move(pattern[self.state])

    def off(self):
        self.move((0, 0, 0, 0))

    def move(self, inpt):
        GPIO.output(self.A1, inpt[0])
        GPIO.output(self.A2, inpt[1])
        GPIO.output(self.B1, inpt[2])
        GPIO.output(self.B2, inpt[3])

class Motor2:

    def __init__(self, enable, direction, pulse):
        self.enable = enable
        self.direction = direction
        self.pulse = pulse
        GPIO.setup(self.enable, GPIO.OUT)
        GPIO.setup(self.direction, GPIO.OUT)
        GPIO.setup(self.pulse, GPIO.OUT)
        GPIO.output(self.enable, 0)
        GPIO.output(self.direction, 0)
        GPIO.output(self.pulse, 1)

    def inc(self):
        print("INC")
        GPIO.output(self.enable, 0)
        GPIO.output(self.direction, 1)
        GPIO.output(self.pulse, 0)
        time.sleep(.05)
        GPIO.output(self.pulse, 1)

    def dec(self):
        GPIO.output(self.enable, 0)
        GPIO.output(self.direction, 0)
        GPIO.output(self.pulse, 0)
        time.sleep(.05)
        GPIO.output(self.pulse, 1)

    def off(self):
        GPIO.output(self.enable, 1)


class Launcher:

    def __init__(self, waterpin, airpin, water_del=1, air_del=.2):
        self.water = waterpin
        self.air = airpin
        self.water_delay = water_del
        self.air_delay = air_del
        GPIO.setup(self.water, GPIO.OUT)
        GPIO.setup(self.air, GPIO.OUT)
        GPIO.output(self.air, 1)
        GPIO.output(self.water, 1)

    def fire(self):
        GPIO.output(self.air, 1)
        GPIO.output(self.water, 1)
        time.sleep(delay)                # allow to settle
        GPIO.output(self.water, 0)  # open water
        time.sleep(self.water_delay)
        GPIO.output(self.water, 1)
        time.sleep(delay)                # allow to settle
        GPIO.output(self.air, 0)    # open air
        time.sleep(self.air_delay)
        GPIO.output(self.air, 1)