import Detection
import Interface
import Drivers
import numpy as np
import argparse
import Utils
import cv2
import time
import importlib.util
import keyboard


def main():
    #parse command line arguments
    parser = argparse.ArgumentParser(description='Initialize autoturret.')
    parser.add_argument('target_class', type=str, help='Pick a target class from the list')
    #parser.add_argument('pin_file', type=str, help='file path to file containing pin info')
    parser.add_argument('--fire', action='store_true', default=False,  help='just track target instead of automatically firing', dest='fire')
    parser.add_argument('--show_stream', action='store_true', default=False, help="if you want to see it work.q")


    results = parser.parse_args()
    USER_TARGET = results.target_class
    PIN_FILEPATH = 'pin_loc.txt' #results.pin_file
    FIRE = results.fire
    SHOW_STREAM = results.show_stream
    ACTIVE = True

    #import from file global constant pin locations
    pin_dict = Utils.parse_pinfile(PIN_FILEPATH)

    #initialize drivers
    print('Initializing Drivers....')
    m1 = Drivers.Motor1(pin_dict['M1A1'], pin_dict['M1A2'], pin_dict['M1B1'], pin_dict['M1B2'])
    m2 = Drivers.Motor2(enable = pin_dict['en2'], direction = pin_dict['dirpin2'], pulse=pin_dict['pul2'])
    launcher = Drivers.Launcher(pin_dict['WaterPin'], pin_dict['AirPin'], 1, .4)

    #initialize interface
    print("Setting up interface...")
    control = Interface.CannonControl(m1, m2, launcher)

    # import the correct version of tensorflow based on whats installed and define and interpreter
    # from on the .tflite file
    if importlib.util.find_spec('tflite_runtime'):
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path='model/detect.tflite')
    else:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path='model/detect.tflite')

    interpreter.allocate_tensors()

    #load labels
    with open('model/labelmap.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    #initialize detection model
    print("Starting detection model...")
    model = Detection.TargetStream(USER_TARGET, interpreter)
    model.start()
    time.sleep(2)
    print("Model ready...")

    #define autotarget functionality

    #toss that in a loop. consiter using threads
    while ACTIVE:
        if SHOW_STREAM:
            model.show_frame()

        if model.object_detected:
            #autotarget
            location = model.target_location
            print(location)
            time.sleep(2)
        else:
            #randomly scan
            pass

        if keyboard.is_pressed('q'):
            print("Stopping model...")
            model.stop()
            ACTIVE = False
            break

if __name__ == "__main__":
    try:
        main()
    except:
        model.stop()
        print('might just be an error')
        raise





