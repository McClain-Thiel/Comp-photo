import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
import pandas as pd
from imutils import face_utils
import dlib
import class_details



class TargetStream():

    def __init__(self, target_class, interpreter, classifier=None, output_shape=(300,300)):
        # setting up video capture/ recognition functionality
        self.stream = cv2.VideoCapture(0)
        self.exp_output_shape = output_shape
        self.grabbed, self.frame = self.stream.read()
        self.model = interpreter
        if not self.grabbed:
            raise ValueError('Problem opening webcam')
        # initiaize attributes specific to the frame
        self.norm_frame = self.get_normalized_frame(self.frame)
        self.target_location = None
        self.target_corners = None
        self.MIN_SCORE_THRESH = .25
        self.target_class = target_class
        self.object_detected = False
        self.active = True
        # set up tflite recongnition pipeline
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        # support to run the upddate funtion on a seperate thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        # initialize support for facial recognition and distancing
        self.focal_length = 829.6942249644 * .48 #see steps in utils function to calculate + const adjust
        self.face_detected = False
        self.face = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('model/face_landmarks.dat')
        self.target_distance_away = 0
        self.scaling_factor = 0
        self.classifier = classifier
        self.pred_class = None

    def update(self, verbose=False):
        """
        This is the workhorse function that runs on an independant thread and keeps all the class attributes up
        to date in close to real time.
        :param verbose: bool - if true, will print a dataframe of all 10 objects detected with location and confidence
        information
        :return: none
        """
        while self.active:
            self.grabbed, frame = self.stream.read()
            self.frame = frame
            self.object_detected = False
            stream_image = self.get_normalized_frame(frame).astype(np.uint8)
            self.model.set_tensor(self.input_details[0]['index'], stream_image)
            self.model.invoke()

            loc_data = self.model.get_tensor(self.output_details[0]['index'])[0]
            location_tuples = [(loc_data[i][0], loc_data[i][1], loc_data[i][2], loc_data[i][3]) for i in range(loc_data.shape[0])]

            d = {
                "locations":location_tuples,
                "class_num": self.model.get_tensor(self.output_details[1]['index'])[0],
                "confidence":self.model.get_tensor(self.output_details[2]['index'])[0],
                "class_label": [labels[int(i)] for i in self.model.get_tensor(self.output_details[1]['index'])[0]]
            }

            df = pd.DataFrame(data = d)
            if verbose:
                print(df)

            targets = df[df.class_label == self.target_class].sort_values('confidence', ascending = False)

            if len(targets):
                best_loc, best_class_num, best_conf, best_class_label  = list(targets.iloc[0,:])
                if best_conf > self.MIN_SCORE_THRESH:
                    self.object_detected = True
                    self.target_location = np.mean([best_loc[0], best_loc[2]]), np.mean([best_loc[1], best_loc[3]])
                    lower, upper= self.draw_box(best_loc)
                    self.target_corners = (lower, upper)

            face = self.find_face()

            if self.face_detected:
                data = face - face.mean(axis=0)
                data = data.reshape((1, 68, 2)).astype(np.float32)
                data = tf.convert_to_tensor(data, np.float32)

                # use model
                input_details = self.classifier.get_input_details()
                output_details = self.classifier.get_output_details()

                self.classifier.set_tensor(input_details[0]['index'], data)
                self.classifier.invoke()

                self.pred_class = class_details.class_index[np.argmax(self.classifier.get_tensor(output_details[0]['index']))]

            if self.face_detected:
                self.find_dist()


    def show_frame(self, draw_box=True, draw_midpoint=False, draw_face = False):
        """
        function that allows user to visualize what the computer is doing. It will draw a
        bounding box and show its estimate of where the target is distancewise if its a
        person
        :param draw_box: bool- if true it will draw the bounding box on the output footage
        :param draw_midpoint: bool - if true, draws the midpoint of the bounding box
        :param draw_face: bool - if true draws facial landmarks on the image if a person is detected
        :return: nothing, but pipes visual output to a video
        """
        if self.face_detected:
            string = "Target Distance from camera: " + str(self.target_distance_away)
            cv2.putText(self.frame, string, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
        else:
            string = "Cannot find: " + self.target_class 
            cv2.putText(self.frame, string ,(30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2,cv2.LINE_AA)

        prep_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        if draw_box and self.object_detected:
            cv2.rectangle(prep_frame, self.target_corners[0], self.target_corners[1], (10, 255, 0), 2)
        if draw_midpoint and self.object_detected:
            cv2.circle(prep_frame, self.draw_midpoint(), 15, (10, 255, 0))
        if draw_face and self.face_detected:
            self.find_dist()
            arr = self.plot_eyes()
            for (x, y) in arr:
                cv2.circle(prep_frame, (x, y), 1, (0, 0, 255), -1)

        cv2.imshow('Object detector', prep_frame)

        if cv2.waitKey(1) == ord('q'):
            self.stop()

    def get_normalized_frame(self, frame1):
        """
        helper to normailze frame into the way the pretrained recognition model expects it
        :param frame1: the unnormailzed frame
        :return: the normalized 1x300x300x3 frame
        """
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.exp_output_shape)
        return frame_resized.reshape((1,self.exp_output_shape[1], self.exp_output_shape[0], 3))


    def draw_box(self, location):
        """
        helper to draw box on the output footage
        :param location: location as reported by the recognition model
        :return: min and max corners that the CV2 function uses to draw a box
        """
        h,w,_ = self.frame.shape
        ymin = int(max(1,(location[0] * h)))
        xmin = int(max(1,(location[1] * w)))
        ymax = int(min(h,(location[2] * h)))
        xmax = int(min(w,(location[3] * w)))
        return (xmin, ymin), (xmax, ymax)

    def draw_midpoint(self):
        """
        draws the midpoint on the output footage
        :return: the midpoint
        """
        new_h, new_w, _ = self.frame.shape
        y,x = self.target_location
        return int(x*new_w), int(y*new_h)

    def find_face(self):
        """
        helper function that finds the location of the face and updates the
        location of the facial landmarks in the instance attributes
        :return: none, updates attributes
        """
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        self.face_detected = False
        if len(rects) > 0:
            self.face_detected = True
            shape = self.predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            self.face = shape
            return shape

    def find_eyes(self):
        """
        returns a tuple of the locations of the eyes
        :return: tuple of 2 arrays with r,c locations of the 5 eye coorda
        """
        face = self.face
        left_eye = face[36:42, :]
        right_eye = face[42:48, :]
        return left_eye, right_eye

    def plot_eyes(self):
        """
        plots the eyes
        :return:
        """
        eyes = self.find_eyes()
        return np.append(eyes[0], eyes[1], axis=0)

    def find_dist(self):
        """
        uses some calibrated constants and projective geometry to estimate how far away a person in frame is
        :return: none, updates class attribute
        """
        if not self.face_detected:
            return
        l, r  = self.find_eyes()

        right_eye_mid = r.mean(axis=0).reshape((1, 2)).astype(int)
        left_eye_mid = l.mean(axis=0).reshape((1, 2)).astype(int)
        eye_dist = np.linalg.norm(left_eye_mid-right_eye_mid) #in pixels
        class_info = self.get_class_attributes()
        self.scaling_factor = eye_dist / class_info['eye_dist']
        dist = (2.55906 * self.focal_length) / self.scaling_factor
        self.target_distance_away = round(dist / 12, 2)
        return

    def start(self):
        """
        call this to start the system up. starts the seperate thread
        :return:
        """
        self.thread.start()
        return self

    def stop(self):
        """
        MUST CALL THIS TO STOP. PUT WHATEVER YOU DO IN A TRY/EXCEPT BLOCK. IF YOU DON'T THE SYSTEM WON'T DEALLOCATE
        CAMERA RESOURCES AND WILL LEAVE THE CAMERA RUNNING, THEN YOU WILL HAVE TO RESTART COMPUTER TO FIX IT
        :return: None
        """
        self.active = False
        self.stream.release()
        cv2.destroyAllWindows()

    def get_target_location(self):
        """
        get function for location tuple. works better with modularity and multithreading
        :return: tuple - (x-min, ymin), (x-max, y-max)
        """
        return self.target_location

    def get_target_dist(self):
        """
        get function for dist num. works better with modularity and multithreading
        :return: float - approx. distance in feet
        """
        return self.target_distance_away

    def get_class_attributes(self):
        """
        this function classifies the target and classifies them as one of 4 classes, then
        returns the expected class attributes in the form of a dict. I plan to add more
        attributes in the future.
        :return: dict - { "eye_dist": float}
        """
        return class_details.class_data[self.pred_class]

    def demo(self, mode):
        """
        this function shows a video that walks the viewer through the steps of
        the calculation.
        :return: None
        """
        prep_frame = self.frame
        if cv2.waitKey(1) == ord('q'):
            self.stop()
            return

        elif mode == "detection":
            if self.target_location:
                string = "Target Midpoint Location in 2D is:  " + str(self.target_location)
                cv2.putText(self.frame, string, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                string = "Cannot find: " + self.target_class
                cv2.putText(self.frame, string, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            if self.object_detected:
                cv2.rectangle(prep_frame, self.target_corners[0], self.target_corners[1], (10, 255, 0), 2)

        elif mode == "face":
            if self.face_detected:
                string = "Extracting facial landmarks:  " + str(self.target_location)
                cv2.putText(self.frame, string, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                string = "Cannot find: " + self.target_class
                cv2.putText(self.frame, string, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            if self.face_detected:
                for (x, y) in self.face:
                    cv2.circle(prep_frame, (x, y), 1, (0, 0, 255), -1)

        elif mode == "classification":
            if self.face_detected:
                info = self.get_class_attributes()
                string = "Predicted Class is:  " + str(info['name'])
                cv2.putText(self.frame, string, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                string = "Cannot find: " + self.target_class
                cv2.putText(self.frame, string, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        elif mode == "dist":
            if self.face_detected:
                arr = self.plot_eyes()
                for (x, y) in arr:
                    cv2.circle(prep_frame, (x, y), 1, (0, 0, 255), -1)
                string = "Target Distance from camera: " + str(self.target_distance_away) + "ft"
                cv2.putText(self.frame, string, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        else:
            print("Unknown mode")

        cv2.imshow('Demo', prep_frame)


# import the correct version of tensorflow based on whats installed and define and interpreter
# from on the .tflite file
if importlib.util.find_spec('tflite_runtime'):
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path='model/detect.tflite')
    classifier = tflite.Interpreter(model_path='model/gender_model.tflite')
else:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path='model/detect.tflite')
    classifier = tf.lite.Interpreter(model_path='model/gender_model.tflite')

interpreter.allocate_tensors()
classifier.allocate_tensors()

#load labels
with open('model/labelmap.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def run_demo():
    init, counter, mult = TargetStream('person', interpreter, classifier), 0, 100
    init.start()
    while counter < 4*mult:
        if counter < 1*mult:
            init.demo('detection')
        elif counter >= 1*mult and counter < 2*mult:
            init.demo('face')
        elif counter >= 2*mult and counter < 3*mult:
            init.demo('classification')
        elif counter >= 3*mult and counter < 4*mult:
            init.demo('dist')
        time.sleep(.01)
        counter += 1

def main():
    init = TargetStream('person', interpreter, classifier)
    init.start()
    while init.active:
        init.show_frame()
        time.sleep(.005)

if __name__ == "__main__":
    try:
        run_demo()
    except:
        init.stop()
