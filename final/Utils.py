import numpy as np
import time
import imutils
import cv2

def to_polar(pts):
    """
    converts an (r,c) tuple from cartesan to polar coordinates. 
    Expects and returns everything in terms of percent of the frame
    params:
        pts - tuple: (r, c) or (y, x)
    returns:
        tuple - (radius, angle) where angle is in degrees
    """
    y, x = pts[0] - .5, pts[1] - .5 #centering and unpacking points
    r = np.sqrt([np.square(x-.5), np.square(y-.5)])
    t = np.arctan2([y-.5], [x-.5]) * (180 / np.pi)
    return r, t[0]

def auto_target(coordinates):
    """
    Like above everything is expected and returned in terms of percent of 
    the frame so resolution and shape are irrelevant. This function centers
    the cannon on the target returned by the detection object

    params:
        coordinates - tuple: location of the target as returned by the 
        Target class defined in Detection.py

    returns:
        2 tuples in the form of:
             (horizontal direction sting i.e. 'left' or 'right', precent horzontal (cosine of angle) )
             (vertical direction sting i.e. 'up' or 'down', precent horzontal (sin of angle) )

    """
    r,t = to_polar(coordinates)
    y, x = coordinates[0] - .5, coordinates[1] - .5 #centering and unpacking points
    x_mult, y_mult = np.cos(t), np.sin(t)
    x_mult, y_mult = x_mult / np.sum([x_mult, y_mult]), y_mult / np.sum([x_mult, y_mult])
    x_dir = 'left' if x < 0 else 'right'
    y_dir = 'up' if y > 0 else 'down'
    return (x_dir, x_mult), (y_dir, x_mult), r

def parse_pinfile(filepath):
    file = open(filepath)
    lines = file.readlines()
    dic = {}
    for line in lines:
        if line[0] != '\n' and line[0] != '#':
            line = line.replace(' ', '')
            line = line.split('=')
            dic[line[0]] = int(line[1].replace('\n', ''))
    return dic

def passive_scan(interface):
    """
    moves the camera around and searchers for the target. 
    """
    interface.left()
    time.sleep(.1)
    return

def move(hor, vert, reps, interface, step_mult = 5):
    if hor[0] == 'left':
        num_turns = int(reps * hor[1] * step_mult)
        print("Moving left by ", num_turns, " steps")
        [interface.left() for _ in range(num_turns)]
    else:
        num_turns = int(reps * hor[1] * step_mult)
        print("Moving right by ", num_turns, " steps")
        [interface.right() for _ in range(num_turns)]

    if vert[0] == 'up':
        num_turns = int(reps * vert[1] * step_mult)
        print("Moving up by ", num_turns, " steps")
        [interface.up() for _ in range(num_turns)]
    else:
        num_turns = int(reps * vert[1] * step_mult)
        print("Moving down by ", num_turns, " steps")
        [interface.down() for _ in range(num_turns)]
    return

def find_marker(image):
     # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)

def find_focal_length():
    """
    put a normal piece of paper 2 feet away to use.
    :return:
    """
    cap = cv2.VideoCapture(0)  # video capture source camera (Here webcam of laptop)
    ret, frame = cap.read()
    while (True):
        ret, frame = cap.read()
        cv2.imshow('img1', frame) # display the captured image
        if cv2.waitKey(1) & 0xFF == ord('y'):  # save on pressing 'y'
            img = frame
            cv2.destroyAllWindows()
            break

    if not ret:
        print('Problem taking picture')

    # initialize the known distance from the camera to the object, which
    # in this case is 24 inches
    KNOWN_DISTANCE = 24.0
    # initialize the known object width, which in this case, the piece of
    # paper is 12 inches wide
    KNOWN_WIDTH = 11.0
    # load the furst image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length
    marker = find_marker(img)
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

    print("focal length of camera in inches: ", focalLength)

    return focalLength


def main():
    find_focal_length()

if __name__ == "__main__":
    main()






