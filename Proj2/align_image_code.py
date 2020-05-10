import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import sys
import cv2



def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale)
    else:
        im2 = sktr.rescale(im2, 1./dscale)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

#todo:
def CC(img1, img2):
    raise NotImplemented

def convolve(img, kernel):
    kernel = cv2.flip(kernel, -1)
    new_img = np.zeros(img.shape)
    for i in np.arange(kernel.shape[0]//2, img.shape[0] - kernel.shape[0]//2):
        for j in np.arange(kernel.shape[0]//2, img.shape[1] - kernel.shape[1]//2):
            for k in range(3): #for colors....?
                center = img[i - kernel.shape[0]//2 : i + kernel.shape[0]//2 , j - kernel.shape[1]//2 : j + kernel.shape[1]//2, k]
                new_img[i, j, k] = (center * kernel).sum() / 255
                
    final_img =(new_img/np.max(new_img))[int(kernel.shape[0]//2):-int(kernel.shape[0]//2), int(kernel.shape[1]//2):-int( kernel.shape[1]//2), :]
    return final_img, kernel.shape[0]//2, kernel.shape[0]//2

def gkern(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def vis_kernal(size, sd):
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = numpy.meshgrid(np.arange(size), np.arange(size))  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, gkern(size, sd)*100)

    plt.show()

def gaussian_blur(img, gaussian_kernel):
    return convolve(img, gaussian_kernel)[0]

def low_pass(img, kernel): #this returns the details of the img, i think thats low pass right..?
    blur,v,h = convolve(img, kernel)
    reshaped_img = img[v:-v, h:-h, :]
    return np.clip(reshaped_img - blur, 0, 1)+.1 #.1 is a constant to make it brighter

def high_pass(img, kernel):
    return gaussian_blur(img, kernel)

def auto_kernel(img):
    x = img.shape[0]
    size = int(x*.085)
    if size % 2 != 0:
        size += 1
    return gkern(size,int(size/5))

def FFT(img):#no idea what this even is
    raise NotImplemented 

def stack_images(img1, img2):
    #add some normalization here?
    
    return img1 + img2



if __name__ == "__main__":
    im1 = sys.argv[1]
    im2 = sys.argv[2]
    im1 = plt.imread(im1)
    im2 = plt.imread(im2)
    im1, im2 = align_images(im1, im2)
    plt.imsave('im1.jpg', im1)
    plt.imsave('im2.jpg', im2)



