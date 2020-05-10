import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd



if __name__ == '__main__':
	im_1_path, im_2_path, num_clicks, save_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
	print(im_1_path, im_2_path, num_clicks)
	im1, im2 = plt.imread(im_1_path), plt.imread(im_2_path)
	if not im1.shape == im2.shape:
		print('Warning: images are not the same shape')
	plt.imshow(im1)
	x = plt.ginput(num_clicks, -1)
	plt.close()
	im1_x, im1_y = zip(*x)


	plt.imshow(im2)
	y = plt.ginput(num_clicks, -1)
	plt.close()
	im2_x, im2_y = zip(*y)
	df = pd.DataFrame(
		data={'im1_x': im1_x,
			  'im1_y': im1_y,
			  'im2_x': im2_x,
			  'im2_y': im2_y})
	df.to_csv(save_path)

	

