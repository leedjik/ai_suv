from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# import tensorflow as tf
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config

# Req. 2-2	세팅 값 저장
def save_config():
	config

# Req. 4-1	이미지와 캡션 시각화
def visualize_img_caption(image, caption):
	plt.title('<start>'+caption+'<end>')
	img = mpimg.imread(config.args.image_path+image)
	plt.imshow(img)
	plt.show()