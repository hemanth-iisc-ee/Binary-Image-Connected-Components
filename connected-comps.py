import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as sio
from skimage.transform import rescale, resize
from collections import deque
import time
import pdb


class ImgCComps():
	def __init__(self, fg=1, connectivity=4):
		if connectivity == 8:
			self.neighbors = np.array([[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1]])
		else:
			self.neighbors = np.array([[0,-1],[-1,0],[0,1],[1,0]]) # 4-connectivity

		self.cc_count = 2
		self.stack = deque([])
		self.fg = fg
		self.bg = 2 + ~fg
		self.ip_img = None
		self.h = None
		self.w = None
		return

	def dfs(self, start_cord):
		def get_ut_neighbors(cord): # get Untarversed neighbors

			neighbors = self.neighbors + np.array(cord)

			# neighbors = neighbors[(neighbors[:,0] >= 0) & (neighbors[:,1] >= 0),:]
			# neighbors = neighbors[(neighbors[:,0] < self.h) & (neighbors[:,1] < self.w),:]		

			# values = self.ip_img[neighbors[:,0],neighbors[:,1]]
			# neighbors = neighbors[values == self.fg,:]
			# return neighbors.tolist()

			return [x for x in neighbors if self.ip_img[x[0],x[1]] == self.fg]
		

		self.ip_img[start_cord[0],start_cord[1]] = self.cc_count
		self.stack.extendleft(get_ut_neighbors(start_cord))
		while len(self.stack) != 0:
			a_cord = self.stack.popleft()
			self.ip_img[a_cord[0],a_cord[1]] = self.cc_count
			self.stack.extendleft(get_ut_neighbors(a_cord))
		return

	def dfs_recursive(self, a_cord):
		self.ip_img[a_cord[0],a_cord[1]] = self.cc_count
		neighbors = self.neighbors + np.array(a_cord)
		for x in neighbors:
			if self.ip_img[x[0],x[1]] == self.fg:
				self.dfs_recursive(x)
		return


	def process(self,img):
		def reset(img):
			self.cc_count = 2
			self.stack = deque([])
			self.ip_img = img.copy()
			h = img.shape[0]
			w = img.shape[1]
			self.ip_img = np.vstack((self.bg*np.ones((1,w)),self.ip_img,self.bg*np.ones((1,w))))
			self.ip_img = np.hstack((self.bg*np.ones((h+2,1)),self.ip_img,self.bg*np.ones((h+2,1))))
			self.h = h + 2
			self.w = w + 2
			return

		reset(img) # reset state of processor
		for i in range(self.h): # scanning
			for j in range(self.w):
				if self.ip_img[i,j] == self.fg:
					cc_seed = (i,j)
					self.dfs_recursive(cc_seed)
					self.cc_count += 1
		return self.ip_img[1:-1,1:-1]


# Main Code
filename = '1.png'
sys.setrecursionlimit(int(1e7))
img = sio.imread(filename,as_gray=True)
img = np.array(resize(img,(img.shape[0],img.shape[1]),anti_aliasing=True))
img[img <= 0.5] = 0
img[img > 0.5] = 1

cc = ImgCComps(fg=1,connectivity=8)
t0 = time.time()
y = cc.process(img)
t1 = time.time()
print('Time: {}sec '.format(t1 - t0))
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title('Input')
plt.subplot(122)
plt.imshow(y,cmap='jet')
plt.title('Labeled Image')
plt.show()