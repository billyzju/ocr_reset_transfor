#_*_coding:utf-8_*_
"""
	dataset.py
"""
import torch
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms
import time
import transformer.Constants as Constants
import HyperParameters as hp

word2idx = {}
idx2word = {}

with open(hp.map_txt) as f:
	lines = f.readlines()

for line in lines:
	line = line.strip()
	word, idx = line[0], int(line[2:]) + 1
	word2idx[word] = idx
	idx2word[idx] = word
print("max_map_value:", max(idx2word.keys()))

class MyDataset(Dataset):
	def __init__(self, img_txt):
		self.imgs = []
		with open(img_txt) as f:
			filenames = f.readlines()
		for filename in filenames:
			filename = filename.strip()
			self.imgs.append(filename)
	
	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, index):
		img_name = self.imgs[index]
		txt_name = os.path.splitext(img_name)[0] + '.txt'
		with open(txt_name) as f:
			anno = f.read().strip().lower()
			#print(anno)
		if anno == "":
			code = [Constants.PAD]
		else:
			code = [word2idx.get(char, Constants.UNK) for char in list(anno)]
		code.append(Constants.EOS)
		length_code = torch.cat((torch.arange(1, len(code)+1), torch.zeros(hp.MAX_LEN-len(code)).long()), 0)
		while(len(code) < hp.MAX_LEN):
			code.append(Constants.PAD)
		
		im = cv2.imread(img_name)
		height, width, channel = im.shape
		new_width = int(width/height*hp.HEIGHT)
		if new_width > hp.WIDTH:
			im = cv2.resize(im, (hp.WIDTH, hp.HEIGHT))
			length_ = hp.enc_input_len
			length_image = torch.arange(1, length_+1)
		else:
			padding_width = hp.WIDTH - new_width
			im = np.concatenate((cv2.resize(im, (new_width, hp.HEIGHT)), np.ones((hp.HEIGHT, padding_width, channel))*235.5), axis=1)
			length_ = min(hp.enc_input_len, (int((new_width/hp.WIDTH)*(hp.WIDTH/hp.scale_ratio))+1)*int((hp.HEIGHT/hp.scale_ratio)))
			length_pad = hp.enc_input_len - length_
			length_image = torch.cat((torch.arange(1, length_+1), torch.zeros(length_pad).long()), 0)
		#im = transforms.ToTensor()(im)*255
		im = torch.from_numpy(im)
		im = im.permute(2, 0, 1).contiguous()

		return img_name, im.double(), length_image, torch.tensor(code), length_code

def getDataLoader(is_train=True, batch_size=50, shuffle=True):
	if is_train:
		return DataLoader(dataset=MyDataset(hp.train_txt), batch_size=batch_size, shuffle=shuffle)
	else:
		return DataLoader(dataset=MyDataset(hp.test_txt), batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
	train_loader = getDataLoader(is_train=True, batch_size=5, shuffle=True)
	test_loader = getDataLoader(is_train=True, batch_size=50, shuffle=False)
	print(len(train_loader))
	count = 0
	for epoch in range(2):
		for img_name, img, length_image, code, length_code in train_loader:
			#image = batch_img[0, :]
			#image = image.numpy()
			#count += 1
			print(img_name[0])
			#name = "img_" + str(count) + ".jpg"
			#cv2.imwrite(name, image)
			#print(batch_length.eq(0))
			#print(length_image)
			#print(code)
			#print(img[0])
			#cv2.imwrite("1.jpg", img[0].int().permute(1, 2, 0).contiguous().numpy())
			#print("batch_image.shape:{}, batch_label.shape:{}, bacth_length.shape{}".format(batch_img.shape, batch_label.shape, batch_length.shape))
			time.sleep(2)
		
		
