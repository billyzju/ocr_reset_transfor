#_*_coding:utf-8_*_
"""
	train.py
"""
import dataset
import resnet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from transformer.Models import Transformer
import transformer.Constants as Constants
import os
from transformer.Optim import ScheduledOptim
from torch.optim.lr_scheduler import ExponentialLR
import HyperParameters as hp
import torch.nn.functional as F

def train(epoch):
	net1.train()
	net2.train()

	global num_step
	#start_t = time.time()
	for _, imgs, length_imgs, labels, length_labels in trainLoader:
		imgs = Variable(imgs).cuda()
		length_imgs = Variable(length_imgs).cuda()
		length_labels = Variable(length_labels).cuda()
		labels = Variable(labels)

		optimizer.zero_grad()
		output1 = net1(imgs.float())
		#print("after the resnet, the feature shape is: ", output1.shape)
		batch_size, channel, height, width = output1.shape
		len_feature = height * width
		output1 = output1.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, channel)
		#print("after reshape, the feature shape is: ", output1.shape)
	
		#数据切片
		batch_bos = torch.ones(batch_size, 1) * Constants.BOS
		#batch_labels_ = batch_labels[:, :-1]
		batch_labels_train = torch.cat((batch_bos.long(), labels[:, :-1]), 1).cuda()
		#print("shape of bactch_labels_train:", batch_labels_train.shape)
		#batch_labels_train = batch_labels_train.cuda()

		output = net2(output1, length_imgs, batch_labels_train, length_labels)
		#len_l, dim = output.shape #[batch_size*len, vocab_size]
		#label = torch.zeros_like(output).scatter(1, batch_labels.view(-1, 1), 1)
		#print("shape of output:", output.shape)
		#print("shape of batch_labels:", batch_labels.shape)
		if hp.label_smoothing:
			eps = 0.1
			n_class = output.size(1)
			labels = labels.cuda()
			one_hot = torch.zeros_like(output).scatter(1, labels.view(-1, 1), 1)
			one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
			log_prb = F.log_softmax(output, dim=1)

			non_pad_mask = labels.view(-1).ne(Constants.PAD)
			loss = -(one_hot * log_prb).sum(dim=1)
			loss = loss.masked_select(non_pad_mask).sum()	
			loss /= non_pad_mask.sum()	
		else:
			loss = lossFunction(output, labels.view(-1).cuda())

		loss.backward()
		optimizer.step()
		#optimizer.step_and_update_lr()
		num_step += 1
		if num_step % hp.n_warmup_steps == 0:
			optimizer_scheduler.step()	

		if num_step % 100 == 0:
			#start_t = time.time()
			##*print("step:{}, learning_rate:{}, loss:{}, time:{}".format(num_step, optimizer.lr, loss, time.time()-start_t))
			print("step:{}, learning_rate:{}, loss:{:.4f}, time:{:.4f}".format(num_step, optimizer_scheduler.get_lr()[0], loss, time.time()-start_t))
		if num_step % hp.save_step == 0:
			save_model = {"state_dict_net1": net1.state_dict(),
                          "state_dict_net2": net2.state_dict(),
                          ##*"optimizer_dict": optimizer._optimizer.state_dict(),
                          "optimizer_dict": optimizer.state_dict(),
						  "num_step": num_step}
			save_name = os.path.join(hp.checkpoint_path, hp.model_path_pre + "_" + str(num_step) + ".pth")
			print("save model at steps of {}".format(num_step))
			torch.save(save_model, save_name)

if __name__ == "__main__":
	torch.cuda.set_device(hp.gpu)
	
	net1 = resnet.resnet34()
	net2 = Transformer(len_encoder=hp.enc_input_len, n_tgt_vocab=hp.num_classes, len_max_seq=hp.MAX_LEN, n_layers=hp.n_layers)
	net1 = net1.cuda()
	net2 = net2.cuda()
		
	trainLoader = dataset.getDataLoader(is_train=True, batch_size=hp.BATCH_SIZE, shuffle=True)
	iter_one_epoch = len(trainLoader)
	print("iteration_every_epoch: ", iter_one_epoch)
	#testloader = dataset.getDataLoader(is_train=False, batch_size=BATCH_SIZE, shuffle=False)
	lossFunction = nn.CrossEntropyLoss(ignore_index = Constants.PAD)
	optimizer_ = optim.Adam([{'params': net1.parameters()}, {'params':filter(lambda x: x.requires_grad, net2.parameters())}], betas=[0.9, 0.98], lr=hp.LEARNING_RATE)
	optimizer = optimizer_
	optimizer_scheduler = ExponentialLR(optimizer_, 0.98)
	#optimizer = ScheduledOptim(optimizer_, learning_rate=hp.LEARNING_RATE, n_warmup_steps=hp.n_warmup_steps)

	if not os.path.exists(hp.checkpoint_path):
		os.makedirs(hp.checkpoint_path)
	num_step = 1
	model_restore_path = os.path.join(hp.checkpoint_path, hp.model_path_pre+"_"+str(hp.model_path_idx)+".pth")
	if hp.model_restore and os.path.exists(model_restore_path):
		print("restore model from {}".format(model_restore_path))
		checkpoint = torch.load(model_restore_path)
		net1.load_state_dict(checkpoint["state_dict_net1"])
		net2.load_state_dict(checkpoint["state_dict_net2"])
		optimizer.load_state_dict(checkpoint["optimizer_dict"])
		#optimizer.init_lr = checkpoint["learning_rate"]
		##*optimizer.n_current_steps = checkpoint["num_step"]
		#print(checkpoint["optimizer_dict"])
		num_step = checkpoint["num_step"]
		print("restore sucessfully!")
	print("******************************begin train******************************")
	start_t = time.time()
	for epoch in range(hp.EPOCH):
		train(epoch)
