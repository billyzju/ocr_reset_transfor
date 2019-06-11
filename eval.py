#_*_coding:utf-8_*_
"""
	eval.py
"""
import dataset
import HyperParameters as hp
from transformer.Models import Transformer
from transformer.Translator import Translator
from torch.autograd import Variable
import torch
import resnet
import HyperParameters as hp
import torch
import torch.nn as nn
import os
import transformer.Constants as Constants
import numpy as np

if __name__ == "__main__":
	torch.cuda.set_device(hp.gpu)
	testLoader = dataset.getDataLoader(is_train=False, batch_size=5, shuffle=False)

	net1 = resnet.resnet34()
	net2 = Transformer(len_encoder=hp.enc_input_len, n_tgt_vocab=hp.num_classes, len_max_seq=hp.max_seq_len, n_layers=hp.n_layers)
	net2.word_prob_prj = nn.LogSoftmax(dim=1)
	net1.cuda().eval()
	#net2.cuda().eval()

	path_to_restore = os.path.join(hp.checkpoint_path, hp.model_path_pre+"_"+str(hp.model_path_idx) + ".pth")
	if os.path.exists(path_to_restore):
		print("restore from:", path_to_restore)
		checkpoint = torch.load(path_to_restore)
		net1.load_state_dict(checkpoint["state_dict_net1"])
		net2.load_state_dict(checkpoint["state_dict_net2"])
		print("restore successfully!")
	else:
		print("fail to restore, path don't exist")
	
	translator = Translator(net2, beam_size=hp.beam_size, max_seq_len=hp.max_seq_len, n_best=hp.n_best)

	print("************************begin infer*********************")
	for imgs_name, imgs, length_imgs, labels, legnth_labels in testLoader:
		imgs = Variable(imgs).cuda()
		length_imgs = Variable(length_imgs).cuda()
      
		enc_img = net1(imgs.float())
		batch_size, channel, height, width = enc_img.shape
		enc_img = enc_img.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, channel)
		batch_pred, batch_prob = translator.translate_batch(enc_img, length_imgs)
        
		label_seq = []
		for seq in labels.data.numpy():
			expression = ""
			for char_idx in seq:
				if char_idx == Constants.EOS:
					break
				else:
					expression += dataset.idx2word.get(char_idx, '')
			label_seq.append(expression)

		pre_seq = []
		for best_pred in batch_pred:
			for seq in best_pred:
				expression = ""
				for char_idx in seq:
					if char_idx == Constants.EOS:
						break
					else:
						expression += dataset.idx2word.get(char_idx, '')
				pre_seq.append(expression)

		for filename, label, pred, prob in zip(imgs_name, label_seq, pre_seq, batch_prob):
			print(filename)
			print("label: ", label)
			print("pred:  ", pred)
			print("porb:  ", np.exp(prob.cpu().data.numpy())[0])
			print()
