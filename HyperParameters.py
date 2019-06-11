#_*_coding:utf-8_*_
"""
	HyperParameters.py
"""
#txt file used in the dataset.py
map_txt = "/data/users/yiweizhu/firm_new/data/map_char_8000.txt"
train_txt = "/data/users/yiweizhu/firm_new/data/train_img_less_50.txt"
test_txt = "/data/users/yiweizhu/firm_new/data/test_img_less_50.txt"
#what the image resize to
HEIGHT = 32
WIDTH = 1200
scale_ratio = 16
#the max_len of train_set sequence
MAX_LEN = 52
num_classes = 8000
max_seq_len = 52

#hyper-paramter
EPOCH = 50
BATCH_SIZE = 50
LEARNING_RATE = 1e-4
n_warmup_steps = 2000
label_smoothing = True

model_restore = True
checkpoint_path = "/data/users/yiweizhu/firm_new/checkpoint"
model_path_pre = "ocr_model"
model_path_idx = 1000
save_step = 10000
gpu = 0

#parameter of transformer
enc_input_len = int((HEIGHT/scale_ratio)*(WIDTH/scale_ratio))
n_layers = 4

#hyper_paramter for translate
beam_size = 2
n_best = 1
