import torch

BOS_TOKEN = "<bos>"  # 句首标记
EOS_TOKEN = "<eos>"  # 句尾标记
PAD_TOKEN = "<pad>"  # 补齐标记
UNK_TOKEN = "<unk>"  # 未知标记

WEIGHT_INIT_RANGE = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
