import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from config import BOS_TOKEN, EOS_TOKEN, device
from util import load_reuters, get_loader, init_weights, save_pretrained


# 1. 数据准备
class NGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # 插入句首、句尾标记
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size:
                continue
            for i in range(context_size, len(sentence)):
                # 模型输入
                context = sentence[i - context_size : i]
                # 模型输出
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return (inputs, targets)


# 2. 模型定义
class FeedForwardNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(FeedForwardNNLM, self).__init__()
        # 词向量层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 线性变换：词向量层 -> 隐含层
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        # 线性变换：隐含层 -> 输出层
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        # 激活函数
        self.activate = F.relu
        init_weights(self)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(inputs.shape[0], -1)
        hidden = self.activate(self.linear1(embeds))
        output = self.linear2(hidden)
        log_probs = F.log_softmax(output, dim=-1)
        return log_probs


# 3. 模型训练
def train():
    embedding_dim = 128
    hidden_dim = 256
    batch_size = 1024
    context_size = 3
    num_epoch = 10

    corpus, vocab = load_reuters()
    dataset = NGramDataset(corpus, vocab, context_size)
    data_loader = get_loader(dataset, batch_size)

    model = FeedForwardNNLM(len(vocab), embedding_dim, context_size, hidden_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    total_losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc="Training Epoch {}".format(epoch)):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            log_probs = model(inputs)
            loss = F.nll_loss(log_probs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Loss: {:.2f}".format(total_loss))
        total_losses.append(total_loss)
    save_pretrained(vocab, model.embeddings.weight.data, "data/ffnnlm.vec")


def main():
    train()


if __name__ == "__main__":
    main()
