import torch
from torch.utils.data import DataLoader
from vocab import Vocab
from config import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, WEIGHT_INIT_RANGE


def load_reuters():
    from nltk.corpus import reuters

    text = reuters.sents()
    text = [[word.lower() for word in sentence] for sentence in text]
    vocab = Vocab.build(text, reserved_tokens=[BOS_TOKEN, EOS_TOKEN, PAD_TOKEN])
    corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]
    return corpus, vocab


def get_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle)
    return data_loader


def init_weights(model):
    for name, param in model.named_parameters():
        if "embedding" not in name:
            torch.nn.init.uniform_(param, a=-WEIGHT_INIT_RANGE, b=WEIGHT_INIT_RANGE)


def save_pretrained(vocab, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write("{} {}\n".format(embeds.shape[0], embeds.shape[1]))
        for idx, token in enumerate(vocab.idx_to_token):
            vec = " ".join(f"{x}" for x in embeds[idx])
            writer.write(f"{token} {vec}\n")


if __name__ == "__main__":
    # load_reuters
    corpus, vocab = load_reuters()
    print(len(corpus))
    print(corpus[10])
    print(len(vocab))
