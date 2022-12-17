import time
import torch
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

input_path = "sentiment_analysis_eng/raw_data/book_reviews.csv"


class TextClassificationModel(nn.Module):
    """
    用于文本分类的模型
    词嵌入之后全联接然后分类
    """

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def yield_tokens(input_path):
    """
    生成迭代器
    主要是把分词后的句子传入词典生成器
    """
    # 分词器 torchtext中集成的分词器 功能比较强大 可以看源码 做了_basic_english_normalize
    tokenizer = get_tokenizer("basic_english")
    all_sample = pd.read_csv(input_path)  # 读取输入样本
    for text in all_sample["text"]:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(input_path), specials=["<unk>"])  # 从迭代器变成一个词典
vocab.set_default_index(vocab["<unk>"])  #

# gpu选择或者cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_class = 2
vocab_size = len(vocab)
embed_dim = 16
# 初始化一个模型
model = TextClassificationModel(vocab_size, embed_dim, num_class).to(device)
# 超参数
EPOCHS = 20  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

# 损失函数
criterion = torch.nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# 学习率衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None


def train(dataloader):
    """
    训练模型
    """
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        """
        text: 批次内的 index 串成一个向量
        offsers: 批次内 每一个句子的位置
        """
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count)
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    """
    测试数据输入模型
    """
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


tokenizer = get_tokenizer("basic_english")

# text转换成 数值 然后是tensor
text_pipeline = lambda x: vocab(tokenizer(x))


def trans_label(label):
    """
    label数据转换
    """
    label_dic = {"__label__1": 0, "__label__2": 1}
    return label_dic[label]


def collate_batch(batch):
    """
    造输入模型的数据
    """
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(trans_label(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def yield_dataset():
    """
    生成所有数据的迭代器
    """
    all_sample = pd.read_csv(input_path)  # 读取输入样本
    text = all_sample["text"]
    label = all_sample["label"]
    for index in range(len(all_sample["text"])):
        yield (text[index], label[index])


all_dataset = to_map_style_dataset(yield_dataset())
# 数据分割，训练机，验证集，测试集
train_data, valid_data, test_data = random_split(all_dataset, [0.8, 0.1, 0.1])

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(epoch, time.time() - epoch_start_time, accu_val)
    )
    print("-" * 59)

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))
