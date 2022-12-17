import time
import torch
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

from model_api import MyTransformerModel

input_path = "sentiment_analysis_eng/raw_data/book_reviews.csv"


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


vocab = build_vocab_from_iterator(yield_tokens(input_path), specials=["<pad>", "<unk>"])  # 从迭代器变成一个词典
vocab.set_default_index(vocab["<unk>"])  #


# gpu选择或者cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(vocab)

SEED = 126
EPOCH_NUM = 5  # 训练轮次
BATCH_SIZE = 128
EMBEDDING_DIM = 100
LEARNING_RATE = 1e-3
SENTENCE_LEN = 120
HEAD_NUM = 2  # 多头self attention 的头数量
# 实例化一个 MyTransformerModel模型
model = MyTransformerModel(vocab_size, EMBEDDING_DIM, p_drop=0.5, h=HEAD_NUM, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)  # 优化器
criteon = nn.BCEWithLogitsLoss()  # 损失函数

# 分词器
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
    label_list, text_list, mask_tensors = [], [], []
    for (_text, _label) in batch:
        label_list.append(trans_label(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        real_sentence_len = processed_text.shape[0]  # 句子真实的长度
        # 限制句子的长度为 SENTENCE_LEN
        if real_sentence_len > SENTENCE_LEN:
            processed_text = processed_text[:SENTENCE_LEN]
        elif real_sentence_len < SENTENCE_LEN:
            processed_text = torch.cat((processed_text, torch.zeros(SENTENCE_LEN - real_sentence_len, dtype=int)), 0)
        mask_tensor = 1 - (processed_text == 0).float()
        mask_tensors.append(mask_tensor)
        text_list.append(processed_text)
    label = torch.tensor(label_list, dtype=torch.float32)

    text = torch.stack(text_list, 0)
    mask = torch.stack(mask_tensors, 0)
    return label.to(device), text.to(device), mask.to(device)


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
# 模型能接受的data 数据加载器
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)


# 计算准确率
def binary_acc(preds, y):

    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


# 训练函数
def train(model, dataloader, optimizer, criteon):

    avg_loss = []
    avg_acc = []
    model.train()  # 表示进入训练模式

    for idx, (label, text, mask) in enumerate(dataloader):
        pred = model(text, mask)
        loss = criteon(pred, label)
        acc = binary_acc(pred, label).item()  # 计算每个batch的准确率
        avg_loss.append(loss.item())
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc


# 评估函数
def evaluate(model, dataloader, criteon):

    avg_loss = []
    avg_acc = []
    model.eval()  # 表示进入测试模式

    with torch.no_grad():

        for idx, (label, text, mask) in enumerate(dataloader):

            pred = model(text, mask)

            loss = criteon(pred, label)
            acc = binary_acc(pred, label).item()
            avg_loss.append(loss.item())
            avg_acc.append(acc)

    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc


# 训练模型，并打印模型的表现
best_valid_acc = float("-inf")

for epoch in range(EPOCH_NUM):

    start_time = time.time()

    train_loss, train_acc = train(model, train_dataloader, optimizer, criteon)
    dev_loss, dev_acc = evaluate(model, valid_dataloader, criteon)

    end_time = time.time()

    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    if dev_acc > best_valid_acc:  # 只要模型效果变好，就保存
        best_valid_acc = dev_acc
        torch.save(model.state_dict(), "sentiment_analysis_eng/save/wordavg-model.pt")

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc*100:.2f}%")


# 用保存的模型参数预测数据
model.load_state_dict(torch.load("sentiment_analysis_eng/save/wordavg-model.pt"))
test_loss, test_acc = evaluate(model, test_dataloader, criteon)
print(f"Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%")
