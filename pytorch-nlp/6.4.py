# 定义两个list分别存放两个板块的帖子数据
import jieba
import torch
import torch.nn as nn
import random
from tqdm import tqdm

torch.randn(2, 2)
academy_titles = []
job_titles = []
trained_lines = 0
with open('academy_titles.txt', encoding='utf8') as f:
    for line in f:  # 按行读取文件
        academy_titles.append(list(jieba.cut(line.strip())))  # strip 方法用于去掉行尾空格
with open('job_titles.txt', encoding='utf8') as f:
    for line in f:  # 按行读取文件
        job_titles.append(list(jieba.cut(line.strip())))  # strip 方法用于去掉行尾空格

print(academy_titles[2])

word_set = set()
for title in academy_titles:
    for word in title:
        word_set.add(word)
for title in job_titles:
    for word in title:
        word_set.add(word)
print(len(word_set))

word_list = list(word_set)
# 加一个 UNK
n_chars = len(word_set) + 1


def title_to_tensor(title):
    tensor = torch.zeros(len(title), dtype=torch.long)
    for li, word in enumerate(title):
        try:
            ind = word_list.index(word)
        except ValueError:
            ind = n_chars - 1
        tensor[li] = ind
    return tensor


class RNN(nn.Module):
    def __init__(self, word_count, embedding_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(word_count, embedding_size)
        self.i2h = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(embedding_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden):
        word_vector = self.embedding(input_tensor)
        combined = torch.cat((word_vector, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def run_rnn(rnn, input_tensor):
    global trained_lines
    hidden = rnn.initHidden()
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(input_tensor[i].unsqueeze(dim=0), hidden)

    trained_lines = trained_lines + 1
    print("line trained", trained_lines)
    return output


def train(rnn, criterion, input_tensor, category_tensor):
    # 前向传播
    output = run_rnn(rnn, input_tensor)
    loss = criterion(output, category_tensor)
    # 反向传播
    rnn.zero_grad()
    loss.backward()

    # 根据梯度更新模型的参数
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def evaluate(rnn, input_tensor):
    with torch.no_grad():
        rnn.initHidden()
        output = run_rnn(rnn, input_tensor)
        return output


# rnn = RNN(n_chars, 100, 128, 2)
# input_tensor = title_to_tensor(academy_titles[0])
# print('input_tensor:\n', input_tensor)
# hidden = rnn.initHidden()
# output, hidden = rnn(input_tensor[0].unsqueeze(dim=0), hidden)
# print('output:\n', output)
# print('hidden:\n', hidden)
# print('size of hidden:\n', hidden.size())

all_data = []
categories = ["考研考博", "招聘信息"]

for line in academy_titles:
    all_data.append((title_to_tensor(line), torch.tensor([0], dtype=torch.long)))
for line in job_titles:
    all_data.append((title_to_tensor(line), torch.tensor([1], dtype=torch.long)))

random.shuffle(all_data)
data_len = len(all_data)
split_ratio = 0.7
train_data = all_data[:int(data_len*split_ratio)]
test_data = all_data[int(data_len*split_ratio):]
print("Train data size: ", len(train_data))
print("Test data size: ", len(test_data))

epoch = 1
embedding_size = 200
n_hidden = 10
n_categories = 2
learning_rate = 0.005
rnn = RNN(n_chars, embedding_size, n_hidden, n_categories)
# rnn.initHidden()
# rnn.train()
criterion = nn.NLLLoss()
loss_sum = 0
all_losses = []
plot_every = 100
for e in range(epoch):
    for ind, (title_tensor, label) in enumerate(tqdm(train_data)):
        output, loss = train(rnn, criterion, title_tensor, label)
        loss_sum += loss
        if ind % plot_every == 0:
            all_losses.append(loss_sum / plot_every)
            loss_sum = 0
    c = 0
    for title, category in tqdm(test_data):
        output = evaluate(rnn, title)
        topn, topi = output.topk(1)
        if topi.item() == category[0].item():
            c += 1
    print('accuracy', c / len(test_data))

