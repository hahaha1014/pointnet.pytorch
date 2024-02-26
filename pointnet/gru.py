import torch
import torch.nn as nn


# 例子1，单向一层网络
pretrained_embedding = torch.randn(3, 1024)  # 假设已有的嵌入向量形状为 (10000, 300)
# 将已有的张量转为模型的可训练参数
embed = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
# embed = nn.Embedding(3, 1024) #一共3个词，每个词的词向量维度设置为50维
x = torch.LongTensor([[0, 1, 2]]) # 3个句子，每个句子只有一个词，对应的索引分别时0，1，2
x_embed = embed(x)
print(x_embed.size())
# torch.Size([1, 3, 50]) # [规整后的句子长度，样本个数（batch_size）,词向量维度]

gru = nn.GRU(input_size=1024, hidden_size=1024) # 词向量维度，隐藏层维度
out, hidden = gru(x_embed)

print(out.size())
# torch.Size([1, 3, 50]) # [seq_len,batch_size,output_dim]

print(hidden.size())
# torch.Size([1, 1, 50]) # [num_layers * num_directions, batch_size, hidden_size]


# 例子2，单向2层网络
gru_seq = nn.GRU(10, 20,2) # x_dim,h_dim,layer_num
gru_input = torch.randn(3, 32, 10) # seq_len,batch_size,x_dim
out, h = gru_seq(gru_input)
print(out.size())
print(h.size())

'''
torch.Size([3, 32, 20]) # [seq_len,batch_size,output_dim]
torch.Size([2, 32, 20]) # [num_layers * num_directions, batch_size, hidden_size]

'''


