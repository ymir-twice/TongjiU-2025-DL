import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn as rnn_lstm

start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems1(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    读者注解:  
        1. 编码使用的是词袋法
        2. process_poems1 用于处理 poems.txt, process_poems2 用于处理 tangshi.txt
    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':', 1)
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]
    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        # content = ''
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' '' ', '').replace('，','').replace('。','')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                    start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # print(content)
                    content = start_token + content + end_token
                    poems.append(content)
                    # content = ''
            except ValueError as e:
                # print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    """
    生成batch数据:
        poems_vec.shape: (poem_num, poem_length)
        x_batches.shape: (n_chunk, batch_size, poem_length)
        y_batches.shape: (n_chunk, batch_size, poem_length)

        作为 label 的 y，其实是 row[1:] + row[-1]
    """
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        # print(x_data[0])
        # print(y_data[0])
        # exit(0)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    # 处理数据集
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    # 生成batch
    print("finish  loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)
    word_embedding = rnn_lstm.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
    rnn_model = rnn_lstm.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    import os
    if os.path.exists('./poem_generator_rnn'):
        rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))

    # optimizer = optim.Adam(rnn_model.parameters(), lr= 0.001)
    optimizer=optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss()
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.

    from tqdm import tqdm
    for epoch in range(20):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        with tqdm(total=n_chunk, desc=f"Epoch {epoch}") as pbar:  # 只在 batch 维度使用进度条
            for batch in range(n_chunk):
                batch_x = batches_inputs[batch]
                batch_y = batches_outputs[batch]  # (batch, time_step)

                loss = 0
                for index in range(BATCH_SIZE):
                    x = np.array(batch_x[index], dtype=np.int64)
                    y = np.array(batch_y[index], dtype=np.int64)
                    x = Variable(torch.from_numpy(np.expand_dims(x, axis=1)))
                    y = Variable(torch.from_numpy(y))
                    pre = rnn_model(x)
                    loss += loss_fun(pre, y)

                loss = loss / BATCH_SIZE  # 计算平均损失
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)  # 注意 `_` 结尾
                optimizer.step()

                pbar.set_postfix(loss=loss.item())  # 更新 batch 级别进度条
                pbar.update(1)  # 手动更新进度条

        # 每轮保存一次模型
        torch.save(rnn_model.state_dict(), './poem_generator_rnn.pt')
        print(f"[Epoch {epoch}] Model saved to './poem_generator_rnn.pt'")


def to_word(predict, vocabs):  # 预测的结果转化成汉字
    """
    :params predict: 预测结果（尚未经过 argmax ）
    :params vocabs: 词表（对应于 process_poem 函数返回值的 words）
    """
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    """
    :param poem: 生成的诗歌
    """
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


def gen_poem(begin_word):
    # poems_vector, word_int_map, vocabularies = process_poems2('./tangshi.txt')  #  use the other dataset to train the network
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn_lstm.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                                   embedding_dim=100, lstm_hidden_dim=128)

    rnn_model.load_state_dict(torch.load('./poem_generator_rnn.pt'))
    # 指定开始的字

    poem = begin_word
    word = begin_word
    while word != end_token:
        input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        input = Variable(torch.from_numpy(input))
        output = rnn_model(input, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        poem += word
        # print(word)
        # print(poem)
        if len(poem) > 30:
            break
    return poem



# run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。


pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("君"))


