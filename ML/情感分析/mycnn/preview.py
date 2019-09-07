import collections
import numpy as np
from tqdm import tqdm

SENTENCE_LIMIT_SIZE = 20
EMBEDDIGN_SIZE = 100
TRAINFILE_PATH = "D:\\LEARNING\\ACM\\ML\\情感分析\\data\\train_2.txt"
TEST_PATH = "D:\\LEARNING\\ACM\\ML\\情感分析\\data\\test_2.txt"
glove_PATH = "D:\\LEARNING\\ACM\\ML\\情感分析\\data\\glove.6B\\glove.6B.100d.txt"
word_to_token = {}
token_to_word = {}
pos_token = []
neg_token = []
words = set()
word_to_vec = {}
#<pad> 用于补全，<unk> 用于替代没有寻找到的词
vocab = ["<pad>", "<unk>"]

#The total size of vocabulary is: 7578


def loadrawdata(PATH):
    ffile = open(PATH, "r")
    lines = ffile.read()
    ffile.close()
    c = collections.Counter(lines.lower().split())
    sorted(c.most_common(), key=lambda x: x[1])
    for w, f in c.most_common():
        if(f > 1):
            vocab.append(w)
    for w, word in enumerate(vocab):
        word_to_token[word] = w
    for word, token in word_to_token.items():
        token_to_word[token] = word
    print("Train data loaded.")


def convert_text_token(sentence, word_to_token_map=word_to_token, linmit_size=SENTENCE_LIMIT_SIZE):
    """
    根据单词-编码映射表将单个句子转化为token
    @param sentence：句子（str）
    @param word_to_token_map: 单词到编码的映射
    @param linmit_size： 句子最大长度。超过该句子长度进行截断，不足进行pad补全
    """
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]
    tokens = []
    flag = 0
    for word in sentence.lower().split():
        if(word == ".|1" or word == "'|1"):
            flag = 1
            break
        if(word == ".|0" or word == "'|0"):
            flag = 0
            break
        tokens.append(word_to_token_map.get(word, unk_id))
    if(len(tokens) < linmit_size):
        tokens.extend([0]*(linmit_size-len(tokens)))
    else:
        tokens = tokens[:linmit_size]
    return [tokens, flag]


def constructMap():
    """
    构造键值对
    """
    ffile = open(TRAINFILE_PATH)
    line = ffile.readline()
    train_data = []
    while line:
        token = convert_text_token(line)
        train_data.append(token)
        if(token[1] == 1):
            pos_token.append(token[0])
        else:
            neg_token.append(token[0])
        line = ffile.readline()
    ffile.close()
    print("Train data constructed.")
    return train_data


#The number of words which have pretrained-vectors in vocab is: 5858
#The number of words which do not have pretrained-vectors in vocab is: 1720


def loadVec():
    """
    加载词向量
    """
    ffile = open(glove_PATH, 'r', errors='ignore')
    i = 0
    tot = 400000
    for i in tqdm(range(tot), ncols=100, desc="Loading vector..."):
        line = ffile.readline()
        line = line.strip().split()
        now_word = line[0]
        words.add(now_word)
        word_to_vec[now_word] = np.array(line[1:], dtype=np.float32)
    print("Vector loaded.")


def embedding():
    VOCAB_SIZE = len(vocab)  # 7578
    static_embeddings = np.zeros((VOCAB_SIZE, EMBEDDIGN_SIZE))
    for word, token in word_to_token.items():
        tmp_vec = word_to_vec.get(
            word, 4 * np.random.random(EMBEDDIGN_SIZE) - 2)
        static_embeddings[token, :] = tmp_vec
    pad_id = word_to_token['<pad>']
    static_embeddings[pad_id, :] = np.zeros(EMBEDDIGN_SIZE)
    print("Embedding finished.")
    return static_embeddings


def getTest():
    ffile = open(TEST_PATH, "r", errors="ignore")
    line = ffile.readline()
    test_data = []
    while line:
        token = convert_text_token(line)
        test_data.append(token)
        line = ffile.readline()
    ffile.close()
    print("Test data constructed.")
    return test_data


def getArgs():
    loadrawdata(TRAINFILE_PATH)
    train_data = constructMap()
    test_data = getTest()
    loadVec()
    static_embeddings = embedding()
    return static_embeddings, len(vocab), EMBEDDIGN_SIZE, SENTENCE_LIMIT_SIZE, train_data, test_data
