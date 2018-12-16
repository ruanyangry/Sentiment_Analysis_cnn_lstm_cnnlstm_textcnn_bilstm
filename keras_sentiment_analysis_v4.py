# _*_ coding:utf-8 _*_

'''
@Author: Ruan Yang
@Date: 2018.12.15
@Purpose: 基于 keras 构建 textcnn 算法
@Reference: https://www.cnblogs.com/bymo/p/9675654.html
'''

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from keras import Input
from keras.layers import Conv1D,MaxPool1D,Dense,Flatten,concatenate,Embedding
from keras.models import Model

import codecs
import jieba

datapaths=r"C:\Users\RY\Desktop\情感分析\SentimentAnalysis-master\data"

positive_data=[]
y_positive=[]
neutral_data=[]
y_neutral=[]
negative_data=[]
y_negative=[]

print("#------------------------------------------------------#")
print("加载数据集")
with codecs.open(datapaths+"\\pos.csv","r","utf-8") as f1,\
     codecs.open(datapaths+"\\neutral.csv","r","utf-8") as f2,\
     codecs.open(datapaths+"\\neg.csv","r","utf-8") as f3:
    for line in f1:
        positive_data.append(" ".join(i for i in jieba.lcut(line.strip(),cut_all=False)))
        y_positive.append([1,0,0])
    for line in f2:
        neutral_data.append(" ".join(i for i in jieba.lcut(line.strip(),cut_all=False)))
        y_neutral.append([0,1,0])
    for line in f3:
        negative_data.append(" ".join(i for i in jieba.lcut(line.strip(),cut_all=False)))
        y_negative.append([0,0,1])
        
print("positive data:{}".format(len(positive_data)))
print("neutral data:{}".format(len(neutral_data)))
print("negative data:{}".format(len(negative_data)))

x_text=positive_data+neutral_data+negative_data
y_label=y_positive+y_neutral+y_negative
print("#------------------------------------------------------#")
print("\n")

from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
import collections

max_document_length=200
min_frequency=1


vocab = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency, tokenizer_fn=list)
x = np.array(list(vocab.fit_transform(x_text)))
vocab_dict = collections.OrderedDict(vocab.vocabulary_._mapping)

with codecs.open(r"C:\Users\RY\Desktop\vocabulary.txt","w","utf-8") as f:
    for key,value in vocab_dict.items():
        f.write("{} {}\n".format(key,value))
        
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("数据混洗")
np.random.seed(10)
y=np.array(y_label)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

test_sample_percentage=0.2
test_sample_index = -1 * int(test_sample_percentage * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

train_positive_label=0
train_neutral_label=0
train_negative_label=0
test_positive_label=0
test_neutral_label=0
test_negative_label=0

for i in range(len(y_train)):
    if y_train[i,0] == 1:
        train_positive_label += 1
    elif y_train[i,1] == 1:
        train_neutral_label += 1
    else:
        train_negative_label += 1
        
for i in range(len(y_test)):
    if y_test[i,0] == 1:
        test_positive_label += 1
    elif y_test[i,1] == 1:
        test_neutral_label += 1
    else:
        test_negative_label += 1
        
print("训练集中 positive 样本个数：{}".format(train_positive_label))
print("训练集中 neutral 样本个数：{}".format(train_neutral_label))
print("训练集中 negative 样本个数：{}".format(train_negative_label))
print("测试集中 positive 样本个数：{}".format(test_positive_label))
print("测试集中 neutral 样本个数：{}".format(test_neutral_label))
print("测试集中 negative 样本个数：{}".format(test_negative_label))

print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("读取預训练词向量矩阵")

pretrainpath=r"E:\中科大MS\預训练模型\\"

embedding_index={}

with codecs.open(pretrainpath+"sgns.wiki.bigram","r","utf-8") as f:
    line=f.readline()
    nwords=int(line.strip().split(" ")[0])
    ndims=int(line.strip().split(" ")[1])
    for line in f:
        values=line.split()
        words=values[0]
        coefs=np.asarray(values[1:],dtype="float32")
        embedding_index[words]=coefs
        
print("預训练模型中Token总数：{} = {}".format(nwords,len(embedding_index)))
print("預训练模型的维度：{}".format(ndims))
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("将vocabulary中的 index-word 对应关系映射到 index-word vector形式")

embedding_matrix=[]
notfoundword=0

for word in vocab_dict.keys():
    if word in embedding_index.keys():
        embedding_matrix.append(embedding_index[word])
    else:
        notfoundword += 1
        embedding_matrix.append(np.random.uniform(-1,1,size=ndims))
        
embedding_matrix=np.array(embedding_matrix,dtype=np.float32) # 必须使用 np.float32
print("词汇表中未找到单词个数：{}".format(notfoundword))
print("#----------------------------------------------------------#")
print("\n")

# 定义 textcnn 函数

def textcnn(filters,max_sequence_length,max_token_num,embedding_dim,num_classes,embedding_matrix=None):
    '''
    TextCNN Network Structure
    1. Embedding layers
    2. Convolution layer
    3. max-pooling
    4. softmax layer
    
    max_sequence_length: 输入句子的最大长度
    max_token_num: 这个是指最大的 token 个数，感觉上是针对 英文字符 说的
    embedding_dim：嵌入矩阵的维度
    num_classes: 输出类别个数，二分类就设定为 2
    embedding_matrix=None：这个的意思是是否适用 嵌入矩阵
    '''
    x_input=Input(shape=(max_sequence_length,))
    logging.info("x_input.shape:{}".format(str(x_input.shape)))
   
    if embedding_matrix is None:
        x_emb=Embedding(input_dim=max_token_num,output_dim=embedding_dim,input_length=max_sequence_length)(x_input)
    else:
        x_emb=Embedding(input_dim=max_token_num,output_dim=embedding_dim,input_length=max_sequence_length,\
                       weights=[embedding_matrix],trainable=False)(x_input)
       
    logging.info("x_emb.shape:{}".format(str(x_emb.shape)))
    
    pool_output=[]
    kernel_sizes=[2,3,4]
    for kernel_size in kernel_sizes:
        c=Conv1D(filters=filters,kernel_size=kernel_size,strides=1)(x_emb)
        #p=MaxPool1D(pool_size=int(c.shape[1]))(c)
        p=MaxPool1D(max_sequence_length-kernel_size+1)(c)
        pool_output.append(p)
        logging.info("kernel_size:{} \t c.shape:{} \t p.shape:{}".format(kernel_size,str(c.shape),str(p.shape)))
    #pool_output = concatenate([p for p in pool_output])
    pool_output = concatenate(pool_output,axis=1)
    x_flatten=Flatten()(pool_output) # (?,6)
    y=Dense(num_classes,activation="softmax")(x_flatten) # (?,2)
    
    logging.info("y.shape:{}\n".format(str(y.shape)))
    
    model=Model([x_input],outputs=[y])
    model.summary()
    
    return model

# 定义模型超参数

filters=128
max_sequence_length=max_document_length
max_token_num=len(vocab_dict)
embedding_dim=ndims
num_classes=3
embedding_matrix=embedding_matrix
batch_size=64
epochs=2
    
# 获得模型
model=textcnn(filters,max_sequence_length,max_token_num,embedding_dim,num_classes,embedding_matrix=embedding_matrix)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# 模型训练
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))

# 训练得分和准确度

score,acc=model.evaluate(x_test,y_test,batch_size=batch_size)

print("#---------------------------------------------------#")
print("预测得分:{}".format(score))
print("预测准确率:{}".format(acc))
print("#---------------------------------------------------#")
print("\n")

# 模型预测

predictions=model.predict(x_test)

print("#---------------------------------------------------#")
print("测试集的预测结果，对每个类有一个得分/概率，取值大对应的类别")
print(predictions)
print("#---------------------------------------------------#")
print("\n")

# 模型预测类别

predict_class=model.predict_classes(x_test)

print("#---------------------------------------------------#")
print("测试集的预测类别")
print(predict_class)
print("#---------------------------------------------------#")
print("\n")

# 模型保存

model.save(r"C:\Users\RY\Desktop\sentiment_analysis_textcnn.h5")

print("#---------------------------------------------------#")
print("保存模型")
print("#---------------------------------------------------#")
print("\n")

# 模型总结

print("#---------------------------------------------------#")
print("输出模型总结")
print(model.summary())
print("#---------------------------------------------------#")
print("\n")

# 模型的配置文件

config=model.get_config()

print("#---------------------------------------------------#")
print("输出模型配置信息")
print(config)
print("#---------------------------------------------------#")
print("\n")

# 模型的画图和图片保存

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import plot_model

plot_model(model,to_file="model.jpg",show_shapes=True)
lena=mpimg.imread("model.jpg")
lena.shape
plt.imshow(lena)
plt.axis("off")
plt.show()