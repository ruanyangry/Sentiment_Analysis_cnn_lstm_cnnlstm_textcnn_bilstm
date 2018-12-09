# _*_ coding:utf-8 _*_

'''
@Author: Ruan Yang
@Date: 2018.12.9
@Purpose: 文本情感分析(positive,negative,neutral)
@Reference: https://github.com/Edward1Chou/SentimentAnalysis
@算法：CNN
@需要有事先标准好的数据集
@positive: [1,0,0]
@neutral: [0,1,0]
@negative:[0,0,1]
'''

import codecs
import jieba

datapaths=r"C:\Users\RY\Desktop\SentimentAnalysis-master\data\\"

positive_data=[]
y_positive=[]
neutral_data=[]
y_neutral=[]
negative_data=[]
y_negative=[]

print("#------------------------------------------------------#")
print("加载数据集")
with codecs.open(datapaths+"pos.csv","r","utf-8") as f1,\
     codecs.open(datapaths+"neutral.csv","r","utf-8") as f2,\
     codecs.open(datapaths+"neg.csv","r","utf-8") as f3:
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

print("#---------------------------------------------------#")
print("Build model .................")
print("NN structure .......")
print("Embedding layer --- CNN layer  --- Dense layer --- Dense layer")
print("#---------------------------------------------------#")
print("\n")

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D

batch_size=64
max_sentence_length=200
embedding_dims=ndims
filters = 250
kernel_size = 3
hidden_dims = 250
dropout=0.2
recurrent_dropout=0.2
num_classes=3
epochs=2

# 定义网络结构
model = Sequential()
model.add(Embedding(len(vocab_dict),
                    embedding_dims,
                    weights=[embedding_matrix],
                    input_length=max_sentence_length,
                    trainable=False))
model.add(Dropout(dropout))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

# 模型编译

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

print("#---------------------------------------------------#")
print("Train ....................")
print("#---------------------------------------------------#")
print("\n")

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

model.save(r"C:\Users\RY\Desktop\sentiment_analysis_lstm.h5")

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
