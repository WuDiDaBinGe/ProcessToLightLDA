#### 1.数据集格式

![img.png](img.png)

每个文件夹下有很多txt数据，txt数据中包含多篇新闻内容，每一篇新闻用空行隔开

#### 2. 数据集处理步骤

对数据集处理转成LightLDA需要的格式需要以下几步：

1. 将数据集合并为一整个txt文件，方便处理。该txt包含了所有收集到的文章，格式为每一行为一篇文章，同时对新闻数据进行清洗
2. 将合并后的txt的每一行(每篇文章)使用jieba进行分词，得到分词后的数据集
3. 利用分词后的数据集得到数据集的词表
4. 结合词表和分词后的数据集形成该数据集的UCI格式

##### 2.1 合并数据集为txt

`clean_merge_all_txt_in_dirs.py` 为该步骤的文件

先读取文件下所有txt路径然后利用多线程处理所有的txt文件，在该文件中定义每个文件夹路径如下。
使用列表存放每个文件夹下的文件路径并使用多进程清洗文件。
```python
base_dir = '/home/lcl/LightLDA/military_20g/20g/'
dir_all_list = [
    base_dir + '5k', base_dir + '10k', base_dir + '15k', base_dir + '20k', base_dir + '25k', base_dir + '30k',
    base_dir + '35k', base_dir + 'new'
]
```
在获取了对每一篇文章做了清洗，去除原创、图片记号，网址以及特殊符号，正则如下：
```python
regix_nosiy_seg = [
    # 去除原创图片信息
    re.compile(r"[\[【]*(.{1,5}图|.{1,5}原创)[\]】]+"),
    # 去除网址
    re.compile(
        r"[https:\/\/|http:\/\/]*(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)"),
    # 特殊字符
    re.compile('[#$%&*@★、…【】]+')
]
```
最终得到了存放所有数据的txt文件`all_docs.txt`，其中每一行为清洗过的文章。
在清洗合并文件之后我们需要使用`tokrnizer_vocab_uci_mp.py`中`pipeline_preprocess_raw`函数中定义了三步，操作函数，最终将`all_docs.txt`转化为UCI格式。


`tokrnizer_vocab_uci_mp.py`文件中定义了下面的三步的函数:
```python
def pipeline_preprocess_raw(num_worker):
    # all_docs.txt 路径 未分词的
    raw_data_url = '/home/lcl/LightLDA/military_20g/20g/all_docs.txt'
    # 分词后的all_docs.txt （英文忽略）
    tokenized_data_url = '/home/lcl/LightLDA/military_20g/20g/all_docs_tokenized.txt'
    # 得到的词表文件
    vocab_url = '/home/lcl/LightLDA/military_20g/20g/vocab.military20.txt'
    # 最后转化得到的UCI格式文件
    uci_url = '/home/lcl/LightLDA/military_20g/20g/docword.military20_new.txt'
    # 1. 得到分词后的数据集(英文可忽略)
    get_tokenized_dataset_mp(num_worker, raw_url=raw_data_url, token_out_url=tokenized_data_url)
    # 2. 得到词表
    build_vocab_dataset_mp(num_worker,tokenized_url=tokenized_data_url,vocab_url=vocab_url)
    # 3. 得到UCI数据格式数据
    # 6086741
    get_UCI_dataset_single(tokenized_url=tokenized_data_url, uci_url=uci_url, vocab_url=vocab_url)
```
在执行以下三步中的一步时，建议注释掉其他两个函数。
##### 2.2分词

使用jieba库对每一行的中文新闻新闻进行分词，分词后各个单词的token用`\  `+一个空格隔开
若使用英文数据直接使用空格进行分词。需要修改分词的代码。
```python
# 1. 得到分词后的数据集(英文可忽略)
    get_tokenized_dataset_mp(num_worker, raw_url=raw_data_url, token_out_url=tokenized_data_url)
```
##### 2.3 得到词表

遍历分词后的文章，得到词表，并过滤掉出现次数的很少的单词，以及去掉停用词。

词表文件vocab.txt每一行为一个单词,行号为单词的ID号。
```python
# 2. 得到词表
    build_vocab_dataset_mp(num_worker,tokenized_url=tokenized_data_url,vocab_url=vocab_url)
```
##### 2.4 转成UCI格式

使用`get_UCI_dataset_single`函数，将词表文件和分词后的数据集文件，得到UCI格式的数据集

UCI格式如下：

注意文章序号和词表序号 均是从1 开始计数的

```shell
docIndex(文章序号) wordId(词表序号) frequent(词在文章中出现次数)
1 1 3
1 3 4
1 100 2
...
```



