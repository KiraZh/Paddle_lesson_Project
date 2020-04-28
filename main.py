from __future__ import print_function
import json
import time
import re  # 正则匹配
import jieba  # 中文分词
import matplotlib.pyplot as plt
import numpy as np
import paddlehub as hub
import requests
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator  # 绘制词云模块


# 请求爱奇艺评论接口，返回response信息
def getMovieinfo(url):
    """
    请求爱奇艺评论接口，返回response信息
    参数  url: 评论的url
    :return: response信息
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/80.0.3987.163 Safari/537.36 Edg/80.0.361.111 '
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(e)
    return None


# 解析json数据，获取评论
def saveMovieInfoToFile(lastId, arr):
    """
    解析json数据，获取评论
    参数  lastId:最后一条评论ID  arr:存放文本的list
    :return: 新的lastId
    """
    url = 'https://sns-comment.iqiyi.com/v3/comment/get_comments.action?agent_type=118&agent_version=9.11.5' \
          '&authcookie=null&business_type=17&content_id=14992525500&hot_size=0&last_id='
    url += str(lastId)
    text = getMovieinfo(url)
    a = json.loads(text)
    comments = a['data']['comments']
    for comment in comments:
        if 'content' in comment.keys():
            arr.append(comment['content'])
            lastId = comment['id']
    # print(lastId)
    return lastId


# 去除文本中特殊字符
def clear_special_char(content):
    '''
    正则处理特殊字符
    参数 content:原文本
    return: 清除后的文本
    '''
    s = re.sub(r'\n', ' ', content)
    s = re.sub(r'\[[^()]*\]', '', s)
    s = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\d,\,.，。!！*%￥$？?]', '', s)
    s = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", s)
    s = re.sub(
        '[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+',
        '', s)
    s = re.sub(r'[\U00010000-\U0010ffff]', '', s)
    return s


def fenci(text):
    """
    利用jieba进行分词
    参数 text:需要分词的句子或文本
    return：分词结果
    """
    jieba.load_userdict("data/userdict.txt")
    seg_list = jieba.cut(text)
    # print(", ".join(seg_list))
    return seg_list


def stopwordslist(file_path):
    """
    创建停用词表
    参数 file_path:停用词文本路径
    return：停用词list
    """
    stopwords = [line.strip() for line in open(file_path, encoding='UTF-8').readlines()]
    return stopwords


def movestopwords(sentence, stopwords, counts):
    """
    去除停用词,统计词频
    参数 sentence:分词列表 stopwords:停用词list counts: 词频统计结果
    return：None
    """
    out = []
    for word in sentence:
        if word not in stopwords:
            if len(word) != 1:
                counts[word] = counts.get(word, 0) + 1
    return None


def drawcounts(counts, num):
    """
    绘制词频统计表
    参数 counts: 词频统计结果 num:绘制topN
    return：none
    """
    x = []
    y = []
    ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for thing in ordered[:num]:
        x.append(thing[0])
        y.append(thing[1])
    plt.rcParams['font.sans-serif'] = ['PingFang HK']  # 指定默认字体,MacOS没有黑体所以会报错
    plt.bar(x, y)
    plt.show()
    return


def drawcloud(word_f):
    """
    根据词频绘制词云图
    参数 word_f:统计出的词频结果
    return：none
    """
    background = np.array(Image.open('data/1.png'))
    w = WordCloud(mask=background,
                  font_path='/System/Library/Fonts/PingFang.ttc',
                  background_color='white',
                  max_words=150,
                  min_font_size=10,
                  max_font_size=100,
                  relative_scaling=0.3)
    w.fit_words(word_f)
    # 匹配颜色
    image_colors = ImageColorGenerator(background)
    w.recolor(color_func=image_colors)
    w.to_file('wordcloud.png')


def text_detection():
    """
    使用hub对评论进行内容分析
    return：分析结果
    """
    porn_detection_lstm = hub.Module(name="porn_detection_lstm")
    f = open('result.txt', 'r', encoding='UTF-8')
    arr = []
    for line in f:
        if (len(line.strip())) < 1:
            continue
        else:
            arr.append(line)
    f.close()
    input_dict = {"text": arr}
    results = porn_detection_lstm.detection(data=input_dict, use_gpu=True, batch_size=1)
    for index, item in enumerate(results):
        if item['porn_detection_key'] == 'porn':
            print(item['text'], ':', item['porn_probs'])


# 尝试使用senta模型进行情感分析：
# 可能由于语句较短，将负面概率调到0.9以上较为合理
def senta():
    senta = hub.Module(name='senta_lstm')
    f = open('result.txt', 'r', encoding='UTF-8')
    arr = []
    for line in f:
        if (len(line.strip())) < 1:
            continue
        else:
            arr.append(line)
    f.close()
    input_dict = {"text": arr}
    results = senta.sentiment_classify(data=input_dict, use_gpu=True, batch_size=1)
    # print(results)
    for index, item in enumerate(results):
        if item['negative_probs'] > 0.9:
            print(item['text'])


if __name__ == "__main__":
    # 爬取条数 = 25*页数
    pagesNum = 120
    # 展示的词数
    num = 10
    arr = []
    lastId = '0'
    stopwords = stopwordslist('data/stopwords.txt')
    counts = {}
    with open('result.txt', 'a', encoding='UTF-8') as f:
        for i in range(1, pagesNum):
            lastId = saveMovieInfoToFile(lastId, arr)
            time.sleep(0.5)
        for content in arr:
            content = clear_special_char(content)
            f.write(content + '\n')
    print('共搜集信息{}条'.format(len(arr)))
    f = open('result.txt', 'r', encoding='UTF-8')
    for line in f:
        words = fenci(line)
        movestopwords(words, stopwords, counts)
    print(counts)
    drawcounts(counts, num)
    drawcloud(counts)
    f.close()
    # text_detection()
    # senta()
