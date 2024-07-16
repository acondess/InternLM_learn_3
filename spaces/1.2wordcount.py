text = """
Got this panda plush toy for my daughter's birthday,
who loves it and takes it everywhere. It's soft and
super cute, and its face has a friendly look. It's
a bit small for what I paid though. I think there
might be other options that are bigger for the
same price. It arrived a day earlier than expected,
so I got to play with it myself before I gave it
to her.
"""

def wordcount(text):
    """
    统计给定英文字符串中每个单词的出现次数。

    参数:
        text (str): 输入的英文字符串。

    返回:
        dict: 字典类型的结果，其中键是单词，值是该单词的出现次数。
    """

    # 将所有字符转换为小写，以便统一处理
    text = text.lower()

    # 定义一个字符串，包含所有需要被移除的标点符号
    punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # 移除所有标点符号
    for char in text:
        if char in punctuation:
            text = text.replace(char, "")

    # 根据空格分割字符串，得到单词列表
    words = text.split()

    # 初始化一个字典来存储单词及其出现次数
    word_count = {}

    # 遍历单词列表，统计每个单词的出现次数
    for word in words:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1

    # 返回统计结果
    return word_count

print(wordcount(text))