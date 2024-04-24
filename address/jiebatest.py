import jieba
words = jieba.cut("杭州市长庆街道新华坊社区23幢2楼办公室")
print("/".join(words))
# print(len(words))
