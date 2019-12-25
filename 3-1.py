# 下载档案

import urllib.request # 载入 urllib 函式库
res = "http://opendata.hccg.gov.tw/dataset/432257df-491f-4875-8b56-dd814aee5d7b/resource/de014c8b-9b75-4152-9fc6-f0d499cefbe4/download/20150305140446074.csv"
urllib.request.urlretrieve(res, '51.csv')
# 下载档案到本地，存成档名 test.csv

# 读档案
fh = open("51.csv", newline='')
# 开启档案
f = fh.read()
# 将档案内容读取到字串 f
fh.close()
# 关闭档案释放资源

# 解析档案内容
import csv # 载入 csv 函式库
reader = csv.reader(f.split('\n'), delimiter=',')
# 利用 csv.reader 存取字串，转成列表
for row in reader:
    print(row)
    # 逐行将数据印出