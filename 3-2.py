# 下载档案

import urllib.request # 载入 urllib 函式库
res = "https://gist.githubusercontent.com/vvtommy/a1bee4cd9e65e6d57b84f35f4e4dd5e1/raw/a0aa736eeec30134ca2a2367c55c115be23d5dd1/%25E4%25B8%25AD%25E5%259B%25BD%25E5%258C%25BA%25E5%258F%25B7.csv"
urllib.request.urlretrieve(res, '中国区号.csv')
# 下载档案到本地，存成档名 test.csv

# 读档案
fh = open("中国区号.csv", newline='')
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