from bs4 import BeautifulSoup
import requests

#查询世界大学学术排名网站
rep = requests.get("http://www.zuihaodaxue.cn/ARWU2019.html")

# 显示网页使用的编码方式，设定爬虫解析编码
print(rep.apparent_encoding)
rep.encoding = rep.apparent_encoding

# 解析网页 html
html = rep.text
soup = BeautifulSoup(html, 'html.parser')

# 使用CSS Selectors 定位数据在html中的位置
university = soup.select("tbody tr td.align-left a")
for row in university:
    print(row.text)

# 哈佛大学
# 斯坦福大学
# 剑桥大学
# ...