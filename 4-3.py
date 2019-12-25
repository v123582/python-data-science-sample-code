from bs4 import BeautifulSoup
import requests

headers = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
}
rep = requests.get("https://newhouse.fang.com/house/saledate/201908.htm", headers = headers)

#显示网页使用的编码方式，设定爬虫解析编码
print(rep.apparent_encoding)
rep.encoding = rep.apparent_encoding

# 解析网页 html
html = rep.text
soup = BeautifulSoup(html, 'html.parser')

#取得储存房子资讯的div列表
houses = soup.find_all("div",class_="nlcd_name")

#解析每个div的房子名称
for house in houses:
    print(house.find_all("a")[0].text)

# 中国中铁·诺德春风和院
# 万橡悦府
# ...