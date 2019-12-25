import json
import requests
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
#一次取回20笔答题者记录
rep = requests.get("https://www.zhihu.com/api/v4/questions/20279570/answers?&limit=20&offset=0",headers=headers)

#设定爬虫解析编码
rep.encoding = rep.apparent_encoding

content = json.loads(html)
for row in content['data']:
    print(row['author']['name']