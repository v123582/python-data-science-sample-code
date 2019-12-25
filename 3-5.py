import requests
# 引入函式库
r = requests.get('https://www.dcard.tw/_api/forums/job/posts?popular=true')
# 想要爬数据的目标网址
response = r.text
# 模拟发送请求的动作

import json
data = json.loads(response)

for d in data:
    print(d['title'])