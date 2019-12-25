import requests
# 引入函式库
r = requests.get('https://api.carbonintensity.org.uk/intensity/date/2018-04-12')
# 想要爬数据的目标网址
response = r.text
# 模拟发送请求的动作

import json
data = json.loads(response)

for d in data['data']:
    print(d)