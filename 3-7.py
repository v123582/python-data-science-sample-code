import requests
# 引入函式库

id = '1816670'
appid = 'a436ac0a93354f5fa133bae3bdac76d0'
url = "https://api.openweathermap.org/data/2.5/forecast?id=%s&appid=%s" % (id, appid)
r = requests.get(url)
# 想要爬数据的目标网址
response = r.text
# 模拟发送请求的动作

import json
data = json.loads(response)

for d in data['list']:
    print(d['weather'][0]['description'])