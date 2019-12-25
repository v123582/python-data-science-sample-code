import requests
# 引入函式库

url = 'https://www.aicoin.cn/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
}
# 将标头包装成字典型态
r = requests.get(url, headers=headers)
# 发送请求到目标网址，加上标头
print(r.text)

'''
<!DOCTYPE html><html><head><meta charSet="utf-8" class="next-head"/><meta name="viewport" content="width=device-width, initial-scale=1 " class="jsx-2966286125 next-head"/><link rel="canonical" href="https://www.aicoin.net.cn?lang=zh" class="jsx-2966286125 next-head"/ ><title class="jsx-2966286125 next-head">AICoin - 为价值· 更高效</title> ... </html>
'''