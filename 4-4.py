from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep
import requests

browser = webdriver.Chrome()
browser.get('https://www.zhihu.com/question/20279570')
# 向下卷动一次，载入更多数据
js = "var action=document.documentElement.scrollTop=2000"
browser.execute_script(js)
username = browser.find_elements_by_xpath("//div[@class='AuthorInfo-head']")
for name in username:
    print(name.text)