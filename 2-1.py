# 编码宣告
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding: utf-8

# 第三方模块引用
import this
from keyword import kwlist

# 变数与赋值
keywords = kwlist

# 流程控制
if len(keywords):
    pirnt(keywords)
    # ['False', 'None', ...]
for keyword in keywords:
    print(keyword)
    # 'False'
    # 'None'
    # ...

input("Input Something...")
# Input Something..._
print("Hello World")
# Hello World