import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = pd.period_range(pd.datetime.now(), periods=200, freq='d')
x = x.to_timestamp().to_pydatetime()
# 产生三组，每组 200 个随机常态分布元素
y = np.random.randn(200, 3).cumsum(0)
plt.plot(x, y)

# 设定标签
plots = plt.plot(x, y)
plt.legend(plots, ('A', 'F', 'G'),
           loc='best', framealpha=0.5, prop={'size': 'large', 'family': 'monospace'})

# 标题与轴标签
plt.title('Trends')
plt.xlabel('Date')
plt.ylabel('Sum')
plt.figtext(0.995, 0.01, 'CopyRight', ha='right', va='bottom')
plt.tight_layout() # 避免被图表元素被盖住
plt.grid(True) # 使用格子
plt.plot(x, y)
plt.show()