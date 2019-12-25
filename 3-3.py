# 下载档案

import urllib.request
import zipfile

dataid = "F-D0047-093"
authorizationkey = "rdec-key-123-45678-011121314"
res = "http://opendata.cwb.gov.tw/govdownload?dataid=%s&authorizationkey=%s" % (dataid, authorizationkey)

urllib.request.urlretrieve(res,"F-D0047-093.zip")
f = zipfile.ZipFile('F-D0047-093.zip')
f.extractall()

# 读档案

fh = open("64_72hr_CH.xml", "r")
xml = fh.read()
fh.close()

# 解析档案内容

import xmltodict
d = dict(xmltodict.parse(xml))

locations = d['cwbopendata']['dataset']['locations']['location']
location = locations[0]
print(location['locationName'])
print(location['weatherElement'][0]['time'][0]['dataTime'])
print(location['weatherElement'][0]['time'][0]['elementValue'])
print(location['weatherElement'][0]['time'][1]['dataTime'])
print(location['weatherElement'][0]['time'][1]['elementValue'])