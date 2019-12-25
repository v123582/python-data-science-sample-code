import urllib.request
import zipfile
res = "http://ftp.eea.europa.eu/www/AirBase_v8/AirBase_v8_xmldata.zip"
urllib.request.urlretrieve(res,"AirBase_v8_xmldata.zip")
f = zipfile.ZipFile('AirBase_v8_xmldata.zip')
f.extractall("./data/")

# 读档案

fh = open("./data/AD_meta.xml", "r")
xml = fh.read()
fh.close()

# 解析档案内容

import xmltodict
d = dict(xmltodict.parse(xml))
country = d['airbase']['country']
stations = d['airbase']['country']['station']
print(country['country_name'])
print(stations[0]['station_info']['station_name'])
print(stations[0]['station_info']['station_latitude_decimal_degrees'])
print(stations[0]['station_info']['station_longitude_decimal_degrees'])
print(stations[1]['station_info']['station_name'])
print(stations[1]['station_info']['station_latitude_decimal_degrees'])
print(stations[1]['station_info']['station_longitude_decimal_degrees'])