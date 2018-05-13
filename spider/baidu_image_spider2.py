import urllib.parse
import urllib.request
import time
import json
import requests
import os
import re
from pyquery import PyQuery as pq
from bloom_filter import BloomFilter

# BloomFilter去重
filter = BloomFilter(filename='baidu_bloom', max_elements=10000, error_rate=0.001)
delay = 2 # 爬取间隔

# baidu图片url公共格式
url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&cg=star&rn=30&gsm=1e&1524217565470=&word="
key1 = "&queryWord=" # 搜索字符 
key2 = "&pn=" # 页码,从0开始每次+30
words = ["章子怡"]
header = {
    'Host': 'image.baidu.com',
    'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gbk&word=%CC%C0%CE%A8&fr=ala&ala=1&alatpl=star&pos=0&hs=2&xthttps=111111',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/65.0.3325.181 Chrome/65.0.3325.181 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}
images = []

def is_url_legal(url):
    if type(url) is not str:
        return False
    if re.match(r'^https?:/{2}\w.+$', url):
        print(url)
        return True  
    return False

def parse_page(json):
    if json:
        items = json.get('data')
        for item in items:
            image = item.get('hoverURL')
            is_legal = is_url_legal(image)
            # 判断图片类型和是否在filter中
            if is_legal and image not in filter:
                images.append(image)

def download_images():
    for url in images:
        try: 
            image = requests.get(url, timeout=2) # 超时判断2秒
        except requests.exceptions.ConnectionError:
            continue
        except requests.exceptions.MissingSchema:
            continue
        filename = get_filename()
        filename = 'images/' + filename + '.jpg'
        with open(filename, 'wb') as f:
            f.write(image.content)
        filter.add(url)
        time.sleep(delay)  # 限制下载速度

def get_filename():
    localtime = time.localtime(time.time())
    filename = "{0}-{1}-{2}-{3}-{4}-{5}".\
        format(localtime.tm_year, localtime.tm_mon, localtime.tm_mday,\
                localtime.tm_hour, localtime.tm_min, localtime.tm_sec)
    return filename

if not os.path.exists('images'):
    os.makedirs('images')
for word in words:
    word = urllib.parse.quote(word)
    url1 = url + word + key1 + word # 关键词已经确定的公共url
    for i in range(0, 300, 30): # 爬取10页
        url2 = url1 + key2 + str(i)
        try:
            response = requests.get(url2, headers=header)
            if response.status_code == 200 and response is not None:
                try:
                    content = response.json()
                except json.JSONDecodeError:
                    continue
                parse_page(content)
        except requests.ConnectionError as e:
            continue
        except requests.exceptions.MissingSchema:
            continue
        time.sleep(delay)  # 限制爬取速度

download_images() # images列表构造完成，开始下载图片