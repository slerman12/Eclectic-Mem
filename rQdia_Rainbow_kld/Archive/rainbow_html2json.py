import json
import time
from collections import defaultdict

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from pathlib import Path
method ='latestcurlrainbow2'

folder = Path('/Users/samlerman/Desktop')/method
def trans2xy(text):
    x,y = text.replace('translate(','').replace(')','').split(',')
    return float(x),float(y)

def element2tick(element):
    ticks=element.find_elements_by_xpath('./*')
    temp = {}
    for tick in ticks:
        text_ele = tick.find_element_by_xpath("./*")
        x,y = trans2xy(text_ele.get_attribute('transform'))
        label = text_ele.get_attribute('data-unformatted')
        coor = x if y==0 else y
        temp[coor]=float(label)

    return list(temp.items())
driver = webdriver.Chrome(options=Options(),
                          executable_path=r"/Users/samlerman/Code/chromedriver")
log = defaultdict(list)
for sub_folder in [f for f in folder.iterdir() if f.is_dir()]:
    name = sub_folder.name.split('-')[0]
    exp=name[:-2]
    htmls = [j.name for j in sub_folder.glob('*.html')]
    if len(htmls)!=2:
        print(f"missing exp: {method}-{exp}")
        continue
    for html in htmls:
        log_name=html.split('.')[0]
        if log_name=='Q':continue
        driver.get('file:///' + str(sub_folder / html))
        time.sleep(1.2)
        text=driver.find_element_by_xpath('/html/body/div/script[3]').get_attribute('innerHTML')
        temp = text.split(',                        {"template"')[0]
        data=json.loads(temp[temp.index('['):])
        # if len(data[0]['x'])!=10:
        #     print(f"missing steps: {method}-{exp}")
        #     break
        # else:
        log[f"{method}-{exp}-{log_name}"].append(data)
with open(Path(__file__).parent/f'{method}.json','w') as f:
    json.dump(log,f)
print()
    # 啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
    # for html in htmls:
    #     driver.get('file:///' + str(sub_folder / html))
    #     time.sleep(3)
    #     # because it use top left so there exists some offset
    #     # x_tick = element2tick(driver.find_element_by_xpath('//*[@class="xaxislayer-above"]'))
    #
    #     y_tick = element2tick(driver.find_element_by_xpath('//*[@class="yaxislayer-above"]'))
    #     step =(y_tick[0][0]-y_tick[1][0])/(y_tick[0][1]-y_tick[1][1])
    #     min_y = min(y_tick, key = lambda t: t[1])
    #     get_y=lambda y: (y - min_y[0])/step + min_y[1]
    #     lines= defaultdict(list)
    #     for trace in driver.find_elements_by_xpath('//*[contains(@class,"trace scatter")]'):
    #         points=trace.find_elements_by_xpath(".//*[@class='point']")
    #         for point in points:
    #             x,y = trans2xy(point.get_attribute('transform'))
    #             lines[x].append(y)
    #     x_tick = {k:i*1000 for i,k in enumerate(lines,1)}
    #     print(exp,seed)