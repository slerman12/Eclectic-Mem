import json
import time
from collections import defaultdict

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from pathlib import Path
# method ='curl-rainbow'
method = 'rQdia_Rainbow_kld_results_seed_3'
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

def extract_name(folder):
    exp_name=folder.name.split('-')[0]
    exp_name=exp_name.replace('kld','')
    exp_name = exp_name[:-1]
    return exp_name
for sub_folder in [f for f in folder.iterdir() if f.is_dir()]:
    htmls = [j.name for j in sub_folder.glob('*.html')]
    exp_name = extract_name(sub_folder)
    if len(htmls)!=2:
        print(f"missing exp: {method}-{exp_name}")
        continue
    for html in htmls:
        log_name=html.split('.')[0]
        if log_name=='Q':continue
        driver.get('file:///' + str(sub_folder / html))
        time.sleep(1.2)
        text=driver.find_element_by_xpath('/html/body/div/script[3]').get_attribute('innerHTML')
        temp = text.split(',                        {"template"')[0]
        data=json.loads(temp[temp.index('['):])
        if len(data[0]['x'])!=10:
            print(f"missing steps: {method}-{exp_name}")
            break
        else:
            log[f"{method}-{exp_name}-{log_name}"].append(data)
with open(Path(__file__).parent/f'{method}.json','w') as f:
    json.dump(log,f)
driver.close()
print()