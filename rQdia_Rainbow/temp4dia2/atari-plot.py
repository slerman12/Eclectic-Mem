import json
import random
from collections import defaultdict

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

plots=defaultdict(dict)

def name_f(name):
    exp_name = name.split('-')[-2]
    return exp_name[1:] if exp_name[0].isnumeric() else exp_name[:-1]
name_f_kld = lambda  name: name.split('-')[-2][:-2]
name_f_curl = lambda name: name.split('-')[-2].replace('curlr','')[:-1]


def extrac_data(method,name_f):
    with open(f'{method}-rainbow.json', 'r') as f:
        data = json.load(f)
    temps = defaultdict(list)
    for name in [k for k in data.keys() if 'Reward' in k]:
        exp_data=[[i for i in v if i['name'] == 'Mean'] for v in data[name]][0]
        exp_name=name_f(name)
        x = [sum(v)/len(v) for v in zip(*[i['x'] for i in exp_data])]
        y = [sum(v)/len(v) for v in zip(*[i['y'] for i in exp_data])]
        temps[exp_name].append((x,y))
    for exp_name, exp_seeds in temps.items():
        x = [sum(value) / len(value) for value in zip(*[i[0] for i in exp_seeds])]
        y = [sum(value) / len(value) for value in zip(*[i[1] for i in exp_seeds])]
        plots[exp_name][method]=(x,y)
meta={
    'rQdia':name_f,
      'curl':name_f_curl,'rQdia-kld':name_f_kld
}

for k,f in meta.items():
    extrac_data(k,f)


# pre define parameter for figure
exp_num = len(plots)
col = 4
row = exp_num // col + 1
fig = make_subplots(rows=row, cols=col,subplot_titles = [' ' for i in range(row*col)])
methods = set(sum([list(i.keys()) for i in list(plots.values())],[]))

palette= random.sample(px.colors.qualitative.G10,len(methods)+1)
method_color = {method:palette[i] for i,method in enumerate(methods)}
groud_legend={method:True for method in methods}
# start plot


for idx,(exp_name,methods) in enumerate(plots.items()):
    r_i = idx // col
    c_i = idx % col
    for method,data in methods.items():
        fig.add_trace(
            go.Scatter(x=data[0], y=data[1], name=method,mode='lines+markers',
                       showlegend=groud_legend[method],
                       marker={'color':method_color[method]}),
            row=r_i+1, col=c_i + 1
        )
        groud_legend[method]=False
    fig.layout.annotations[r_i * col + c_i]['text'] = exp_name
fig.show()


# seed_num = max([len(v) for v in [data[i] for i in name]])
# palette= random.sample(px.colors.qualitative.G10,seed_num+1)
# result=[]
# for r in range(row):
#     for c in range(col):
#         try:
#             exp_name = name[r * col + c]
#             exp_data = data[exp_name]
#             mean_score=[]
#             for idx, exp_seeds in enumerate(exp_data.items()):
#                 for line in [i for i in exp_seeds if i['name'] == 'Mean']:
#                     fig.add_trace(
#                         go.Scatter(x=line['x'], y=line['y'], name=seed,marker={'color':palette[int(seed)-1]}),
#                         row=r+1, col=c+1
#                     )
#                     mean_score.append(line['y'])
#             mean_score = [sum(i)/len(i) for i in zip(*mean_score)]
#             fig.add_trace(
#                 go.Scatter(x=line['x'], y=mean_score, name='mean', marker={'color': palette[seed_num]}),
#                 row=r + 1, col=c + 1)
#             fig.layout.annotations[r * col + c]['text'] = exp_name
#             result.append(f"{exp_name.split('-')[-2]:<20} |{max(mean_score):.2f}")
#         except:
#             continue
# fig.show()
# for i in sorted(result):
#     print(i)
# # fig.write_html('test.html')
# print()
