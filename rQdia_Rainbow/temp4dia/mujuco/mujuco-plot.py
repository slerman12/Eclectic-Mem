import json
import random
from pathlib import Path
import pandas
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

data=[]
for file in Path(__file__).parent.glob('*.csv'):
    data.append((file.stem,pd.read_csv(file)))
methods=['rQdia-sac','rQdia-dqr']
palette= random.sample(px.colors.qualitative.G10,len(methods)+1)
method_color = {method:palette[i] for i,method in enumerate(methods)}
groud_legend={method:True for method in methods}

exp_num = len(data)
col = 3
row = exp_num // col
fig = make_subplots(rows=row, cols=col,subplot_titles = [' ' for i in range(row*col)])
for idx,(exp_name,exp_data) in enumerate(data):
    r_i = idx//col
    c_i = idx%col
    x = exp_data['step']
    for method in methods:
        fig.add_trace(
            go.Scatter(x=x, y=exp_data[method], name=method,mode='lines+markers',
                       showlegend=groud_legend[method],
                       marker={'color':method_color[method]}),
            row=r_i+1, col=c_i + 1
        )
        groud_legend[method]=False
    fig.layout.annotations[r_i * col + c_i]['text'] = exp_name
fig.show()
print()