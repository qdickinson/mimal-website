import sklearn
import sklearn.metrics
import plotly.express as px
import numpy as np
import pandas as pd
import pickle
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
import scipy.stats
from dash.dependencies import Input, Output, State# Load Data
df = px.data.tips()
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output# Load Data
import pickle
import shap
import matplotlib as mpl

import plotly.tools as tls
import dash_core_components as dcc
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost
import shap
import matplotlib
import plotly.graph_objs as go
import scipy.stats

import matplotlib.pyplot as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

import json
import dash
import dash_cytoscape as cyto
import dash_html_components as html
from jupyter_dash import JupyterDash 


#from dash import Dash, dcc, html, callback, Input, Output
from jupyter_dash import JupyterDash
#import dash_core_components as dcc
#import dash_html_components as html
#from dash.dependencies import Input, Output, State
import warnings
import base64
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
warnings.filterwarnings("ignore")
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost

import flask
import os
from flask import Flask, send_from_directory
import base64
import os
from urllib.parse import quote as urlquote

import random
import string
import umap.umap_ as umap
from sklearn.cluster import OPTICS
import networkx
data = json.load(open('clustering_final.cyjs'))

cdict1 = {
    'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
            (1.0, 0.9607843137254902, 0.9607843137254902)),

    'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
              (1.0, 0.15294117647058825, 0.15294117647058825)),

    'blue': ((0.0, 0.8980392156862745, 0.8980392156862745),
             (1.0, 0.3411764705882353, 0.3411764705882353)),

    'alpha': ((0.0, 1, 1),
              (0.5, 1, 1),
              (1.0, 1, 1))
}  # #1E88E5 -> #ff0052
red_blue = LinearSegmentedColormap('RedBlue', cdict1)

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

red_blue = matplotlib_to_plotly(red_blue, 255)

#force_plot = shap.plots.force(shap_values[20],matplotlib=False)

# force_plot = shap.force_plot(expected[met_all.columns[20]], shap_all[met_all.columns[20]][20,:], pro_all.iloc[20,:], 
#                 show=True, 
#                 matplotlib=False,
#                text_rotation=90)

pro_all = pd.read_pickle('pro_all.pickle')
met_all = pd.read_pickle('met_all.pickle')
#shap_all = pd.read_pickle('shap_pmet_all.pickle')

pro_ferm = pd.read_pickle('pro_ferm.pickle')
met_ferm = pd.read_pickle('met_ferm.pickle')
#shap_ferm = pd.read_pickle('shap_ferm.pickle')

pro_resp = pd.read_pickle('pro_resp.pickle')
met_resp = pd.read_pickle('met_resp.pickle')
#shap_resp = pd.read_pickle('shap_resp.pickle')

expected = pd.read_pickle('expected_values_all.pickle')
columns = list(pro_all.columns)
metabolites = list(met_all.columns)
proteins = list(pro_all.columns)









UPLOAD_DIRECTORY = "/home/dickinsonq/mimal/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.umask(0)
    os.makedirs(UPLOAD_DIRECTORY, mode=0o777)


shap_html = pickle.load(open("figures/force/20-Citric acid.pickle", 'rb'))


shap_meanabs = pd.read_pickle('shap_mean_abs_pmet_all.pickle')
all_correlations = pd.read_pickle("metabolite_correlations.pickle")


#shap_dictall = pickle.load(open('shap_pmet_all.pickle', 'rb'))


pro_all = pd.read_pickle('pro_all.pickle')


met_all = pd.read_pickle('met_all.pickle')


pred =  pd.read_pickle('ETmet_pred_dict.pickle')

met_test =  pd.read_pickle('met_test.pickle')

keys = list(shap_meanabs.keys())

column_numbers = [x for x in range(met_test.shape[1])]  # list of columns' integer indices

column_numbers.remove(29) #removing column integer index 0
met_test = met_test.iloc[:, column_numbers]


temp = {}

for i in range(len(met_test.keys())):
    temp[list(met_test.keys())[i]] = pred[list(pred.keys())[i]]
pred_test = temp


all_labels = []
for value in list(pro_all.index):
    all_labels.append(value)
feature_names=list(pro_all.columns)


shap_values = pickle.load(open('SHAP/all/Citric acid.pickle', 'rb'))
plotly_fig = pickle.load(open('figures/summary/Citric acid.pickle', 'rb'))

expected = pd.read_pickle('expected_values_all.pickle')
columns = list(pro_all.columns)
metabolites = list(met_all.columns)
proteins = list(pro_all.columns)


def save_file(name, content, upload):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(upload, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files(upload):
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(upload):
        path = os.path.join(upload, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)





clusterstyle = [

         {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                #'background-color': 'mapData(__mclCluster, 1, 34, blue, red)'
                #'background-color': "setNodeColorMapping('score', colors=paletteColorBrewerSet3, mapping.type='d')"
            }
        },
        {
           'selector': '[__mclCluster=1]',
           'style': {
               'background-color': '#FF2D00'
           }
        },
        {
           'selector': '[__mclCluster=2]',
           'style': {
               'background-color': '#FDAA7D'
           }
        },
        {
           'selector': '[__mclCluster=3]',
           'style': {
               'background-color': '#F6901E'
           }
        },
        {
           'selector': '[__mclCluster=4]',
           'style': {
               'background-color': '#EBB124'
           }
        },
        {
           'selector': '[__mclCluster=5]',
           'style': {
               'background-color': '#DCCE61'
           }
        },
        {
           'selector': '[__mclCluster=6]',
           'style': {
               'background-color': '#BEC900'
           }
        },
        {
           'selector': '[__mclCluster=7]',
           'style': {
               'background-color': '#A1B463'
           }
        },
        {
           'selector': '[__mclCluster=8]',
           'style': {
               'background-color': '#609B0E'
           }
        },
        {
           'selector': '[__mclCluster=9]',
           'style': {
               'background-color': '#448119'
           }
        },
        {
           'selector': '[__mclCluster=10]',
           'style': {
               'background-color': '#366527'
           }
        },
        {
           'selector': '[__mclCluster=11]',
           'style': {
               'background-color': '#055001'
           }
        },
        {
           'selector': '[__mclCluster=12]',
           'style': {
               'background-color': '#426C47'
           }
        },
        {
           'selector': '[__mclCluster=13]',
           'style': {
               'background-color': '#09882E'
           }
        },
        {
           'selector': '[__mclCluster=14]',
           'style': {
               'background-color': '#26A260'
           }
        },
        {
           'selector': '[__mclCluster=15]',
           'style': {
               'background-color': '#3EBA8E'
           }
        },
        {
           'selector': '[__mclCluster=16]',
           'style': {
               'background-color': '#04CFAB'
           }
        },
        {
           'selector': '[__mclCluster=17]',
           'style': {
               'background-color': '#95E0E0'
           }
        },
        {
           'selector': '[__mclCluster=18]',
           'style': {
               'background-color': '#0AC6EE'
           }
        },
        {
           'selector': '[__mclCluster=19]',
           'style': {
               'background-color': '#45B9F8'
           }
        },
        {
           'selector': '[__mclCluster=20]',
           'style': {
               'background-color': '#499EFE'
           }
        },
        {
           'selector': '[__mclCluster=21]',
           'style': {
               'background-color': '#0A52FF'
           }
        },
        {
           'selector': '[__mclCluster=22]',
           'style': {
               'background-color': '#AAB4FB'
           }
        },
        {
           'selector': '[__mclCluster=23]',
           'style': {
               'background-color': '#1406F4'
           }
        },
        {
           'selector': '[__mclCluster=24]',
           'style': {
               'background-color': '#704CE7'
           }
        },
        {
           'selector': '[__mclCluster=25]',
           'style': {
               'background-color': '#7734D8'
           }
        },
        {
           'selector': '[__mclCluster=26]',
           'style': {
               'background-color': '#780CC4'
           }
        },
        {
           'selector': '[__mclCluster=27]',
           'style': {
               'background-color': '#9E6BAD'
           }
        },
        {
           'selector': '[__mclCluster=28]',
           'style': {
               'background-color': '#8C0295'
           }
        },
        {
           'selector': '[__mclCluster=29]',
           'style': {
               'background-color': '#7A2E71'
           }
        },
        {
           'selector': '[__mclCluster=30]',
           'style': {
               'background-color': '#5E1348'
           }
        },
        {
           'selector': '[__mclCluster=31]',
           'style': {
               'background-color': '#580732'
           }
        },
        {
           'selector': '[__mclCluster=32]',
           'style': {
               'background-color': '#744153'
           }
        },
        {
           'selector': '[__mclCluster=33]',
           'style': {
               'background-color': '#8F001A'
           }
        },
        {
           'selector': '[__mclCluster=34]',
           'style': {
               'background-color': '#A84848'
           }
        },
    ]


sortedconditions =  list(met_all.index).sort()





image_filename = '/home/dickinsonq/Flask/Meyer.png' 
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
image_filename = '/home/dickinsonq/Flask/correlation.png'
correlation_image = base64.b64encode(open(image_filename, 'rb').read())
image_filename = '/home/dickinsonq/Flask/shap.png' 
shap_image = base64.b64encode(open(image_filename, 'rb').read())
image_filename = '/home/dickinsonq/Flask/network.png' 
network_image = base64.b64encode(open(image_filename, 'rb').read())
image_filename = '/home/dickinsonq/Flask/mimal.png' 
mimal_image = base64.b64encode(open(image_filename, 'rb').read())


navbar = html.Div([dbc.NavbarSimple(

    [

    dbc.NavItem(dbc.NavLink(html.Span(
            "About",
            id="tooltip-about",
            style={"cursor": "pointer"}), href="https://www.biorxiv.org/content/10.1101/2022.05.11.491527v1")),

    dbc.Tooltip(
        "View the paper in on Biorxiv",
        target="tooltip-about"
        ),

    dbc.DropdownMenu(
        children=[
            dbc.DropdownMenuItem("Home", href="/"),
            dbc.DropdownMenuItem("Correlations", href="/page-1"),
            dbc.DropdownMenuItem("SHAP Summary", href="/page-2"),
            dbc.DropdownMenuItem("Network", href="/page-3"),
            dbc.DropdownMenuItem("Calculate MIMaL", href="/page-4")
        ],
        nav=True,
        in_navbar=True,
        label="Navigate",
    ),

    dbc.NavItem(html.Div(html.A(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
            style={"height":"100px"}),
            href="https://www.jessemeyerlab.com/"),
            className="d-none d-lg-block ml-4"))

    ],
    brand="MIMaL: Multi-Omic Integration by Machine Learning",
    brand_style={"font-size":"xxx-large", "font-style":"bold"},
    
    color="#FFFFFF",
    dark=False,
    className="mt-1",
    fluid=True
    ),
    html.Hr(style={"width":"100%","text-align":"left","margin-left":"0"})
    ])









# feature_names=pro_all.columns
# max_display = 20
# mpl_fig = shap_summary_plot(shap_values, pro_all, feature_names=pro_all.columns, max_display=20)

# plotly_fig = tls.mpl_to_plotly(mpl_fig)

# plotly_fig['layout'] = {'xaxis': {'title': 'SHAP value (impact on model output)'}}

# feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])
# feature_order = feature_order[-min(max_display, len(feature_order)):]
# text = [feature_names[i] for i in feature_order]
# text = iter(text)

# for i in range(1, len(plotly_fig['data']), 2):
#     t = text.__next__()
#     plotly_fig['data'][i]['name'] = ''
#     all_labels = []
#     for value in list(pro_all.index):
#         all_labels.append(value)
#     plotly_fig['data'][i]['text'] = all_labels
#     plotly_fig['data'][i]['hoverinfo'] = 'text'

# colorbar_trace  = go.Scatter(x=[None],
#                              y=[None],
#                              mode='markers',
#                              marker=dict(
#                                  colorscale=red_blue, 
#                                  showscale=True,
#                                  cmin=-5,
#                                  cmax=5,
#                                  colorbar=dict(thickness=5, tickvals=[-5, 5], ticktext=['Low', 'High'], outlinewidth=0)
#                              ),
#                              hoverinfo='none'
#                             )

# plotly_fig['layout']['showlegend'] = False
# plotly_fig['layout']['hovermode'] = 'closest'
# plotly_fig['layout']['height']=600
# plotly_fig['layout']['width']=500

# plotly_fig['layout']['xaxis'].update(zeroline=True, showline=True, ticklen=4, showgrid=False)
# plotly_fig['layout']['yaxis'].update(dict(visible=False))
# plotly_fig.add_trace(colorbar_trace)
# plotly_fig.layout.update(
#                          annotations=[dict(
#                               x=1.18,
#                               align="right",
#                               valign="top",
#                               text='Feature value',
#                               showarrow=False,
#                               xref="paper",
#                               yref="paper",
#                               xanchor="right",
#                               yanchor="middle",
#                               textangle=-90,
#                               font=dict(family='Calibri', size=14)
#                             )
#                          ],
#                          margin=dict(t=20)
#                         )
plotly_fig2 = px.scatter(
        x=list(pred_test['Citric acid']), y=list(met_test['Citric acid']),       
        render_mode="webgl"
    )
plotly_fig2['data'][0]['hovertemplate'] = '%{text}<br>x=%{x}<br>y=%{y}<extra></extra>'
plotly_fig2['data'][0]['text'] = np.array(met_test.index)

plotly_fig2.update_layout(
    title="Citric acid<br>R2 = " + str(round(sklearn.metrics.r2_score(list(met_test['Citric acid']),
                                                                     pred_test['Citric acid']), 3)),
    xaxis_title="Predicted",
    yaxis_title="True",
    title_x=0.5,
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="Black"
    )
)

plotly_fig2['layout']['height']=500
plotly_fig2['layout']['width']=500


x_disp = all_correlations['Citric acid']
       
y_disp = shap_meanabs['Citric acid']


plotly_fig4 = px.scatter(
    x=x_disp, y=y_disp,

    render_mode="webgl", 
    log_y=True
)
plotly_fig4['data'][0]['hovertemplate'] = '%{text}<br>x=%{x}<br>y=%{y}<extra></extra>'
plotly_fig4['data'][0]['text'] = np.array(proteins)

plotly_fig4.update_layout(

    title=str("Citric acid"),
    xaxis_title="Spearman's ρ",
    yaxis_title="|Mean SHAP|",
    title_x=0.5,
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="Black"
    )
    )
plotly_fig4['layout']['height']=500
plotly_fig4['layout']['width']=500







server = Flask(__name__)



@server.route("/download/<path:path>")

def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)





app = Dash(__name__, server = server, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
#@app.route('/')
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    navbar,
    dbc.Container([
        dbc.Row(
                [
                html.H1("MIMaL is a new method for integrating multiomic data using SHAP model explanations", style={'textAlign': 'center'}),
                html.H2("Explore our data or use MIMaL on your own dataset:", style={'textAlign': 'center'})
                ], justify="center", align="center"
                ),
    

    
    
   # dbc.Row([]),
    dbc.Row([
        dbc.Col([dbc.Card([               
                dbc.CardHeader("Correlation",
                     style={"background-color":"#6831ff",
                             "font-weight":"bold",
                             "font-size":"large",
                             "color":"white"}),
                dbc.CardBody(
                    html.Div([
                        dbc.NavItem(html.Div(html.A(html.Img(src='data:image/png;base64,{}'.format(correlation_image.decode()),
                        style={"height":"400px"}),
                        href="/page-1"),
                        className="d-none d-lg-block ml-4"))
                    ])
                )
            ])
        ]),
        dbc.Col([dbc.Card([               
                dbc.CardHeader("SHAP Summary",
                     style={"background-color":"#6831ff",
                             "font-weight":"bold",
                             "font-size":"large",
                             "color":"white"}),
                dbc.CardBody(
                    html.Div([
                        dbc.NavItem(html.Div(html.A(html.Img(src='data:image/png;base64,{}'.format(shap_image.decode()),
                        style={"height":"400px"}),
                        href="/page-2"),
                        className="d-none d-lg-block ml-4"))
                    ])
                )
            ])
        ]),
        dbc.Col([dbc.Card([               
                dbc.CardHeader("Network",
                     style={"background-color":"#6831ff",
                             "font-weight":"bold",
                             "font-size":"large",
                             "color":"white"}),
                dbc.CardBody(
                    html.Div([
                        dbc.NavItem(html.Div(html.A(html.Img(src='data:image/png;base64,{}'.format(network_image.decode()),
                        style={"height":"400px"}),
                        href="/page-3"),
                        className="d-none d-lg-block ml-4"))
                    ])
                )
            ])         
        ]),
        dbc.Col([dbc.Card([               
                dbc.CardHeader("Calculate",
                     style={"background-color":"#6831ff",
                             "font-weight":"bold",
                             "font-size":"large",
                             "color":"white"}),
                dbc.CardBody(
                    html.Div([
                        dbc.NavItem(html.Div(html.A(html.Img(src='data:image/png;base64,{}'.format(mimal_image.decode()),
                        style={"height":"400px"}),
                        href="/page-4"),
                        className="d-none d-lg-block ml-4"))
                    ])
                )
            ])
                        
        ]),
    ]),
    dbc.Row([
        dbc.Card([               
                dbc.CardHeader("References",
                     style={"background-color":"#6831ff",
                             "font-weight":"bold",
                             "font-size":"large",
                             "color":"white"}),
                dbc.CardBody(
                    html.Div([
                        html.Div([                        
                        "Preprint: Dickinson Q, Aufschnaiter A, Ott M, Meyer JG. Multi-Omic Integration by Machine Learning (MIMaL) Reveals Protein-Metabolite Connections and New Gene Functions. bioRxiv 2022.05.11.491527; doi: https://doi.org/10.1101/2022.05.11.491527"
                        ]),
                        html.Br(),
                        html.Div([
                        "Data Source: Stefely JA, Kwiecien NW, Freiberger EC, Richards AL, Jochem A, Rush MJP, Ulbrich A, Robinson KP, Hutchins PD, Veling MT, Guo X, Kemmerer ZA, Connors KJ, Trujillo EA, Sokol J, Marx H, Westphall MS, Hebert AS, Pagliarini DJ, Coon JJ. Mitochondrial Protein Functions Elucidated by Multi-Omic Mass Spectrometry Profiling. Nature Biotechnology 34, 1191–1197 (2016). doi:10.1038/nbt.3683"])    
                        ])
                )
            ])
        
        
    ])       
     ]),       
    
])



####PAGE1#####






page_1_layout = dbc.Container([
    navbar,

    dbc.Row([
        dbc.Col([  
            
            dbc.Card([               
                dbc.CardHeader("Navigation",
                     style={"background-color":"#6831ff",
                             "font-weight":"bold",
                             "font-size":"large",
                             "color":"white"}),
                dbc.CardBody(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Correlation", active=True, href="pca", style={"background-color":"grey"})),
                        
                        dbc.NavItem(
                            dbc.NavLink(

                            html.Span(
                                    "SHAP Summary",
                                    id="tooltip-lr",
                                    style={"cursor": "pointer", "color":"grey"},
                                ),disabled=False, href="/page-2")),

                        

                        

                        dbc.NavItem(dbc.NavLink(
                            html.Span(
                                    "Network",
                                    id="tooltip-cg",
                                    style={"cursor":"pointer", "color":"grey"},
                                ),disabled=False, href="/page-3")),
                        
                        dbc.NavItem(dbc.NavLink(
                            html.Span(
                                    "SHAP Calculation",
                                    id="tooltip-cg",
                                    style={"cursor":"pointer", "color":"grey"},
                                ),disabled=False, href="/page-4")),

                        html.Hr()
                    ])
                )

            ]),
            dbc.Card([
                dbc.CardHeader("Options",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color": "white"}),

                dbc.CardBody(
                    html.Div([
                        "Condition",
                        dcc.RadioItems(
                           id='condition',
                           options=[
                               {'label': 'All', 'value': 'all'},
                               {'label': 'Fermentation', 'value': 'ferm'},
                               {'label': 'Respiration', 'value': 'resp'},
                           ],
                           value='all'
                        ),
                        "Data X",
                        dcc.RadioItems(
                           id='option1',
                           options=[
                               {'label': 'Metabolite ', 'value': 'metabolite'},
                               {'label': 'Protein ', 'value': 'protein'},
                               {'label': 'SHAP ', 'value': 'shap'},
                               
                           ],
                           value='protein'
                        ),
                        html.Div([
                            "Metabolite X",
                            dcc.Dropdown(
                                id='metabolite1', clearable=False,
                                value='Alanine', options=[
                                    {'label': c, 'value': c}
                                    for c in metabolites

                                ]),        
                            'Protein X',
                            dcc.Dropdown(
                                id='protein1', clearable=False,
                                value='AAC1 (YMR056c)', options=[
                                    {'label': d, 'value': d}
                                    for d in proteins

                                ])
                        ], id='dropdown1'),
                        "Data Y",
                        dcc.RadioItems(
                           id='option2',
                           options=[
                               {'label': 'Metabolite ', 'value': 'metabolite'},
                               {'label': 'Protein ', 'value': 'protein'},
                               {'label': 'SHAP ', 'value': 'shap'},
                               
                           ],
                           value='protein'
                        ),
                        html.Div([
                            "Metabolite Y",
                            dcc.Dropdown(
                                id='metabolite2', clearable=False,
                                value='Alanine', options=[
                                    {'label': d, 'value': d}
                                    for d in metabolites

                                ]),
                            'Protein Y',
                            dcc.Dropdown(
                                id='protein2', clearable=False,
                                value='AAC1 (YMR056c)', options=[
                                    {'label': d, 'value': d}
                                    for d in proteins

                                ])
                        ], id='dropdown2')
                    ])
                )


            ]),
            
            dbc.Card([
                dbc.CardHeader("Help",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color": "white"}),

                dbc.CardBody(
                    html.Div([
                        html.Div(["This page allows you to see correlations (Spearman's rank correlation) between any of the datasets used in this study"]),
                        html.Br(),
                        html.Div(["Condition: Select which experimental conditions you would like displayed"]),
                        html.Br(),
                        html.Div(["Data X: Select the type of data you would like on the X axis."]),
                        html.Br(),
                        html.Div(["Data Y: Select the type of data you would like on the Y axis."]),
                        html.Br(),
                        html.Div(["Metabolite: Fold change of Metabolite vs control."]),
                        html.Br(),
                        html.Div(["Protein: Fold change of Protein vs control."]),
                        html.Br(),
                        html.Div(["SHAP: Calculated SHAP contribution of specified protein for specified metabolite."]),
                    ])
                )


            ])

        ], width = 2),
        
        dbc.Col(
            [dbc.Card(
                [dbc.CardHeader("Correlations",
                            style={"background-color":"#6831ff",
                                    "font-weight":"bold",
                                    "font-size":"large",
                                    "color":"white"}),
                
                
                 dbc.CardBody(dcc.Graph(id='graph'))
                ]
            )
        ]),
        
    ])
], fluid = True)
@app.callback(
    #Output('page-1-content', 'children'),
    Output('graph', 'figure'),
    Output('dropdown1', 'children'),
    Output('dropdown2', 'children'),
    #Output('shap1', 'children'),
    #Output('shap2', 'children'),
    [Input("condition", "value"),
    Input("option1", "value"),
    Input("metabolite1", "value"),
    Input("protein1", "value"),
    Input("option2", "value"),
    Input("metabolite2", "value"),
    Input("protein2", "value")]
    #[Input('page-1-dropdown', 'value')]
    )
    







def page_1_dropdown(condition, option1,  metabolite1, protein1, option2, metabolite2, protein2):
    
    
    if condition == 'all':
        pro = pro_all
        met = met_all
        path = 'all//'
        #shap = shap_all
        
    if condition == 'ferm':
        pro = pro_ferm
        met = met_ferm
        path = 'ferm//'
        #shap = shap_ferm
        
    if condition == 'resp':
        pro = pro_resp
        met = met_resp
        path = 'resp//'
        #shap = shap_resp
    
    x_disp = []
    xtitle = ''
    y_disp = []
    ytitle = ''
    
    if option1 == 'protein':
        x_disp = pro[protein1]
        xtitle = protein1
        dropdown1 = html.Div([
                            "Metabolite X",
                            dcc.Dropdown(
                                id='metabolite1', disabled = True, clearable=False,
                                value=metabolite1, options=[
                                    {'label': c, 'value': c}
                                    for c in metabolites

                                ]),        
                            'Protein X',
                            dcc.Dropdown(
                                id='protein1', disabled = False, clearable=False,
                                value=protein1, options=[
                                    {'label': d, 'value': d}
                                    for d in proteins

                                ])
                        ], id='dropdown1')
        
        
        
    if option1 == 'metabolite':
        x_disp = met[metabolite1]
        xtitle = metabolite1
        dropdown1 = html.Div([
                            "Metabolite X",
                            dcc.Dropdown(
                                id='metabolite1', disabled = False, clearable=False,
                                value=metabolite1, options=[
                                    {'label': c, 'value': c}
                                    for c in metabolites

                                ]),        
                            'Protein X',
                            dcc.Dropdown(
                                id='protein1', disabled = True, clearable=False,
                                value=protein1, options=[
                                    {'label': d, 'value': d}
                                    for d in proteins

                                ])
                        ], id='dropdown1')
        
    if option1 == 'shap':
        index1 = list(met.columns).index(metabolite1)
        index2 = list(pro.columns).index(protein1)
        x_disp = pickle.load(open('SHAP//'+ path + metabolite1.split("?")[0] +'.pickle', 'rb'))[:,index2]
        xtitle = "SHAP of " + protein1 + " for " + metabolite1
        dropdown1 = html.Div([
                            "Metabolite X",
                            dcc.Dropdown(
                                id='metabolite1', disabled = False, clearable=False,
                                value=metabolite1, options=[
                                    {'label': c, 'value': c}
                                    for c in metabolites

                                ]),        
                            'Protein X',
                            dcc.Dropdown(
                                id='protein1', disabled = False, clearable=False,
                                value=protein1, options=[
                                    {'label': d, 'value': d}
                                    for d in proteins

                                ])
                        ], id='dropdown1')
        
        
    if option2 == 'protein':
        y_disp = pro[protein2]
        ytitle = protein2
        dropdown2 = html.Div([
                            "Metabolite Y",
                            dcc.Dropdown(
                                id='metabolite2', disabled = True, clearable=False,
                                value=metabolite2, options=[
                                    {'label': d, 'value': d}
                                    for d in metabolites

                                ]),
                            'Protein Y',
                            dcc.Dropdown(
                                id='protein2', disabled =False, clearable=False,
                                value=protein2, options=[
                                    {'label': d, 'value': d}
                                    for d in proteins

                                ])
                        ], id='dropdown2')
        
    if option2 == 'metabolite':
        y_disp = met[metabolite2]
        ytitle = metabolite2
        dropdown2 = html.Div([
                            "Metabolite Y",
                            dcc.Dropdown(
                                id='metabolite2', disabled = False, clearable=False,
                                value=metabolite2, options=[
                                    {'label': d, 'value': d}
                                    for d in metabolites

                                ]),
                            'Protein Y',
                            dcc.Dropdown(
                                id='protein2', disabled = True, clearable=False,
                                value=protein2, options=[
                                    {'label': d, 'value': d}
                                    for d in proteins

                                ])
                        ], id='dropdown2')
    if option2 == 'shap':
        index3 = list(met.columns).index(metabolite2)
        index4 = list(pro.columns).index(protein2)
        y_disp = pickle.load(open('SHAP//'+ path + metabolite2.split("?")[0] +'.pickle', 'rb'))[:,index4]
        ytitle = "SHAP of " + protein2 + " for " + metabolite2
        dropdown2 = html.Div([
                            "Metabolite Y",
                            dcc.Dropdown(
                                id='metabolite2', disabled = False, clearable=False,
                                value=metabolite2, options=[
                                    {'label': d, 'value': d}
                                    for d in metabolites

                                ]),
                            'Protein Y',
                            dcc.Dropdown(
                                id='protein2', disabled = False, clearable=False,
                                value=protein2, options=[
                                    {'label': d, 'value': d}
                                    for d in proteins

                                ])
                        ], id='dropdown2')
    
    plotly_fig3 = px.scatter(
        x=x_disp, y=y_disp,       
        render_mode="webgl"
    )
    plotly_fig3['data'][0]['hovertemplate'] = '%{text}<br>x=%{x}<br>y=%{y}<extra></extra>'
    plotly_fig3['data'][0]['text'] = np.array(met.index)
    
    plotly_fig3.update_layout(
        
        title="<br>ρ = " + str(round(scipy.stats.spearmanr(list(x_disp),list(y_disp))[0], 3)),
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        title_x=0.5
        )
    
    plotly_fig3['layout']['height']=600
    plotly_fig3['layout']['width']=600
    
    
    
    
   
    return plotly_fig3, dropdown1, dropdown2




#######PAGE 2#################



# fourth_card = dbc.Card(
#     [
#         dbc.CardHeader("BIOMOLECULE BOXPLOT",
#                             style={"background-color":"#5bc0de",
#                                     "font-weight":"bold",
#                                     "font-size":"large"}),
#         dbc.CardBody(dcc.Graph(id='biomolecule-boxplot',
#         config=plotly_config))
#     ])












page_2_layout = dbc.Container([
    navbar,
    
    
    dbc.Row([
        
        dbc.Col([
               
            
            dbc.Card([               
                dbc.CardHeader("Navigation",
                     style={"background-color":"#6831ff",
                             "font-weight":"bold",
                             "font-size":"large"}),
                dbc.CardBody(
                    dbc.Nav([
                        dbc.NavItem(
                            dbc.NavLink(

                            html.Span(
                                    "Correlation",
                                    id="tooltip-lr",
                                    style={"cursor": "pointer", "color":"grey"},
                                ),disabled=False, href="/page-1")),

                        dbc.NavItem(dbc.NavLink("SHAP Summary", active=True, href="pca", style={"background-color":"grey"})),

                        

                        dbc.NavItem(dbc.NavLink(
                            html.Span(
                                    "Network",
                                    id="tooltip-cg",
                                    style={"cursor":"pointer", "color":"grey"},
                                ),disabled=False, href="/page-3")),
                        dbc.NavItem(dbc.NavLink(
                            html.Span(
                                    "SHAP Calculation",
                                    id="tooltip-cg",
                                    style={"cursor":"pointer", "color":"grey"},
                                ),disabled=False, href="/page-4")),

                        html.Hr()
                    ])
                )

                ], inverse=True),
                dbc.Card([
                    dbc.CardHeader("Options",
                             style={"background-color":"#6831ff",
                                     "font-weight":"bold",
                                     "font-size":"large",
                                     "color": "white"}),

                    dbc.CardBody(
                            html.Div(['Metabolite',
                            dcc.Dropdown(
                            id='metabolite-dropdown', clearable=False,
                            value='Citric acid', options=[
                                {'label': c, 'value': c}
                                for c in sorted(list(met_all.columns))
                            ]
                         )
                                     ])
                    ),
                    dbc.CardBody(
                        html.Div(["Condition",
                        html.Div([
                            dcc.Dropdown(
                            id='condition-dropdown', clearable=False,
                            value='∆MEF1_resp', options=[
                                {'label': c, 'value': c}
                                for c in sorted(list(met_all.index))
                            ]
                         )
                        ], id = 'condition-dropdown-html')
                                  ])
                    )


                ]),
                dbc.Card([
                dbc.CardHeader("Help",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color": "white"}),

                dbc.CardBody(
                    html.Div([
                        html.Div(["This page allows you to explore the data related to the calculated SHAP values for each condition for each metabolite"]),
                        html.Br(),
                        html.Div(["Metabolite: Select which metabolite you would like displayed"]),
                        html.Br(),
                        html.Div(["Condition: Select the condition (each knockout under respiration or fermentation). You can also click dots on plots to change conditions."]),
                        html.Br(),
                        html.Div(["Summary Plot: Displays SHAP values across all conditions for the top 20 most important proteins."]),
                        html.Br(),
                        html.Div(["Real vs Predicted: Displays model performance for metabolite based on test set."]),
                        html.Br(),
                        html.Div(["Force Plot: Displays the SHAP values for each protein for selected condition."]),
                        
                    ])
                )


            ]),
            
            
                html.Div(['asdf'], id = 'sel1', style = dict(display='none')),
                html.Div(['asdf'], id = 'sel2', style = dict(display='none')),
                html.Div(['asdf'], id = 'sel3', style = dict(display='none')),
            ], width = 2),
        
        
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Summary Plot",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large"}),

                dbc.CardBody(
                    dcc.Graph(id='graph1', figure = plotly_fig))

            ], inverse=True)

        ]),
                 
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Real vs Predicted",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color":"white"}),
                
                
                
                dbc.CardBody(
                    html.Div([
                    dcc.Graph(id='graph2', figure = plotly_fig2),
                    dcc.RadioItems(
                           id='testset',
                           options=[
                               {'label': 'Test ', 'value': 'True'},
                               {'label': 'All ', 'value': 'False'},
                               
                               
                           ],
                           value='True'
                        )])
                )
            ])
        ]),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Feature Importance",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large"}),

                dbc.CardBody(
                    dcc.Graph(id='graph4', figure = plotly_fig4))

            ], inverse=True)

        ])
    ]),
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Div([html.Div(["MEF1"])], id = 'shapheader'),
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large"}),
                
                
                
                dbc.CardBody(            
                    html.Div([html.Iframe(srcDoc=shap_html, width = "100%")], id = 'shapplot') 
                )
            ], inverse=True)
        ])
    ])
], fluid = True)


                    


@app.callback(Output('graph1', 'figure'), 
              Output('graph2', 'figure'),
              Output('shapplot', 'children'),
              Output('shapheader', 'children'),
              Output('graph4', 'figure'),
              Output('sel1', 'children'),
              Output('sel2', 'children'),
              Output("condition-dropdown-html", "children"),
              Output('sel3', 'children'),
              [Input("metabolite-dropdown", "value")],
              [Input("graph1", "clickData")],
              [Input("graph2", "clickData")],
              [Input("sel1", "children")],
              [Input("sel2", "children")],
              [Input("testset", "value")],
              [Input("condition-dropdown", "value")],
              [Input('sel3', 'children')]
)



def page_2_radios(metabolite, click1, click2, sel1, sel2, testset, conditiondrop, sel3):
    ##print(metabolite)
    ##print(click1)
    ##print(click2)
    shap_values = pickle.load(open('SHAP//all//'+metabolite.split('?')[0]+'.pickle', 'rb'))
    plotly_fig = pickle.load(open('figures//summary//' + metabolite.split('?')[0] + '.pickle', 'rb'))
    condition_dropdown_html = html.Div([
                            dcc.Dropdown(
                            id='condition-dropdown', clearable=False,
                            value='∆MEF1_resp', options=[
                                {'label': c, 'value': c}
                                for c in sorted(list(met_all.index))
                            ]
                         )
                        ])
    
    
    if testset == "True":
        ##print('test')
        plotly_fig2 = px.scatter(
            x=list(pred_test[metabolite]), y=list(met_test[metabolite]),       
            render_mode="webgl"
        )
        
        plotly_fig2.update_layout(
        title=metabolite + "<br>R2 = " + str(round(sklearn.metrics.r2_score(list(met_test[metabolite]),
                                                                         pred_test[metabolite]), 3)),
        xaxis_title="Predicted",
        yaxis_title="True",
        title_x=0.5,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="Black"
            )
        )
        plotly_fig2['data'][0]['hovertemplate'] = '%{text}<br>x=%{x}<br>y=%{y}<extra></extra>'
        plotly_fig2['data'][0]['text'] = np.array(met_test.index)
    else:
        #print('all')
        plotly_fig2 = px.scatter(
            x=list(pred[metabolite.replace(' ', '-')]), y=list(met_all[metabolite]),       
            render_mode="webgl"
        )
        
        plotly_fig2.update_layout(
        title=metabolite + "<br>R2 = " + str(round(sklearn.metrics.r2_score(list(met_all[metabolite]),
                                                                         pred[metabolite.replace(' ', '-')]), 3)),
        xaxis_title="Predicted",
        yaxis_title="True",
        title_x=0.5,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="Black"
            )
        )
        plotly_fig2['data'][0]['hovertemplate'] = '%{text}<br>x=%{x}<br>y=%{y}<extra></extra>'
        plotly_fig2['data'][0]['text'] = np.array(met_all.index)
    
    
    
    plotly_fig2['layout']['height']=500
    plotly_fig2['layout']['width']=500
    
    ##print(shapheader)
    #header = shapheader
    ##print(type(header))
    
    
    
    
    
    #if type(header)!=type([]):
    #    header = [header]
        
    ##print(click1['points'][0]['text'])
    ##print(header[0]['props']['children'][0])
    
    header = '∆MEF1_resp'
    value = '∆MEF1_resp'
    if (type(click1) != type(None)):
        ##print(click1)
        if click1['points'][0]['text'] != sel1[0]:
            header = html.Div([click1['points'][0]['text']])
            value = click1['points'][0]['text']
            sel1 = [value]
            sel3 = [value]
            conditiondrop = value
            ##print('conditiondropchange')
            condition_dropdown_html = html.Div([
                            dcc.Dropdown(
                            id='condition-dropdown', clearable=False,
                            value=value, options=[
                                {'label': c, 'value': c}
                                for c in sorted(list(met_all.index))
                            ]
                            )
                        ])
    if type(click2) != type(None):
        ##print(click2)
        if click2['points'][0]['text'] != sel2[0]:
            header = html.Div([click2['points'][0]['text']])
            value = click2['points'][0]['text']
            sel2 = [value]
            sel3 = [value]
            conditiondrop = value
            ##print('conditiondropchange')
            condition_dropdown_html = html.Div([
                            dcc.Dropdown(
                            id='condition-dropdown', clearable=False,
                            value=value, options=[
                                {'label': c, 'value': c}
                                for c in sorted(list(met_all.index))
                            ]
                            )
                        ])
    if conditiondrop != sel3[0]:
            #print(conditiondrop)
            header = html.Div([conditiondrop])
            value = conditiondrop
            sel3 = [value]
            condition_dropdown_html = html.Div([
                            dcc.Dropdown(
                            id='condition-dropdown', clearable=False,
                            value=value, options=[
                                {'label': c, 'value': c}
                                for c in sorted(list(met_all.index))
                            ]
                            )
                        ])

    fp_ind = list(met_all.index).index(value)
    
    
    
    force_plot = shap.force_plot(expected[metabolite], shap_values[fp_ind,:], pro_all.iloc[fp_ind,:], show=True, 
                matplotlib=False,
                text_rotation=90)


    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    #shap_html  = pickle.load(open('figures//force//' + str(fp_ind) + "-" + metabolite.split('?')[0]+'.pickle', 'rb'))
    
    
    shapplot = html.Iframe(srcDoc=shap_html, width = "100%")
    
    
    x_disp = all_correlations[metabolite]
       
    y_disp = shap_meanabs[metabolite]
     
    
    plotly_fig4 = px.scatter(
        x=x_disp, y=y_disp,
        
        render_mode="webgl", 
        log_y=True
    )
    plotly_fig4['data'][0]['hovertemplate'] = '%{text}<br>x=%{x}<br>y=%{y}<extra></extra>'
    plotly_fig4['data'][0]['text'] = np.array(proteins)
    
    plotly_fig4.update_layout(
        
        title=str(metabolite),
        xaxis_title="Spearman's ρ",
        yaxis_title="|Mean SHAP|",
        title_x=0.5
        )
    plotly_fig4['layout']['height']=500
    plotly_fig4['layout']['width']=500
    
    
    
    
    
    return plotly_fig, plotly_fig2, shapplot, header, plotly_fig4, sel1, sel2, condition_dropdown_html, sel3


#################page 3############





page_3_layout = dbc.Container([
    navbar,
    
    
    dbc.Row([
        
        dbc.Col([
               
            
            dbc.Card([               
                dbc.CardHeader("Navigation",
                     style={"background-color":"#6831ff",
                             "font-weight":"bold",
                             "font-size":"large"}),
                dbc.CardBody(
                    dbc.Nav([
                        dbc.NavItem(
                            dbc.NavLink(

                            html.Span(
                                    "Correlation",
                                    id="tooltip-lr",
                                    style={"cursor": "pointer", "color":"grey"},
                                ),disabled=False, href="/page-1")),
                        
                        dbc.NavItem(dbc.NavLink(
                            html.Span(
                                    "SHAP Summary",
                                    id="tooltip-de",
                                    style={"cursor": "pointer", "color":"grey"},
                                ),disabled=False, href="/page-2")),                  

                        
                        

                    
                        
                        dbc.NavItem(dbc.NavLink("Network", active=True, href="pca", style={"background-color":"grey"})),
                        
                        dbc.NavItem(dbc.NavLink(
                            html.Span(
                                    "SHAP Calculation",
                                    id="tooltip-cg",
                                    style={"cursor":"pointer", "color":"grey"},
                                ),disabled=False, href="/page-4")),


                        html.Hr()
                    ])
                )

            ], inverse=True),
            dbc.Card([
                dbc.CardHeader("Options",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color":"white"}),

                dbc.CardBody(
                        html.P("n/a")
                )


            ], inverse=True),
            dbc.Card([
                dbc.CardHeader("Help",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color": "white"}),

                dbc.CardBody(
                    html.Div([
                        html.Div(["This page allows you to explore the network related to the calculated SHAP values"]),
                        html.Br(),
                        html.Div(["Network: Full network of conditions, connected by their relatedness"]),
                        html.Br(),
                        html.Div(["Sub-Network: Click node on main network to view neighbors and the calculated weights."]),
                        
                        
                    ])
                )


            ])

        ], width = 2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Network",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color":"white"}),

                dbc.CardBody(
                    cyto.Cytoscape(
                        id='cytoscape-event-callbacks-1',
                        layout={'name': 'preset'},
                        style={'width': '100%', 'height': '800px'},
                        elements=data['elements'],

                        stylesheet=clusterstyle

                    )
                )
            ])
        ]),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sub-Network",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large"}),

                dbc.CardBody(
                    html.Div([
                        cyto.Cytoscape(
                            
                            layout={'name': 'preset'},
                            style={'width': '100%', 'height': '800px'},
                            elements={},

                            stylesheet=clusterstyle

                        )
                    ], id='cytoscape-event-callbacks-2')
                    
                )
            ], inverse=True)
        ], width = 3)
    ])

       
    
], fluid = True)

@app.callback(Output('cytoscape-event-callbacks-2', 'children'),
              Input('cytoscape-event-callbacks-1', 'tapNodeData'))


def displayTapNodeData(node):
    #print(node)
    edges = []
    elm = {}
    if type(node) != None:
        #print(type(node))
        for edge in data['elements']['edges']:
            if node['id'] == edge['data']['source'] or node['id'] == edge['data']['target']:
                edges.append(edge)
        nodes = []
        for edge in edges:
            for node in data['elements']['nodes']:
                if node['data']['id'] == edge['data']['source']:
                    nodes.append(node)
                if node['data']['id'] == edge['data']['target']:
                    nodes.append(node)
    
        elm['edges'] = edges
        elm['nodes'] = nodes
    
    
    asdf = html.Div([cyto.Cytoscape(
                        
                        layout={'name': 'preset'},
                        style={'width': '100%', 'height': '800px'},
                        elements=elm,
                        stylesheet=[

                                 {
                                    'selector': 'node',
                                    'style': {
                                        'label': 'data(label)',
                                        #'background-color': 'mapData(__mclCluster, 1, 34, blue, red)'
                                        #'background-color': "setNodeColorMapping('score', colors=paletteColorBrewerSet3, mapping.type='d')"
                                    }
                                },
                                {
                                    'selector': 'edge',
                                    'style': {
                                        'label': 'data(weight)',
                                        #'background-color': 'mapData(__mclCluster, 1, 34, blue, red)'
                                        #'background-color': "setNodeColorMapping('score', colors=paletteColorBrewerSet3, mapping.type='d')"
                                    }
                                },
                            
                                {
                                   'selector': '[__mclCluster=1]',
                                   'style': {
                                       'background-color': '#FF2D00'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=2]',
                                   'style': {
                                       'background-color': '#FDAA7D'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=3]',
                                   'style': {
                                       'background-color': '#F6901E'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=4]',
                                   'style': {
                                       'background-color': '#EBB124'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=5]',
                                   'style': {
                                       'background-color': '#DCCE61'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=6]',
                                   'style': {
                                       'background-color': '#BEC900'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=7]',
                                   'style': {
                                       'background-color': '#A1B463'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=8]',
                                   'style': {
                                       'background-color': '#609B0E'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=9]',
                                   'style': {
                                       'background-color': '#448119'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=10]',
                                   'style': {
                                       'background-color': '#366527'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=11]',
                                   'style': {
                                       'background-color': '#055001'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=12]',
                                   'style': {
                                       'background-color': '#426C47'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=13]',
                                   'style': {
                                       'background-color': '#09882E'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=14]',
                                   'style': {
                                       'background-color': '#26A260'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=15]',
                                   'style': {
                                       'background-color': '#3EBA8E'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=16]',
                                   'style': {
                                       'background-color': '#04CFAB'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=17]',
                                   'style': {
                                       'background-color': '#95E0E0'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=18]',
                                   'style': {
                                       'background-color': '#0AC6EE'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=19]',
                                   'style': {
                                       'background-color': '#45B9F8'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=20]',
                                   'style': {
                                       'background-color': '#499EFE'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=21]',
                                   'style': {
                                       'background-color': '#0A52FF'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=22]',
                                   'style': {
                                       'background-color': '#AAB4FB'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=23]',
                                   'style': {
                                       'background-color': '#1406F4'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=24]',
                                   'style': {
                                       'background-color': '#704CE7'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=25]',
                                   'style': {
                                       'background-color': '#7734D8'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=26]',
                                   'style': {
                                       'background-color': '#780CC4'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=27]',
                                   'style': {
                                       'background-color': '#9E6BAD'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=28]',
                                   'style': {
                                       'background-color': '#8C0295'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=29]',
                                   'style': {
                                       'background-color': '#7A2E71'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=30]',
                                   'style': {
                                       'background-color': '#5E1348'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=31]',
                                   'style': {
                                       'background-color': '#580732'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=32]',
                                   'style': {
                                       'background-color': '#744153'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=33]',
                                   'style': {
                                       'background-color': '#8F001A'
                                   }
                                },
                                {
                                   'selector': '[__mclCluster=34]',
                                   'style': {
                                       'background-color': '#A84848'
                                   }
                                },
                            ]
    
    
    
    
    
    
                    )
                ])
    
    #print(type(asdf))
    return asdf


#########PAGE 4#################


page_4_layout = dbc.Container([
    
        navbar,
        dbc.Row([
        
            dbc.Col([


                dbc.Card([               
                    dbc.CardHeader("Navigation",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large"}),
                    dbc.CardBody(
                        dbc.Nav([
                            dbc.NavItem(
                                dbc.NavLink(

                                html.Span(
                                        "Correlation",
                                        id="tooltip-lr",
                                        style={"cursor": "pointer", "color":"grey"},
                                    ),disabled=False, href="/page-1")),

                            dbc.NavItem(dbc.NavLink(
                            html.Span(
                                    "SHAP Summary",
                                    id="tooltip-cg",
                                    style={"cursor":"pointer", "color":"grey"},
                                ),disabled=False, href="/page-2")),


                            dbc.NavItem(dbc.NavLink(
                                html.Span(
                                        "Network",
                                        id="tooltip-cg",
                                        style={"cursor":"pointer", "color":"grey"},
                                    ),disabled=False, href="/page-3")),
                            
                            
                            
                            dbc.NavItem(dbc.NavLink("SHAP Calculation", active=True, href="pca", style={"background-color":"grey"})),

                            html.Hr()
                        ])
                    )

                    ], inverse=True),
                
                dbc.Card([
                dbc.CardHeader("Options",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color":"white"}),

                dbc.CardBody(
                    html.P([
                        html.P([
                        "Model",    
                        dcc.RadioItems(
                           id='modeltype',
                           options=[
                               {'label': 'Random Forest ', 'value': 'rand'},
                               {'label': 'Extra Trees ', 'value': 'etr'},
                               {'label': 'XGBoost ', 'value': 'xgb'},
                               
                           ],
                           value='etr'
                        )]),
                        "Imput Data",      
                        html.P([dcc.RadioItems(
                           id='imput',
                           options=[
                               {'label': 'True ', 'value': 'imptrue'},
                               {'label': 'False ', 'value': 'impfalse'},
                               
                               
                           ],
                           value='impfalse'
                        )]),
                        "Scale Data",        
                        html.P([dcc.RadioItems(
                           id='normalize',
                           options=[
                               {'label': 'True ', 'value': 'normtrue'},
                               {'label': 'False ', 'value': 'normfalse'},
                               
                               
                           ],
                           value='normfalse'
                        )]),
                        'Number Trees',
                        html.P([dcc.RadioItems(
                           id='numtrees',
                           options=[
                               {'label': '10 ', 'value': '10'},
                               {'label': '100 ', 'value': '100'},
                               {'label': '1000 ', 'value': '1000'},
                               
                           ],
                           value='100'
                        )]),
                        'Tree Depth',
                        html.P([dcc.RadioItems(
                           id='treedepth',
                           options=[
                               {'label': '3 ', 'value': '3'},
                               {'label': '10 ', 'value': '10'},
                               {'label': '30 ', 'value': '30'},
                               
                           ],
                           value='10'
                        )]),
                        
                        html.Div(["Session ID:"]),
                        html.Div(["Not Generated"], id = 'upload-loc'),
                        html.Div([html.Button('Generate New Session', id='generate-session', n_clicks=0)], id = 'generate-session-html'),
                        html.Div(["Load Session ID:"]),
                        dcc.Input(id="load-session-id", type="text", placeholder="", debounce=True),
                        html.Div([html.Button('Load Session', id='load-session', n_clicks=0)], id = 'load-session-html'),       
                               
                               
                               
                    ])
                )

                ]),
                
                dbc.Card([
                dbc.CardHeader("Help",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color": "white"}),

                dbc.CardBody(
                    html.Div([
                        html.Div(["This page allows you to run MIMaL on your own data"]),
                        html.Br(),
                        html.Div(["Model: Select the machine learning model used"]),
                        html.Br(),
                        html.Div(["Imput Data: True-Use KNN imputation for missing values in your dataset. False-Removes all samples containing missing values."]),
                        html.Br(),
                        html.Div(["Scale Data: True-Use StandardScaler to standardize features to a mean of 0 and a variance of 1. False-No Scaling."]),
                        html.Br(),
                        html.Div(["Number Trees: Number of trees used in the model training. Higher values may lead to overfitting"]),
                        html.Br(),
                        html.Div(["Tree Depth: Maximum branches in each tree. Higher values can lead to overfitting"]),
                        html.Br(),
                        html.Div(["Session ID: Value used to recognize session. Sessions will be saved for one week. Each dataset should be run with a new session"]),
                        html.Br(),
                        html.Div(["Load Session: Enter Session ID and click button to load session."]),
                    ])
                )


                ])
                
                
                
                
                
                ], width = 2),


        
        dbc.Col([
            dbc.Card([               
                    dbc.CardHeader("Input",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color":"white"}),
                    dbc.CardBody(
                        html.Div([
                            html.H2("Upload Data"),
                            
                            html.Div([dcc.Upload(
                                id="input-data",
                                children=html.Div(
                                    ["Click to select a .csv file to upload."]
                                ),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px",
                                },
                                multiple=True,
                            )], id = 'upload-html'),
                            html.H3("Input Data"),
                            html.Div([
                                    dcc.Dropdown(
                                    id='input-dropdown', clearable=False,
                                    value='Pick Input', options=[]
                                     )
                                    ], id = 'input-dropdown-html'),
                            html.H3("Output Data"),
                            html.Div([
                                    dcc.Dropdown(
                                    id='output-dropdown', clearable=False,
                                    value='Pick Output', options=[]
                                     )
                                    ], id = 'output-dropdown-html'),
                            
                            html.Div([html.Button('Submit', id='submit-val', n_clicks=0)], id = 'button-html'),
                            html.H2("File List"),
                            html.Ul(id="file-list"),
                            
                        ])
                    )
                ])
            ], width = 2),
            dbc.Col([
                dbc.Card([               
                    dbc.CardHeader("Output",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color":"white"}),
                    dbc.CardBody(
                        dbc.Spinner(html.Div([
                            html.Div([], id = 'output-html'),
                            #html.Div([], id = 'metabolite-dropdown'),
                            html.Div([], id = 'outputshap-dropdown'),
                            #,
                            html.Div([html.Button('Submit', id = 'umap-button', n_clicks=0, disabled = True)], id = 'umap-button-html'),
                            
                        ]))
                    )
                ])
            ], width = 4),
            
            dbc.Col([
                dbc.Card([               
                    dbc.CardHeader("UMAP Output",
                         style={"background-color":"#6831ff",
                                 "font-weight":"bold",
                                 "font-size":"large",
                                 "color":"white"}),
                    dbc.CardBody(
                        html.Div([
                            html.Div([], id = 'output-umap-html'),
                            #html.Div([], id = 'metabolite-dropdown'),
                            html.Div([], id = 'network')
                        ])
                    )
                ])
            ], width = 4)
        ])#,
        #style={"max-width": "500px"},
    ], fluid = True)

@app.callback(
    Output("file-list", "children"),
    Output("input-dropdown-html", "children"),
    Output("output-dropdown-html", "children"),
    Output("button-html", "children"),
    Output("output-html", "children"),
    Output("umap-button-html", 'children'),
    Output("network", 'children'),
    Output("upload-loc", 'children'),
    Output("generate-session-html", 'children'),
    Output('upload-html', 'children'),
    Output('load-session-html', 'children'),
    [Input("input-data", "filename")], 
    [Input("input-data", "contents")],
    [Input("input-dropdown", "value")],
    [Input("output-dropdown", "value")],
    [Input("submit-val", "n_clicks")],
    [Input("outputshap-dropdown", "value")],
    [Input("output-html", "children")],
    [Input("modeltype", "value")],
    [Input("imput", "value")],
    [Input("normalize", "value")],
    [Input("numtrees", "value")],
    [Input("treedepth", "value")],
    [Input("umap-button", "n_clicks")],
    [Input("network", "children")],
    [Input("upload-loc", "children")],
    [Input("generate-session", "n_clicks")],
    [Input("load-session", "n_clicks")],
    [Input("load-session-id", "value")],
)

   
    
    
def update_page4(input_filenames, input_file_contents, input_dropdown, 
                 output_dropdown, submit_val, outputshap_dropdown, output_html, 
                 modeltype, imput, normalize, numtrees, treedepth, umap_button, network,
                 upload_loc, generate_session, load_session, load_session_id):
    #"""Save uploaded files and regenerate the file list."""
    
    
    
    
    
    
    
    output_html = html.Div([output_html])
    network = network
    #print(outputshap_dropdown)
    ##print(input_file_contents)
    new_generate = False
    buttonhtml = html.Div([html.Button('Submit', id='submit-val', n_clicks=0)])
     
    generate_session_html = html.Div([html.Button('Generate New Session', id='generate-session', n_clicks=0)]) 
    input_dropdown_html = html.Div([
                    dcc.Dropdown(
                    id='input_dropdown', clearable=False,
                    value='Upload Input File', options=['Upload Input File']
                     )
                    ])
    
    output_dropdown_html = html.Div([
                    dcc.Dropdown(
                    id='output_dropdown', clearable=False,
                    value='Upload Output File', options=['Upload Output File']
                     )
                    ])
    
    upload_html = html.Div([dcc.Upload(
                                id="input-data",
                                children=html.Div(
                                    ["Click to select a .csv file to upload."]
                                ),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px",
                                },
                                multiple=True,
                            )], id = 'upload-html')
    
    load_session_html = html.Div([html.Button('Load Session', id='load-session', n_clicks=0)], id = 'load-session-html')
    #print(load_session)
    loaded = False
    if load_session >0 and load_session_id != "" and type(load_session_id) != type(None):
        temp_UPLOAD_DIRECTORY = "/home/dickinsonq/mimal/" + load_session_id
        if not os.path.exists(temp_UPLOAD_DIRECTORY):
            load_session_html = html.Div([html.Button('Load Session', id='load-session', n_clicks=0),
                                         html.H2("Session ID not Found")], id = 'load-session-html')
        else:
            loaded = True
            session_id = load_session_id
            upload_loc = [session_id]
            UPLOAD_DIRECTORY = "/home/dickinsonq/mimal/" + load_session_id
        
    if (upload_loc == ["Not Generated"] or generate_session > 0) and load_session == 0:
        
        letters = string.ascii_letters
        session_id = ( ''.join(random.choice(letters) for i in range(15)) )
        
    
        UPLOAD_DIRECTORY = "/home/dickinsonq/mimal/" + session_id
        upload_loc = [session_id]
        output_html = html.Div([
                            #html.Div(error, id = 'error'),
                            #html.Div([], id = 'output-html'),
                            #html.Div([], id = 'metabolite-dropdown'),
                            html.Div([], id = 'outputshap-dropdown'),
                            #html.Div([], id = 'umap-button'),
                            
                        ])
        network = html.Div([], id = 'network')
        os.umask(0)
        os.makedirs(UPLOAD_DIRECTORY, mode=0o777)
        new_generate = True
        #print(upload_loc)
    
        
    else:
        #print(upload_loc)
        UPLOAD_DIRECTORY = "/home/dickinsonq/mimal/" + upload_loc[0]
        
       
    
    
    
    if (input_filenames is not None and input_file_contents is not None and new_generate != True):
        #print('upload')
        error = []
        for name, data in zip(input_filenames, input_file_contents):
            if sys.getsizeof(data) <  52428800:
                save_file(name, data, UPLOAD_DIRECTORY)
            else:
                error.append(name+ "is greater than 50mb ")
                output_html = html.Div([
                            html.Div(error, id = 'error'),
                            #html.Div([], id = 'output-html'),
                            #html.Div([], id = 'metabolite-dropdown'),
                            html.Div([], id = 'outputshap-dropdown'),
                            #html.Div([], id = 'umap-button')
                            
                        ])
                
    files = uploaded_files(UPLOAD_DIRECTORY)
    #if files != []:
    input_dropdown_html = html.Div([
                dcc.Dropdown(
                id='input-dropdown', clearable=False,
                value=input_dropdown, options=files
                 )
                ]),
    output_dropdown_html = html.Div([
                dcc.Dropdown(
                id='output-dropdown', clearable=False,
                value=output_dropdown, options=files
                 )
                ])
    if submit_val > 0 and (input_dropdown in files) and (output_dropdown in files) and new_generate != True:
        ##print('validated')
        error = []
        loadsuccess = True
        error = []
        try:
            input_x = pd.read_csv(UPLOAD_DIRECTORY + '/' + input_dropdown, index_col = 0)
        except:
            loadsuccess = False
            error.append('Input X unable to be loaded. Is it a .csv file?')
        try:
            output_y = pd.read_csv(UPLOAD_DIRECTORY + '/' + output_dropdown, index_col = 0)
        except:
            loadsuccess = False
            error.append('Output Y unable to be loaded. Is it a .csv file?')
        
        if loadsuccess == True:
            if list(input_x.index) != list(output_y.index):
                loadsuccess = False
                error.append('Index of Input X and Output Y do not match')
                
            if len(output_y.T) > 1:
                loadsuccess = False
                error.append('Number of output dimensions is greater than 1. This does not support multidimensional output')
                
            if len(input_x.T) < 2:
                loadsuccess = False
                error.append('Number of input dimensions is less than 2. Unable to train with less than 2 dimensions.')
                
                
        if loadsuccess == True:
            if modeltype == 'rand':
                model = RandomForestRegressor(n_estimators=int(numtrees),max_depth=int(treedepth), n_jobs=-1)       

            if modeltype == 'etr':
                model = ExtraTreesRegressor(n_estimators=int(numtrees),max_depth=int(treedepth), n_jobs=-1)

            if modeltype == 'xgb':
                model = xgboost.XGBRegressor(n_estimators=int(numtrees),max_depth=int(treedepth), n_jobs=-1)


            if imput == 'imptrue':
                pf = input_x.T
                kimpp = KNNImputer(n_neighbors=2)
                pfi = kimpp.fit_transform(pf)
                X_data = np.asarray(pfi.T)
                input_x = pd.DataFrame(data = pfi.T, index = pf.T.index, columns = pf.T.columns)
            elif imput == 'impfalse':
                input_x = (input_x.T.dropna()).T

            if normalize == 'normtrue':
                scaler = StandardScaler()
                X_data = scaler.fit_transform(input_x)
                input_x = pd.DataFrame(data = X_data, index = input_x.index, columns = input_x.columns)








            input_train, input_test, output_train, output_test = train_test_split(input_x, output_y, test_size = 0.10)

            ##print(np.array(output_train))
            model.fit(input_train, np.array(output_train))
            output_pred = model.predict(input_test)
            test = np.array(output_test[list(output_test.columns)[0]])
            ##print(np.array(output_pred))
            plotly_fig5 = px.scatter(
                x=np.array(output_pred), y=test,       
                render_mode="webgl"
            )

            plotly_fig5.update_layout(
            title="<br>R2 = " + str(round(sklearn.metrics.r2_score(test, list(output_pred)), 3)),
            xaxis_title="Predicted",
            yaxis_title="True",
            title_x=0.5,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="Black"
                )
            )
            plotly_fig5['data'][0]['hovertemplate'] = '%{text}<br>x=%{x}<br>y=%{y}<extra></extra>'
            plotly_fig5['data'][0]['text'] = np.array(input_x.index)



            plotly_fig5['layout']['height']=500
            plotly_fig5['layout']['width']=500




            explainer = shap.TreeExplainer(model)

            shap_values_calc = explainer.shap_values(input_x)
            ##print(len(shap_values_calc[0]))
            ##print(len(shap_values[:,0]))
            ##print(len(np.array(input_x)))

            force_plot = shap.force_plot(explainer.expected_value, shap_values_calc[0], input_x.T[list(input_x.T.columns)[0]], 
                        show=True, 
                        matplotlib=False,
                       text_rotation=90)


            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"



            #print(imput)
            output_html = html.Div([
                        dcc.Graph(id='graph5', figure = plotly_fig5),
                        html.Div([html.Iframe(srcDoc=shap_html, width = "100%")], id = 'genshapplot'),
                        dcc.Dropdown(id='outputshap-dropdown', clearable=False, value=list(input_x.index)[0], options=list(input_x.index)),
                        'Generate UMAP plot',
                        #html.Div([html.Button('Submit', id='umapbutton', n_clicks=0)], id = 'umapbutton-html'),
                        
                        ])              


            #print('R2: ' + str(r2_score(test, output_pred)))
            
            if not os.path.exists(UPLOAD_DIRECTORY+"/pickle/"):
                os.umask(0)
                os.makedirs(UPLOAD_DIRECTORY+"/pickle/", mode=0o777)
                

            inputfp = UPLOAD_DIRECTORY + '/' + input_dropdown    
            outputfp = UPLOAD_DIRECTORY + '/' + output_dropdown
            variables = [inputfp, outputfp, imput, normalize]
            ##print(filepaths)
            pickle.dump(variables, open(UPLOAD_DIRECTORY+"/pickle/" +"variables.pickle", 'wb'))
            pickle.dump(model, open(UPLOAD_DIRECTORY+"/pickle/" +"model.pickle", 'wb'))
            pickle.dump(shap_values_calc, open(UPLOAD_DIRECTORY+"/pickle/" + '/' +"shap_values.pickle", 'wb'))
            pickle.dump(explainer, open(UPLOAD_DIRECTORY+"/pickle/" +"explainer.pickle", 'wb'))
            pickle.dump(input_test, open(UPLOAD_DIRECTORY+"/pickle/" +"input_test.pickle", 'wb'))
            pickle.dump(output_test, open(UPLOAD_DIRECTORY+"/pickle/" +"output_test.pickle", 'wb'))
            pickle.dump(input_train, open(UPLOAD_DIRECTORY+"/pickle/" +"input_train.pickle", 'wb'))
            pickle.dump(output_train, open(UPLOAD_DIRECTORY+"/pickle/" +"output_train.pickle", 'wb'))
            
            shap_values_output = pd.DataFrame(shap_values_calc)
            shap_values_output.columns = input_x.columns
            shap_values_output.index = input_x.index
            shap_values_output.to_csv(UPLOAD_DIRECTORY+"/shap_values.csv")
            #pickle.dump(etr, open(UPLOAD_DIRECTORY + '/' +"model.pickle", 'wb'))
        else:
            output_html = html.Div([
                            html.Div(error, id = 'error'),
                            #html.Div([], id = 'output-html'),
                            #html.Div([], id = 'metabolite-dropdown'),
                            html.Div([], id = 'outputshap-dropdown'),
                            #html.Div([], id = 'umap-button')
                            
                        ])
    
    if ((outputshap_dropdown != None and submit_val == 0) or loaded == True) and (new_generate != True):
               
        loadsuccess = True
        error = []
        
        variables = pickle.load(open(UPLOAD_DIRECTORY+"/pickle/" +"variables.pickle", 'rb'))
        #print('variables')
        try:
            input_x = pd.read_csv(variables[0], index_col = 0)
        except:
            loadsuccess = False
            error.append('Input X unable to be loaded. Is it a .csv file?')
        try:
            output_y = pd.read_csv(variables[1], index_col = 0)
        except:
            loadsuccess = False
            error.append('Output Y unable to be loaded. Is it a .csv file?')
        
        if loadsuccess == True:
            if list(input_x.index) != list(output_y.index):
                loadsuccess = False
                error.append('Index of Input X and Output Y do not match')
                
            if len(output_y.T) > 1:
                loadsuccess = False
                error.append('Number of output dimensions is greater than 1. This does not support multidimensional output')
                
            if len(input_x.T) < 2:
                loadsuccess = False
                error.append('Number of input dimensions is less than 2. Unable to train with less than 2 dimensions.')
        if loadsuccess == True:
            
            if variables[2] == 'imptrue':
                pf = input_x.T
                kimpp = KNNImputer(n_neighbors=2)
                pfi = kimpp.fit_transform(pf)
                X_data = np.asarray(pfi.T)
                input_x = pd.DataFrame(data = pfi.T, index = pf.T.index, columns = pf.T.columns)
            elif variables[2] == 'impfalse':
                input_x = (input_x.T.dropna()).T

            if variables[3] == 'normtrue':
                scaler = StandardScaler()
                X_data = scaler.fit_transform(input_x)
                input_x = pd.DataFrame(data = X_data, index = input_x.index, columns = input_x.columns)








            model = pickle.load(open(UPLOAD_DIRECTORY+"/pickle/" +"model.pickle", 'rb'))
            shap_values_calc = pickle.load(open(UPLOAD_DIRECTORY+"/pickle/" +"shap_values.pickle", 'rb'))
            explainer = pickle.load(open(UPLOAD_DIRECTORY+"/pickle/" + "explainer.pickle", 'rb'))
            input_test = pickle.load( open(UPLOAD_DIRECTORY+"/pickle/" +"input_test.pickle", 'rb'))
            output_test = pickle.load( open(UPLOAD_DIRECTORY+"/pickle/"  +"output_test.pickle", 'rb'))
            ##print('forceplot')
            if loaded == True:
                outputshap_dropdown = list(input_x.index)[0]
                try:
                    network = pickle.load(open(UPLOAD_DIRECTORY+"/pickle/" +"network.pickle", 'rb'))
                except:
                    network = html.Div([], id = 'network')
            force_plot = shap.force_plot(explainer.expected_value, shap_values_calc[list(input_x.index).index(outputshap_dropdown)], input_x.T[outputshap_dropdown], 
                        show=True, 
                        matplotlib=False,
                       text_rotation=90)

            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            #print('forceplotend')
            output_pred = model.predict(input_test)
            test = np.array(output_test[list(output_test.columns)[0]])
            #print('predicted')
            ##print(np.array(output_pred))
            plotly_fig5 = px.scatter(
                x=np.array(output_pred), y=test,       
                render_mode="webgl"
            )
            #print('scatter')
            plotly_fig5.update_layout(
            title="<br>R2 = " + str(round(sklearn.metrics.r2_score(test, list(output_pred)), 3)),
            xaxis_title="Predicted",
            yaxis_title="True",
            title_x=0.5,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="Black"
                )
            )
            plotly_fig5['data'][0]['hovertemplate'] = '%{text}<br>x=%{x}<br>y=%{y}<extra></extra>'
            plotly_fig5['data'][0]['text'] = np.array(input_x.index)



            plotly_fig5['layout']['height']=500
            plotly_fig5['layout']['width']=500
            ##print(imput_submit)
            output_html = html.Div([
                        dcc.Graph(id='graph5', figure = plotly_fig5),
                        html.Div([html.Iframe(srcDoc=shap_html, width = "100%")], id = 'shapplot'),
                        dcc.Dropdown(id='outputshap-dropdown', clearable=False, value=outputshap_dropdown, options=list(input_x.index)),
                        'Generate UMAP plot',
                        #html.Div([html.Button('Submit', id='umapbutton', n_clicks=0)], id = 'umapbutton-html')
                        
                        ])
            #print('output')
        else:
            output_html = html.Div([
                            html.Div(error, id = 'error'),
                            #html.Div([], id = 'output-html'),
                            #html.Div([], id = 'metabolite-dropdown'),
                            html.Div([], id = 'outputshap-dropdown'),
                            #html.Div([], id = 'umap-button')
                            
                        ])
    try:
        shap_values_calc = pickle.load(open(UPLOAD_DIRECTORY+"/pickle/" +"shap_values.pickle", 'rb'))
        umapbuttonhtml = html.Div([html.Button('Submit', id='umap-button', n_clicks=0)], id = 'umap-button-html')
    except:
        umapbuttonhtml = html.Div([html.Button('Submit', id='umap-button', disabled=True, n_clicks=0)], id = 'umap-button-html')
    
    
    
    
    
    
    if (type(umap_button) != type(None)) and (new_generate != True):
        if umap_button > 0:
            shap_values_calc = pickle.load(open(UPLOAD_DIRECTORY + '/pickle/' +"shap_values.pickle", 'rb'))
            loadsuccess = True
            error = []
            ##print(input_filepath_submit[0])
            ##print(output_filepath_submit[0])
            variables = pickle.load(open(UPLOAD_DIRECTORY+"/pickle/" +"variables.pickle", 'rb'))
            
            try:
                input_x = pd.read_csv(variables[0], index_col = 0)
            except:
                loadsuccess = False
                error.append('Input X unable to be loaded. Is it a .csv file?')
            try:
                output_y = pd.read_csv(variables[1], index_col = 0)
            except:
                loadsuccess = False
                error.append('Output Y unable to be loaded. Is it a .csv file?')

            if loadsuccess == True:
                if list(input_x.index) != list(output_y.index):
                    loadsuccess = False
                    error.append('Index of Input X and Output Y do not match')

                if len(output_y.T) > 1:
                    loadsuccess = False
                    error.append('Number of output dimensions is greater than 1. This does not support multidimensional output')

                if len(input_x.T) < 2:
                    loadsuccess = False
                    error.append('Number of input dimensions is less than 2. Unable to train with less than 2 dimensions.')
            if loadsuccess == True:

                ##print(imput_submit[0])
                if variables[2] == 'imptrue':
                    pf = input_x.T
                    kimpp = KNNImputer(n_neighbors=2)
                    pfi = kimpp.fit_transform(pf)
                    X_data = np.asarray(pfi.T)
                    input_x = pd.DataFrame(data = pfi.T, index = pf.T.index, columns = pf.T.columns)
                elif variables[2] == 'impfalse':
                    input_x = (input_x.T.dropna()).T

                if variables[3] == 'normtrue':
                    scaler = StandardScaler()
                    X_data = scaler.fit_transform(input_x)
                    input_x = pd.DataFrame(data = X_data, index = input_x.index, columns = input_x.columns)



                y = list(input_x.index)
                x = np.array(shap_values_calc)
                #x = shap_values[compound]
                #x = pro_all.values
                iterator = 0
                repeats = 10
                labelarray = []
                nclusters = []
                #x = StandardScaler().fit_transform(x)

                while iterator<repeats:
                    #it_time = time.time()
                    reducer = umap.UMAP(n_components=10, n_neighbors=3, min_dist = 0, metric='manhattan')
                    embedding =  reducer.fit_transform(x)
                    clustering = OPTICS(min_samples=2).fit(reducer.embedding_)
                    labelarray.append(clustering.labels_)
                    ##print("iteration:" + str(iterator) + " Time:" + str(round(time.time()-it_time, 2)) + 
                    #                                                    " Total:" + str(round(time.time()-comptime, 2)), end='\r')
                    iterator+=1

                clustersarray = []

                i = 0
                while i < len(labelarray):
                    j=0
                    temp = []
                    while j < max(labelarray[i])+1:
                        temp.append([])
                        j+=1
                    j = 0
                    while j < len(labelarray[i]):
                        if labelarray[i][j] != -1:
                            temp[int(labelarray[i][j])].append(y[j])
                        j+=1
                    for value in temp:
                        clustersarray.append(value)
                    i+=1

                i = 0
                G = networkx.Graph() 
                while i < len(clustersarray):
                    j = 0

                    while j < len(clustersarray[i]):
                        G.add_node(clustersarray[i][j])
                        j+=1
                    j = 0
                    while j < len(clustersarray[i]):
                        k = 0
                        while k < len(clustersarray[i]):
                            if G.has_edge(clustersarray[i][j], clustersarray[i][k]):
                                # we added this one before, just increase the weight by one
                                G[clustersarray[i][j]][clustersarray[i][k]]['weight'] += 1
                            else:
                                # new edge. add with weight=1
                                G.add_edge(clustersarray[i][j], clustersarray[i][k], weight=1)
                            k+=1
                        j+=1
                    i+=1
                    
                G.remove_edges_from(networkx.selfloop_edges(G))
                cy = networkx.cytoscape_data(G)


                network = html.Div([cyto.Cytoscape(

                            layout={'name': 'cose'},
                            style={'width': '100%', 'height': '800px'},
                            elements=cy['elements'],
                            stylesheet=[

                                 {
                                    'selector': 'node',
                                    'style': {
                                        'label': 'data(id)',
                                        #'background-color': 'mapData(__mclCluster, 1, 34, blue, red)'
                                        #'background-color': "setNodeColorMapping('score', colors=paletteColorBrewerSet3, mapping.type='d')"
                                    }
                                },
                                {
                                    'selector': 'edge',
                                    'style': {
                                        'label': 'data(weight)',
                                        #'background-color': 'mapData(__mclCluster, 1, 34, blue, red)'
                                        #'background-color': "setNodeColorMapping('score', colors=paletteColorBrewerSet3, mapping.type='d')"
                                    }
                                }]
                            
                        )
                    ], id = 'network')
                pickle.dump(network, open(UPLOAD_DIRECTORY+"/pickle/" +"network.pickle", 'wb'))
                
            
    
    
    
    
    
    
    
    
    if len(files) == 0:
        return ([html.Li("No files yet!")], input_dropdown_html, output_dropdown_html,
                buttonhtml, output_html, umapbuttonhtml, network, upload_loc, generate_session_html,
                upload_html, load_session_html)
    else:
        return ([html.Li(file_download_link(filename)) for filename in files], input_dropdown_html, 
                output_dropdown_html, buttonhtml, output_html, umapbuttonhtml,
                network, upload_loc, generate_session_html, upload_html, load_session_html)








# Update the index
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    elif pathname == '/page-4':
        return page_4_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', debug = True)