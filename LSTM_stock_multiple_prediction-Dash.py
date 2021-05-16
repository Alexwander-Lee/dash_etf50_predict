#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.graph_objects as go
import plotly.express as px
import plotly
# import pandas_datareader as web
# import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import tensorflow as tf

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import dash_table
import pandas as pd
from jupyter_dash import JupyterDash
# import dash_labs as dl

basic_length=60
prediction_length=10

def prepare_figure_data(excel_docu='', model='', basic_length='', prediction_length=''):
    df = pd.read_excel(
    #     '/home/alex/510050raw_10seconds.xlsx',
    #     '/home/alex/510050raw_minutes.xlsx',
#         '/home/alex/510050raw_day.xlsx', 
                        excel_docu,
                       skipfooter=1,
                      index_col=0, skiprows=3,)
    # model = tf.keras.models.load_model('/home/alex/LSTM_STOCK_minutes_multiple10.h5')
    model = tf.keras.models.load_model (model)
#     model= model
#     N=60 #训练的个数
#     M=10 #预测的个数
    N= basic_length
    M= prediction_length
    
    
    df=df.iloc[:,:5]
    df.columns=['Open', 'High', 'Low', 'Close', 'Volume']


    dataset =df.filter(['Close'])

    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))


    scaled_dataset =scaler.fit_transform(dataset)

    training_length = int(len(dataset)* 0.8)
#     Training_Length = training_length+M
#     data_training = scaled_dataset[:Training_Length]

#     x_training, y_training =[], []

#     for i in range(N, len(data_training)-M):

#         x_training.append(data_training[i-N:i, 0])
#         y_training.append(data_training[i:i+M, 0])

#         if i<=N:
#             print(x_training)
#             print(y_training)


#     x_training, y_training = np.array(x_training), np.array(y_training)

    test_dataset = scaled_dataset[training_length:,:]
    x_test = []
    y_test= []

    for i in range (N, len(test_dataset)-M):
        x_test.append(test_dataset[i-N:i,0])
        y_test.append(dataset[training_length:][i:i+M])


    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

    predictions = model.predict(x_test)

    predictions= scaler.inverse_transform(predictions)

    df_test = dataset[training_length+N:-M]
#     print(df_test.shape, predictions.shape)
    df_test['Prediction'] =predictions.tolist()

    df2=pd.DataFrame()

    predicted, date_predict, date =[], [], []

    for i in range(len(df_test)):

        predicted.append(df_test.iloc[i,1])
        date_predict.append(dataset[training_length+N:].index[i+1:i+M+1].tolist())
        date.append(df_test.index.tolist()[i])

    df2['predicted']=  predicted    
    df2['date_predict']= date_predict         
    df2['date']= date     


    df2['close'] = df_test.Close.to_list()

    df3=df2.copy()
    df3['predicted'] = df3['close'].apply(lambda x:[x] ) + df3['predicted']
    df3['date_predict'] = df3['date'].apply(lambda x:[x]) + df3['date_predict']
    
#     return df3, dataset, training_length, M #df3 for predicted data, dataset for actual data available
    return [df3.to_dict(), dataset.to_dict(), training_length, M] #df3 for predicted data, dataset for actual data available

    
def get_figure(df3='', dataset='', training_length='', basic_length='', prediction_length='', 
               predicted_index_to_show='', historical_data_span=''):    
    i=predicted_index_to_show
    N= basic_length
    M=prediction_length
    s=historical_data_span

    df4= dataset[training_length +i+M+N-s: training_length+ i+M+N+1]
#     df4= dataset[400:1000]
#     print(f'dataset: {dataset}')
#     print(f'df4:{df4}')
    trace0=go.Scatter(x=df4.index, y=df4.Close,  mode='lines', name='Actual Close Price', )

    x=df3.date_predict[i]
    
    y= [round(i,3) for i in df3.predicted[i]]
    trace2 = go.Scatter(x=x, y=y, mode='lines+markers', name='Price Predicted',)

    layout = go.Layout(title='ETF50 Close Price and Prediction', )
    data = [trace0,trace2]
    fig=go.Figure(data=data, layout=layout, )
    dff= pd.DataFrame([x,y, df4.iloc[-M-1:,0]], 
                      index=['Date/Time','Predicted', 'Actual']).reset_index().to_dict('records')
    return fig, dff
    
# df3, dataset, training_length = prepare_figure_data(excel_docu= '/home/alex/510050raw_day.xlsx',
#                          model='/home/alex/LSTM_STOCK_daily_multiple10.h5', 
#                          basic_length=60, prediction_length=10)
# print(dataset)

# fig, x, y = get_figure(df3=df3, dataset=dataset, basic_length=60, prediction_length=10,
#            predicted_index_to_show=150, historical_data_span=300)

# fig.show()
# print(x,y)


# In[2]:


data_daily =prepare_figure_data(
            excel_docu= '/home/alex/510050raw_day.xlsx',
             model='/home/alex/LSTM_STOCK_daily_multiple10.h5', 
             basic_length=60, prediction_length=10)
 
data_1minute = prepare_figure_data(
            excel_docu= '/home/alex/510050raw_minutes.xlsx',
            model='/home/alex/LSTM_STOCK_minutes_multiple10.h5',         
            basic_length=60, prediction_length=10)
                
data_daily=list(data_daily)
data_1minute= list(data_1minute)
# df3, dataset, training_length, prediction_length= data_daily


# In[3]:


# df3, dataset, training_length, prediction_length = data_daily


# In[4]:


# list(data_daily)


# In[5]:


app = JupyterDash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
               meta_tags=[{'name': 'viewport',
                         'content': 'width=device-width, inital-scale=1.0'}])

server= app.server

app.layout= dbc.Container([
    dcc.Interval(
        id='update_time',
        interval=1*1000,
        n_intervals=0,
    ),
    
    dcc.Graph(
        id='graph',
            
    ),
    
    dbc.Row([

        dbc.Col(html.P('Adjust the time span'), width=3,
               ),
        dbc.Col(
        dcc.Slider(id='slider',
                    min=50,
                    max=2000,
                    step=20,
                    value=300,
                    marks={
                        50:'50',
#                         300: {'label': 'Adjust the time span', 'style':{'margin-top': 20}},
                        300:'300',
                        500: '500',
                        1000: '1000',
                        1500: '1500',
                        2000: '2000'
                    }, tooltip={'always_visible': True, 'placement': 'topRight'},
                ), width={'size':6, 'offset': 0}
           ), 
    ], className='bg-info text-white' #no_gutters=False, #align='center', #justify=True
    ),
    
   
    dbc.Row([

        dbc.Col(html.P('Select the Frequency'), width=3, 
               ),
        
        dbc.Col(
            dcc.RadioItems(
                id='freq',
                options=[
                    {'label': 'Daily', 'value': 'D'},
                    {'label': '1 Minute', 'value': 'M'},
                    ],
                value='D',
                labelStyle={'display': 'inline-block'},
                labelClassName='mr-4',
                
            ),  width={'offset': 0},
        ),
    ], className='bg-warning text-white' #no_gutters=False, #align='center', #justify=True
    ),
  
    
    
    
    
    dcc.Store(id='data_dict'),
    html.H6(''),
    html.H6('Below is the predicted closing price:'),
    dash_table.DataTable(id='mytable',
                        data = [{}],
#                         columns = [{'name': i, 'id':i} for i in {}[1]],
                        columns= [{'name': ' ', 'id':'index'}, {'name':'Present Time', 'id': '0'}] 
                         + 
                         [{'name':str(i), 'id': str(i)} for i in range(1,prediction_length)],
                         style_data={'whiteSpace': 'normal', 'height': 'auto'},
                        ),
], #fluid=True
)

# dash_table.DataTable()


# In[6]:


@app.callback(Output('data_dict', 'data'),
             Input('freq', 'value'))
def select_frequency(f):
    if f=='M':
        prepared_data=data_1minute
    else:
        prepared_data=data_daily
#     data_dict= dict(df3=prepared_data[0], dataset=prepared_data[1], training_length=prepared_data[2],
#                prediction_length=prepared_data[3])
#     print(data_dict)
    return prepared_data


# In[7]:


@app.callback([Output(component_id='graph', component_property='figure'),
#               Output(component_id='text', component_property='children'),
               Output('mytable', 'data'),
              ],
             [Input(component_id='update_time', component_property='n_intervals'),
              Input('slider', 'value'),
              Input('data_dict', 'data'),
             ]
             )



def figure_update(i, slider_value, data_dict):
    
#     print(data_dict)
    df3, dataset, training_length, prediction_length = data_dict
    df3= pd.DataFrame(df3)
    dataset=pd.DataFrame(dataset)
#     training_length = data_dict['training_length']
#     prediction_length = data_dict['prediction_length']
    
    fig, dff = get_figure(df3=df3, dataset=dataset, training_length=training_length, basic_length=60, 
        prediction_length=prediction_length, predicted_index_to_show=i, historical_data_span=slider_value)
    
#     dff1=[]
#     status_list=['Date/Time', 'Predicted', 'Actual']
#     for i,j in zip(dff, status_list):
#         dff1.append({**{'status':j}, **i})
    return fig, dff



if __name__ == '__main__':
#     server= app.server()
    app.run_server(mode = 'jupyterlab', debug=True)


# In[8]:


# a= list((1,2,3))
# a


# In[9]:


get_ipython().run_line_magic('tb', '')

