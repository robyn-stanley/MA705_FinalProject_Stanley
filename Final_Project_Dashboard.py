# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:29:10 2020

@author: 17818
"""

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px


#DASHBOARD PORTION
data_filepath = "C:/Users/17818/OneDrive/Documents/Bentley/Fall 2020/MA705/GroupProject/StatePollDataFrame.csv"
df = pd.read_csv(data_filepath)

stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# pandas dataframe to html table
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], id="my-table")

app = dash.Dash(__name__, external_stylesheets=stylesheet)

server = app.server


fig = px.bar(df_final, x="College", y="Probability", 
             title='Probability of Contracting COVID-19 by College', 
             height=700,
             labels={"College": "College", 
                     "Probability": "Probability (%)"},
             )
fig.update_layout(xaxis_tickangle=-45, title_x=0.5#,xaxis={'categoryorder':'total ascending'}
                  )

app.layout = html.Div([
    html.H1('COVID-19 Risk at United States Colleges', style={'textAlign': 'center'}),
    
    html.Div([
        html.H4('Select State to Display:'),
              dcc.Dropdown(
                  options=[
                      {'label': z,'value': z} for z in sorted(df_final['State'].unique()
                      )],
                  id="state_select_dropdown",
                  multi=True,
                  placeholder='Select a state...',
                  value=[]),
             html.H4('Select College to Display:'),
              dcc.Dropdown(
                  id='colleges_select_dropdown',
                  multi=True,
                  placeholder='Select a college...',
                  value=[]
                  ),
              html.H4('Select City to Display:'),
              dcc.Dropdown(
                  id='city_select_dropdown',
                  placeholder='Select a city...',
                  multi=True,
                  value=[]
                  )],
             style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top' , 'padding-left':'5%'}),
    html.Div([generate_table(df_final)],
             style={'width': '60%', 'display': 'inline-block', 'padding-left': '5%',
                    'overflowY':'scroll','height':'375px'}),
    html.Br(),
    html.Br(),
    html.Div([html.H6('Note: Average probability across all colleges is: ',style={'float':'left','vertical-align': 'top','padding-left':'2%'}),
              html.Var([Average],style={'display':'inline-block','font-size':'20px','padding-left':'1%','vertical-align': '-14px'})],
             style={'border-style':'ridge','width':'500px','height':'50px','margin-left':'600px','background-color':'AliceBlue'}),
    dcc.Graph(id='graph', figure=fig),
    html.Br(),
    html.Div([html.A("COVID Case Count Source", href='https://github.com/nytimes/covid-19-data/blob/master/colleges/colleges.csv')]),
    html.Div([html.A("College Size Source", href='https://www.collegeraptor.com/college-rankings/details/TotalEnrollment/')])
    ]) 

@app.callback(Output(component_id='my-table', component_property='children'),
    [Input(component_id='state_select_dropdown', component_property='value'),
    Input(component_id='colleges_select_dropdown', component_property='value'),
    Input(component_id='city_select_dropdown', component_property='value')]
    )
def update_table(states_selected, colleges_selected, cities_selected):
    filtered_df=df_final
    if(len(states_selected)>0):
        filtered_df=filtered_df[filtered_df['State'].isin(states_selected)]
    if(len(colleges_selected)>0):
        filtered_df=filtered_df[filtered_df['College'].isin(colleges_selected)]
    if(len(cities_selected)>0):
        filtered_df=filtered_df[filtered_df['City'].isin(cities_selected)]
    max_filtered_rows = len(filtered_df)
    return generate_table(filtered_df,max_filtered_rows)

@app.callback(Output(component_id='graph', component_property='figure'),
    [Input(component_id='state_select_dropdown', component_property='value'),
    Input(component_id='colleges_select_dropdown', component_property='value'),
    Input(component_id='city_select_dropdown', component_property='value')]
    )
def update_graph(states_selected, colleges_selected, cities_selected):
    filtered_df=df_final
    if(len(states_selected)>0):
        filtered_df=filtered_df[filtered_df['State'].isin(states_selected)]
    if(len(colleges_selected)>0):
        filtered_df=filtered_df[filtered_df['College'].isin(colleges_selected)]
    if(len(cities_selected)>0):
        filtered_df=filtered_df[filtered_df['City'].isin(cities_selected)]
    #issue filtering due to plotly version https://github.com/plotly/plotly.py/issues/2350
    new_fig = px.bar(filtered_df, x='College', y="Probability",  
             height=700,
             labels={"College": "College", 
                     "Probability": "Probability (%)"},
             )
    new_fig.update_layout(xaxis_tickangle=-45, title_x=0.5#, xaxis={'categoryorder':'total ascending'}
                          )
    return new_fig

#Source for below: https://community.plotly.com/t/dropdown-updating-dropdown/5831/2
@app.callback(Output(component_id='colleges_select_dropdown', component_property='options'),
              [Input(component_id='state_select_dropdown', component_property='value')])
def set_colleges_options(selected_state):
   if(selected_state==[]):
       return [{'label':i, 'value':i} for i in df_final.College]
   filtered_colleges=[]
   for i, row in df_final.iterrows():
       if(row.State in selected_state):
           option={'label':row.College,'value':row.College}
           filtered_colleges.append(option)
   return filtered_colleges   

@app.callback(Output(component_id='city_select_dropdown', component_property='options'),
              [Input(component_id='state_select_dropdown', component_property='value')])
def set_cities_options(selected_state):
   if(selected_state==[]):
       return [{'label':i, 'value':i} for i in df_final.City]
   filtered_cities=[]
   for i, row in df_final.iterrows():
       if(row.State in selected_state):
           option={'label':row.City,'value':row.City}
           filtered_cities.append(option)
   return filtered_cities

if __name__ == '__main__':
    app.run_server(debug=False)  
