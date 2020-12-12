# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:01:53 2020

@author: 17818
"""

import pandas as pd
import scipy.stats as stats
from bs4 import BeautifulSoup
import requests
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

#DATA IMPORT AND CLEANING PORTION

# file path of college data txt
data_filepath = "C:/Users/17818/OneDrive/Documents/Bentley/Fall 2020/MA705/FinalProject/CollegeCovidCases.txt"

original_headers=['date', 'state', 'county', 'city', 'ipeds_id', 'college', 'cases', 'notes']

#read in txt data
college_data=[]
with open(data_filepath) as f:
    for line in f:
        data = line.split("\t")
        college_data.append(data)
        
# creating df from data list and headers
df1 = pd.DataFrame(college_data, columns=original_headers)

#remove colleges with no case data
df1 = df1[df1['cases'] != ' ']

#remove colleges with a note indicating no data on double counting cases
df1 =df1[ df1['notes'] == '\n']

#convert cases to int data type
df1['cases'] = pd.to_numeric(df1['cases'])
#source: https://stackoverflow.com/questions/15891038/change-column-type-in-pandas
    
print(df1)

# web scraping for college sizes
site = "https://www.stateuniversity.com/rank/tot_enroll_rank.html"
headers = {'User-Agent': 'Safari'}

#generate list of all links
all_links=[]
for page_num in range(25):
    all_links.append( "https://www.stateuniversity.com/rank/tot_enroll_rank/" + str(page_num+1))  

#Initialize dictionary
college_dict={}

#iterate over all links
for link in all_links:
    link_resp = requests.get(link, headers=headers)
    link_resp.raise_for_status()
    link_soup = BeautifulSoup(link_resp.text, 'html.parser')
    college_table = link_soup.select('tr')
    #iterate over all rows    
    i=2
    nrow=21
    while i <=nrow: 
        college_name=college_table[i].select('td')[2].text
        college_size=college_table[i].select('td')[3].text
        college_size = int(college_size.replace(',',''))
        college_dict[college_name] = college_size
        i+=1        
print(college_dict)

   
#create df of college sizes
df2 = pd.DataFrame(list(college_dict.items()), columns = ['college', 'size'])
print(df2)
#source: https://datatofish.com/dictionary-to-dataframe/

#merge the sources
df_final = pd.merge(left=df1, right=df2, left_on='college', right_on='college')
print(df_final)
#source: https://datatofish.com/dictionary-to-dataframe/

#Add probability of getting covid to dataframe
df_final['p-hat'] = df_final['cases'] / df_final['size']

#ANALYSIS PORTION

#set number of simulations
simulations = 100

# create results dataframe
results = pd.DataFrame(index=df_final['college'])

#remove duplicates after noticing error later on
#print(results[results.index.duplicated()])
#source: https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean
results = results[~results.index.duplicated()]

#loop through multiple simulations
for i in range(simulations):
    #create dataframe to store results
    sim_column = ("Trial " + str(i + 1))
    results.insert(i, sim_column, '') 
    df_final_colleges = list(df_final['college'])
    #get p_hat and sd for each college   
    for college in results.index:
        index_college = df_final_colleges.index(college)
        p_hat = df_final['p-hat'][index_college]
        #per source below the false positive rate of covid test ranges from 2% to 37%. I assumed a uniform distribution
        #source: https://www.health.harvard.edu/blog/which-test-is-best-for-covid-19-2020081020734#:~:text=The%20reported%20rate%20of%20false%20negative%20results%20is%20as%20high,a%20more%20accurate%20antigen%20test.
        sd = np.random.uniform(low=.02, high=.37)    
        #create normal distribution and simulate once
        covid_curve = stats.norm(p_hat, sd)
        covid_prob = covid_curve.rvs(1)
        #add probability to results dataframe
        results.at[college,sim_column] = covid_prob
print(results)

#take average for each row
results['Average Probability'] = (results.mean(axis=1))

#set negative results to 0
Avg_prob = results['Average Probability']
neg_row = []
for i in range(len(Avg_prob)):
    if Avg_prob[i] < 0:
        neg_row.append(i)

for row in neg_row:
    Avg_prob[row]=0

results['Average Probability'] = Avg_prob

#remove nan values
results.dropna()

#add Average Probability column to df_final
df_final = pd.merge(left=df_final, right=results, left_on='college', right_on='college')
headers=['state', 'college','county', 'city', 'Average Probability']
df_final = df_final[headers] 
Average=df_final['Average Probability'].mean()
Average=round(Average*100,2)
Average=str(Average)+'%'
print(Average)   
df_final['Average Probability'] = round(df_final['Average Probability']*100,2).astype(str) + '%'
df_final = df_final.rename(columns={'state': 'State', 'college': 'College', 'county': 'County', 'city': 'City','Average Probability':'Probability'})


#DASHBOARD PORTION

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
