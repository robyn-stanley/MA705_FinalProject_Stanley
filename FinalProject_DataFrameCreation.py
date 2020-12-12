# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:28:43 2020

@author: 17818
"""


import pandas as pd
import scipy.stats as stats
from bs4 import BeautifulSoup
import requests
import numpy as np

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
df_final = df_final.rename(columns={'state': 'State', 'college': 'College', 'county': 'County', 'city': 'City','Average Probability':'Probability'})

df_final.to_csv('df_Final.csv', index=False)
