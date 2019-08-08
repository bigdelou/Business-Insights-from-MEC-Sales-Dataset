# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:38:03 2019

@author: mbigdelou
"""

import os
#os.chdir(r'C:\Users\mbigdelou\Desktop\BA Project')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.interactive(False)
import seaborn as sns
import statsmodels.api as sm

#==============Import Dataset
df=pd.read_csv('MEC_Retail.csv')
df.head()

df.shape

df.info()
describ = df.describe()
list(df)

#The objective function in to maxmize profit in mid-run
#A general understanding of the situation

# Q# 1-1) What are the Total and Average Revenue, Cost, Profit of Product lines, Product Type, Product and country? 
df['Product line'].unique()
year = df['Year'].unique()

#['Camping Equipment', 'Personal Accessories', 'Outdoor Protection','Golf Equipment', 'Mountaineering Equipment']
#Total Revenue, Cost, Profit by Product line and country
Productline = pd.DataFrame([])
Productline['RevenueTTL'] = df.groupby('Product line')['Revenue'].sum() 
Productline['CostTTL'] = df.groupby('Product line')['Product cost'].sum() 
Productline['ProfitTTL'] = df.groupby('Product line')['Gross profit'].sum() 
Productline['RevenueAVG'] = df.groupby('Product line')['Revenue'].mean() 
Productline['CostAVG'] = df.groupby('Product line')['Product cost'].mean() 
Productline['ProfitAVG'] = df.groupby('Product line')['Gross profit'].mean() 
Productline = Productline.reset_index()

Retailercountry = pd.DataFrame([])
Retailercountry['RevenueTTL'] = df.groupby('Retailer country')['Revenue'].sum() 
Retailercountry['CostTTL'] = df.groupby('Retailer country')['Product cost'].sum() 
Retailercountry['ProfitTTL'] = df.groupby('Retailer country')['Gross profit'].sum() 
Retailercountry['RevenueAVG'] = df.groupby('Retailer country')['Revenue'].mean() 
Retailercountry['CostAVG'] = df.groupby('Retailer country')['Product cost'].mean() 
Retailercountry['ProfitAVG'] = df.groupby('Retailer country')['Gross profit'].mean() 
Retailercountry = Retailercountry.reset_index()

#Total Revenue, Cost, Profit by Product Type 
Productype = pd.DataFrame([])
Productype['RevenueTTL'] = df.groupby('Product type')['Revenue'].sum() 
Productype['CostTTL'] = df.groupby('Product type')['Product cost'].sum() 
Productype['ProfitTTL'] = df.groupby('Product type')['Gross profit'].sum() 
Productype['RevenueAVG'] = df.groupby('Product type')['Revenue'].mean() 
Productype['CostAVG'] = df.groupby('Product type')['Product cost'].mean() 
Productype['ProfitAVG'] = df.groupby('Product type')['Gross profit'].mean() 
Productype = Productype.reset_index()

ax = sns.barplot("Product type", y="RevenueTTL", data=Productype, ci=None, color="salmon", saturation=.5, )
plt.xticks(rotation='vertical')
plt.show()


# Q# 1-3) What are the Total Revenue, Cost, Profit, highest margin (Unit sale price- unit cost), highest gap bettwen unit price and uni price sold and highest gap between planned revenue and revenue based on product? 
TTlPrdct = pd.DataFrame([])
TTlPrdct['Revenue'] = df.groupby('Product')['Revenue'].sum()
TTlPrdct['Product cost'] = df.groupby('Product')['Product cost'].sum() 
TTlPrdct['Gross profit'] = df.groupby('Product')['Gross profit'].sum() 
TTlPrdct['Mean_Revenue'] = df.groupby('Product')['Revenue'].mean() 
TTlPrdct['Mean_Cost'] = df.groupby('Product')['Product cost'].mean() 
TTlPrdct['Mean_Profit'] = df.groupby('Product')['Gross profit'].mean() 

TTlPrdct['Unit sale price'] = df.groupby('Product')['Unit sale price'].mean()
TTlPrdct['Unit price'] = df.groupby('Product')['Unit price'].mean()
TTlPrdct['Unit cost'] = df.groupby('Product')['Unit cost'].mean()
TTlPrdct['Planned revenue'] = df.groupby('Product')['Planned revenue'].mean()

TTlPrdct = TTlPrdct.reset_index()

TTlPrdct['Profit_Margin'] = TTlPrdct['Unit sale price'] - TTlPrdct['Unit cost']
TTlPrdct['Unit_price_gap'] = TTlPrdct['Unit sale price'] - TTlPrdct['Unit price']
TTlPrdct['Mean_Revenue_gap'] = TTlPrdct['Mean_Revenue'] - TTlPrdct['Planned revenue']
#TTlPrdct = TTlPrdct.sort_values(['Mean_Profit'], ascending=True)




# Q# 1-4) What are the Total Sales of Product lines and their ranks for each year? 

#2015
Data2015 = df[(df.Year == 2015)]
Data2015Total = Data2015.groupby('Product line')['Revenue'].sum() 

#2016
Data2016 = df[(df.Year == 2016)]
Data2016Total = Data2016.groupby('Product line')['Revenue'].sum() 

#2017
Data2017 = df[(df.Year == 2017)]
Data2017.groupby('Product line')['Revenue'].sum() 

#2018
Data2018 = df[(df.Year == 2018)]
Data2018.groupby('Product line')['Revenue'].sum() 

# Q# 2) What are the trends of Sales of each Product lines over the period of data set? 
#['Camping Equipment', 'Personal Accessories', 'Outdoor Protection','Golf Equipment', 'Mountaineering Equipment']
OveralRev = df.groupby('Year')['Revenue'].sum() 
Camping_Equipment = df[(df['Product line'] == 'Camping Equipment')].groupby('Year')['Revenue'].sum() 
Personal_Accessories = df[(df['Product line'] == 'Personal Accessories')].groupby('Year')['Revenue'].sum() 
Outdoor_Protection = df[(df['Product line'] == 'Outdoor Protection')].groupby('Year')['Revenue'].sum() 
Golf_Equipment = df[(df['Product line'] == 'Golf Equipment')].groupby('Year')['Revenue'].sum() 
Mountaineering_Equipment = df[(df['Product line'] == 'Mountaineering Equipment')].groupby('Year')['Revenue'].sum() 

plt.plot(year, OveralRev, color='red', label="Overall_Revenue")
plt.plot(year, Camping_Equipment, color='g', label="Camping_Equipment")
plt.plot(year, Personal_Accessories, color='orange', label="Personal_Accessories")
plt.plot(year, Outdoor_Protection, color='brown', label="Outdoor_Protection")
plt.plot(year, Golf_Equipment, color='blue', label="Golf_Equipment")
plt.plot(year, Mountaineering_Equipment, color='grey', label="Mountaineering_Equipment")
plt.xlabel('Product line')
plt.ylabel('Revenue')
plt.title('Revenue of Each Product line Over Time')
plt.legend(loc=2, borderaxespad=0.)
plt.show()

# Q# 3) Share of each Product Line in Total Revenue of MEC in 2018? 
Total2018Prdctline = Data2018['Revenue'].sum()
Camping_Equipment = Data2018[(Data2018['Product line'] == 'Camping Equipment')]['Revenue'].sum()
Personal_Accessories = Data2018[(Data2018['Product line'] == 'Personal Accessories')]['Revenue'].sum()
Outdoor_Protection = Data2018[(Data2018['Product line'] == 'Outdoor Protection')]['Revenue'].sum()
Golf_Equipment = Data2018[(Data2018['Product line'] == 'Golf Equipment')]['Revenue'].sum()
Mountaineering_Equipment = Data2018[(Data2018['Product line'] == 'Mountaineering Equipment')]['Revenue'].sum()

values = [Camping_Equipment, Personal_Accessories, Outdoor_Protection, Golf_Equipment, Mountaineering_Equipment]
colors = ['y', 'b', 'orange', 'g', 'r']
labels = ['Camping Equipment', 'Personal Accessories', 'Outdoor Protection', 'Golf Equipment', 'Mountaineering Equipment']
explode = (0.0, 0.0, 0.0, 0.0, 0.0)
plt.title('Share of Each Product line of Total Revenue of MEC')
plt.legend(labels,loc=2)
plt.pie(values, colors=colors, labels=labels, explode=explode, autopct='%1.1f%%', counterclock=True, shadow=True)
plt.show()



# Q# 4)  "Product line" BCG Matrix for the latest year? 
#Share
Total2018Prdctline = Data2018['Revenue'].sum()
Productlinerev2018 = pd.DataFrame(Data2018.groupby('Product line')['Revenue'].sum())
Productlinerev2018 = Productlinerev2018.reset_index()
Productlinerev2018['Share'] = Productlinerev2018['Revenue'] / Total2018Prdctline

#Growth Rate
#2017
Revenuetprdctline2017 = pd.DataFrame(Data2017.groupby('Product line')['Revenue'].sum())
Revenuetprdctline2017 = Revenuetprdctline2017.reset_index()
#Growth 17->18
Productlinerev2018 = Productlinerev2018.merge(Revenuetprdctline2017, on='Product line', how='outer', suffixes=('2018', '2017'))
Productlinerev2018['Growth_Rate'] = (Productlinerev2018['Revenue2018']-Productlinerev2018['Revenue2017'])/Productlinerev2018['Revenue2017']


ax = sns.scatterplot(x='Share', y='Growth_Rate', hue='Product line', size="Revenue2018",sizes=(20, 1500), 
                     hue_norm=(0, 7), alpha=.6 , palette='Set1', legend=False, data=Productlinerev2018)
#plt.scatter(x='Share', y='Growth_Rate', alpha=.6 , data=Productlinerev2018)
label_point(Productlinerev2018['Share'], Productlinerev2018['Growth_Rate'], Productlinerev2018['Product line'], plt.gca()) 
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.title('BCG Matrix for Product Line of MEC')
plt.xlabel('Share from Total Revenue %')
plt.ylabel('Growth Rate %')
plt.show()

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+0.00001, point['y'], str(point['val']))


# Q# 5) Howmany Product types, Products and Retailer countries are in the dataset?
df['Product type'].unique()
df['Product'].unique()
df['Retailer country'].unique()


# Q# 6) What are the Top Selling product overall and in 2018?
#Overal
dfs = df.copy()
productbestseller = pd.DataFrame(dfs.groupby('Product')['Revenue'].sum())
productbestseller = productbestseller.reset_index()
BestSellerTotal= productbestseller.sort_values(['Revenue'], ascending=False)
BestSellerTotal.head(10)

#2018
productbestseller2018 = pd.DataFrame(Data2018.groupby('Product')['Revenue'].sum())
productbestseller2018 = productbestseller2018.reset_index()
productbestseller2018 = productbestseller2018.sort_values(['Revenue'], ascending=False)

# Q# 6) What are the Top Selling country overall and in 2018?
dfs = df.copy()
#Overal
Bestsellercountries = pd.DataFrame(dfs.groupby('Retailer country')['Revenue'].sum())
Bestsellercountries = Bestsellercountries.reset_index()
Bestsellercountries = Bestsellercountries.sort_values(['Revenue'], ascending=False)
#2018
Bestsellercountries2018 = pd.DataFrame(Data2018.groupby('Retailer country')['Revenue'].sum())
Bestsellercountries2018 = Bestsellercountries2018.reset_index()
Bestsellercountries2018 = Bestsellercountries2018.sort_values(['Revenue'], ascending=False)


# Q# 7) Share of each country from total revenue in 2018
United_States = Data2018[(Data2018['Retailer country'] == 'United States')]['Revenue'].sum()
Canada = Data2018[(Data2018['Retailer country'] == 'Canada')]['Revenue'].sum()
Mexico = Data2018[(Data2018['Retailer country'] == 'Mexico')]['Revenue'].sum()
Brazil = Data2018[(Data2018['Retailer country'] == 'Brazil')]['Revenue'].sum()
Japan = Data2018[(Data2018['Retailer country'] == 'Japan')]['Revenue'].sum()
Korea = Data2018[(Data2018['Retailer country'] == 'Korea')]['Revenue'].sum()
China = Data2018[(Data2018['Retailer country'] == 'China')]['Revenue'].sum()
Singapore = Data2018[(Data2018['Retailer country'] == 'Singapore')]['Revenue'].sum()
Australia = Data2018[(Data2018['Retailer country'] == 'Australia')]['Revenue'].sum()
Netherlands = Data2018[(Data2018['Retailer country'] == 'Netherlands')]['Revenue'].sum()
Sweden = Data2018[(Data2018['Retailer country'] == 'Sweden')]['Revenue'].sum()
Finland = Data2018[(Data2018['Retailer country'] == 'Finland')]['Revenue'].sum()
Denmark = Data2018[(Data2018['Retailer country'] == 'Denmark')]['Revenue'].sum()
France = Data2018[(Data2018['Retailer country'] == 'France')]['Revenue'].sum()
Germany = Data2018[(Data2018['Retailer country'] == 'Germany')]['Revenue'].sum()
United_Kingdom = Data2018[(Data2018['Retailer country'] == 'United Kingdom')]['Revenue'].sum()
Belgium = Data2018[(Data2018['Retailer country'] == 'Belgium')]['Revenue'].sum()
Switzerland = Data2018[(Data2018['Retailer country'] == 'Switzerland')]['Revenue'].sum()
Austria = Data2018[(Data2018['Retailer country'] == 'Austria')]['Revenue'].sum()
Italy = Data2018[(Data2018['Retailer country'] == 'Italy')]['Revenue'].sum()
Spain = Data2018[(Data2018['Retailer country'] == 'Spain')]['Revenue'].sum()

values = [United_States, Canada, Mexico, Brazil, Japan, Korea, China, Singapore, Australia, Netherlands, Sweden, Finland, Denmark, France, Germany, United_Kingdom, Belgium, Switzerland, Austria, Italy, Spain]
labels = ['United States', 'Canada', 'Mexico', 'Brazil', 'Japan', 'Korea', 'China', 'Singapore', 'Australia', 'Netherlands', 'Sweden','Finland', 'Denmark', 'France', 'Germany', 'United Kingdom', 'Belgium', 'Switzerland', 'Austria', 'Italy', 'Spain']
plt.title('Share of Each Retailer country of Total Revenue of MEC')
plt.legend(labels,loc=2)
plt.pie(values, labels=labels, autopct='%1.1f%%', counterclock=True, shadow=False)
plt.show()

#2017
Revenuecountries2017 = pd.DataFrame(Data2017.groupby('Retailer country')['Revenue'].sum())
Revenuecountries2017 = Revenuecountries2017.reset_index()

#2018
Revenuecountries2018 = pd.DataFrame(Data2018.groupby('Retailer country')['Revenue'].sum())
Revenuecountries2018 = Revenuecountries2018.reset_index()

#Growth 17->18
Revenuecountries2018 = Revenuecountries2018.merge(Revenuecountries2017, on='Retailer country', how='outer', suffixes=('2018', '2017'))
Revenuecountries2018['Growth_Rate'] = (Revenuecountries2018['Revenue2018']-Revenuecountries2018['Revenue2017'])/Revenuecountries2018['Revenue2017']
Total2018Prdctline = Data2018['Revenue'].sum()
Revenuecountries2018['Share'] = Revenuecountries2018['Revenue2018'] / Total2018Prdctline

plt.figure(figsize=(15,10))
ax = sns.scatterplot(x='Share', y='Growth_Rate', hue='Retailer country', size="Revenue2018",sizes=(20, 1500), hue_norm=(0, 60), alpha=.6 , palette='Set1', legend=False, data=Revenuecountries2018)
#plt.scatter(x='Share', y='Growth_Rate', alpha=.6 , data=Revenuecountries2018)
label_point(Revenuecountries2018['Share'], Revenuecountries2018['Growth_Rate'], Revenuecountries2018['Retailer country'], plt.gca()) 
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.title('BCG Matrix for Retailer country of MEC')
plt.xlabel('Share from Total Revenue %')
plt.ylabel('Growth Rate %')
plt.show()


# Q# 7) Trend of Gross Profit for each country over time?
#2015
GProfit2015 = pd.DataFrame(Data2015.groupby('Retailer country')['Gross profit'].sum())
GProfit2015 = GProfit2015.reset_index()

#2016
GProfit2016 = pd.DataFrame(Data2016.groupby('Retailer country')['Gross profit'].sum())
GProfit2016 = GProfit2016.reset_index()

#2017
GProfit2017 = pd.DataFrame(Data2017.groupby('Retailer country')['Gross profit'].sum())
GProfit2017 = GProfit2017.reset_index()

#2018
GProfit2018 = pd.DataFrame(Data2018.groupby('Retailer country')['Gross profit'].sum())
GProfit2018 = GProfit2018.reset_index()

#Gross Profit
Grossprofit = GProfit2015.merge(GProfit2016,on='Retailer country').merge(GProfit2017,on='Retailer country').merge(GProfit2018,on='Retailer country')
Grossprofit.columns = ['Country','GrossProfit2015','GrossProfit2016', 'GrossProfit2017', 'GrossProfit2018']
Grossprofit = Grossprofit.sort_values(['GrossProfit2016'], ascending=True)


# Q# 8) Trend of Revenue for each country over time?
#2015
CtryRevenue2015 = pd.DataFrame(Data2015.groupby('Retailer country')['Revenue'].sum())
CtryRevenue2015 = CtryRevenue2015.reset_index()

#2016
CtryRevenue2016 = pd.DataFrame(Data2016.groupby('Retailer country')['Revenue'].sum())
CtryRevenue2016 = CtryRevenue2016.reset_index()

#2017
CtryRevenue2017 = pd.DataFrame(Data2017.groupby('Retailer country')['Revenue'].sum())
CtryRevenue2017 = CtryRevenue2017.reset_index()

#2018
CtryRevenue2018 = pd.DataFrame(Data2018.groupby('Retailer country')['Revenue'].sum())
CtryRevenue2018 = CtryRevenue2018.reset_index()

#Gross Profit
CtryRevenue = CtryRevenue2015.merge(CtryRevenue2016,on='Retailer country').merge(CtryRevenue2017,on='Retailer country').merge(CtryRevenue2018,on='Retailer country')
CtryRevenue.columns = ['Country','Revenue15','Revenue16', 'Revenue17', 'Revenue18']
CtryRevenue = CtryRevenue.sort_values(['GrossProfit2016'], ascending=True)

#Annual Growth Rate
CtryRevenue['Growth_Rate16'] = (CtryRevenue['Revenue16']-CtryRevenue['Revenue15'])/CtryRevenue['Revenue15']
CtryRevenue['Growth_Rate17'] = (CtryRevenue['Revenue17']-CtryRevenue['Revenue16'])/CtryRevenue['Revenue16']
CtryRevenue['Growth_Rate18'] = (CtryRevenue['Revenue18']-CtryRevenue['Revenue17'])/CtryRevenue['Revenue17']


# Q# 9) look deep into Denmark?
dfs = df.copy()
productbestseller = pd.DataFrame(dfs.groupby('Product')['Revenue'].sum())
Denmarkprdct = Data2018[(Data2018['Retailer country'] == 'Denmark')].groupby('Product')['Revenue'].sum()
Denmarkprdct = Denmarkprdct.reset_index()
Denmarkprdct= Denmarkprdct.sort_values(['Revenue'], ascending=True)

#Denmark 2018
Denmark2018 = Data2018[(Data2018['Retailer country'] == 'Denmark')]

#Denmark 2018 by Product type
Denmarkprdct2018 = pd.DataFrame([])
Denmarkprdct2018['Revenue_ttl'] = Denmark2018.groupby('Product type')['Revenue'].sum() 
Denmarkprdct2018['Revenue_avg'] = Denmark2018.groupby('Product type')['Revenue'].mean() 

Denmarkprdct2018['Gross_profit_ttl'] = Denmark2018.groupby('Product type')['Gross profit'].sum()
Denmarkprdct2018['Gross_profit_avg'] = Denmark2018.groupby('Product type')['Gross profit'].mean()
Denmarkprdct2018['Product_cost_ttl'] = Denmark2018.groupby('Product type')['Product cost'].sum()
Denmarkprdct2018['Product_cost_avg'] = Denmark2018.groupby('Product type')['Product cost'].mean()

Denmarkprdct2018['Unit_sale_price'] = Denmark2018.groupby('Product type')['Unit sale price'].mean()
Denmarkprdct2018['Unit_price'] = Denmark2018.groupby('Product type')['Unit price'].mean()
Denmarkprdct2018['Unit_cost'] = Denmark2018.groupby('Product type')['Unit cost'].mean()
Denmarkprdct2018['Planned_revenue'] = Denmark2018.groupby('Product type')['Planned revenue'].mean()

Denmarkprdct2018 = Denmarkprdct2018.reset_index()

Denmarkprdct2018['Profit_Margin'] = Denmarkprdct2018['Unit_sale_price'] - Denmarkprdct2018['Unit_cost']
Denmarkprdct2018['Unit_price_gap'] = Denmarkprdct2018['Unit_sale_price'] - Denmarkprdct2018['Unit_price']
Denmarkprdct2018['Mean_Revenue_gap'] = Denmarkprdct2018['Revenue_avg'] - Denmarkprdct2018['Planned_revenue']

Denmarkprdct2018sorted = Denmarkprdct2018.sort_values(['Mean_Revenue_gap'], ascending=True)


#Denmark 2017
Denmark2017 = Data2017[(Data2017['Retailer country'] == 'Denmark')]
Denmarkprdct2017 = pd.DataFrame([])
Denmarkprdct2017['Revenue_ttl2017'] = Denmark2017.groupby('Product type')['Revenue'].sum() 
Denmarkprdct2017 = Denmarkprdct2017.reset_index()

#2018
DenmarkBCG2018 = pd.DataFrame(Denmark2018.groupby('Product type')['Revenue'].sum())
DenmarkBCG2018 = DenmarkBCG2018.reset_index()

#Growth 17->18
DenmarkBCG2018 = DenmarkBCG2018.merge(Denmarkprdct2017, on='Product type', how='outer')
DenmarkBCG2018['Growth_Rate'] = (DenmarkBCG2018['Revenue']-DenmarkBCG2018['Revenue_ttl2017'])/DenmarkBCG2018['Revenue_ttl2017']
Total2018PrdctDenmark = Denmark2018['Revenue'].sum()
DenmarkBCG2018['Share'] = DenmarkBCG2018['Revenue'] / Total2018PrdctDenmark
DenmarkBCG2018['Share'].sum()

plt.figure(figsize=(15,10))
ax = sns.scatterplot(x='Share', y='Growth_Rate', hue='Product type', size="Revenue",sizes=(20, 1500), hue_norm=(0, 60), alpha=.6 , palette='Set1', legend=False, data=DenmarkBCG2018)
#plt.scatter(x='Share', y='Growth_Rate', alpha=.6 , data=DenmarkBCG2018)
label_point(DenmarkBCG2018['Share'], DenmarkBCG2018['Growth_Rate'], DenmarkBCG2018['Product type'], plt.gca()) 
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.title('BCG Matrix for Product type of MEC in Denmark')
plt.xlabel('Share from Total Revenue %')
plt.ylabel('Growth Rate %')
plt.show()
