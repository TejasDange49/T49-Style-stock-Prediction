#import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta

from plotly.graph_objs.layout import yaxis
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Title
app_name = 'T49 Style Stock Market App'
st.write(app_name)
st.subheader('This app is created to predict the stock market prices of the selected company.')
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Take input from user

# sidebar
st.sidebar.header('Select the parameters from below')

start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))

#add ticker symbol list
ticker_list=["AAPL","A","MSFT","GOOG","META","TSLA","NVDA","ADBE","PYPL","IRCTC","INTC","CHCSA","NFLX","PEP","PEP"," AACG","NFL","AACI","AACT","XOM","GE","MSFT","BP","C","PG","WMT","PFE","HBC","TM","JNJ","BAC","AIJ","TOT","NVS","MO","GSK","MTU","JPM","RDS/A","CVX","SNY","VOD","INTC","IBM","E","CSCO","BRK/A","UBS","WFC","T","RDS/B","KO","CHL","PEP","VZ","COP","DNA","AMGN","STD","HPQ","HD","WB","SI","NOK","UNH","TWX","QCOM","ING","DCM","EON","AZN","MRK","SLB","TEF","MBT","NTT","BHP","CSR","BCS","DELL","MER","BBV","MDT","ABT","DT","AXP","MS","SAP","S","ORCL","LLY","AZ","GS","AXA","WYE","STO","EBAY","PBR","UTX","FTE","AAUK","DCX","DB","ERICY","MOT","FNM","BA","MMM","SPY","RTP","USB","CAJ","TSM","MC","TYC","ABN","HMC","EN","BLS","LYG","NSANY","RY","LOW","SNE","YHOO","DIS","MFC","TGT","TEM","BTI","UPS","WLP","TXN","SZE","FRE","BBL","CAT","BMY","PHG","MCD","WAG","DEO","ECA","BF","AMX","WM","NAB","HAL","DOW","UN","BRG","PBR/A","OXY","ACL","BNS","VLO","RIO","NMR","EXC","PRU","MET","TD","CMCSA","LEH","TI","GLW","SU","V","DD","CEO","ANZ","NWS/A","FDC","MLEA","BR","ALL","CNQ","REP","CCL","WBK","HON","BUD","EMC","NHY","EMR","STA","AMAT","BT","CAH","FDX","BAY","UL","DVN","ELE","BMO","BNI","DA","VIA/B","LMT","KEP","CL","SGP","MRO","NGG","GILD","SSL","AET","AA","TEVA","KB","RIG","KMB","BHI","DUK","D","STI","SO","AEG","IMI","NEM","BEN","APC","ADP","APA"]
ticker = st.sidebar.selectbox('Select the company',ticker_list)


#fetch data from user input yfinance laibrary
data = yf.download(ticker, start=start_date,end=end_date)
#add Date as a column to the Dataframe
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True, inplace=True)
st.write('data from', start_date,'To', end_date)
st.write(data)



#plot the data
st.header('Data visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date',y=data.columns, title='closing price of the stock',width=900,height=700)
st.plotly_chart(fig)

#add a select box to select column from data
column = st.selectbox('Select the column to be used for casting',data.columns[1:])

# subsetting the data
data = data[['Date', column]]
st.write("Selected data")
st.write(data)

#ADF test check stationarity
st.header('It data Stationarity?')
st.write('**Note:** If p-value is less than 0.05 , then data is stationary')
st.write(adfuller(data[column])[1] < 0.05)
#lets decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive',period=12)
st.write(decomposition.plot())
#make a same polo in ploaty
st.write("## Plotting the decomposition in plotly ")
st.plotly_chart(px.line(x=data["Date"],y=decomposition.trend, title='Trend',width=900,height=420,labels={'x','Date','y','Price'}).update_traces(line_color='blue'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.seasonal, title='Seasonality',width=900,height=420,labels={'x','Date','y','Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.resid, title='Resid',width=900,height=420,labels={'x','Date','y','Price'}).update_traces(line_color='Red' , line_dash='dot'))

#lets run the model
#user input for three parameters of the model and seasonal order
p = st.slider('Select the value of p', 0, 5, 2)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 2)
seasonal_order = st.number_input('Select the value of seasonal p',0, 24, 12)

model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

#print model summary
st.header('Model Summary')
st.write(model.summary())
st.write("---")


# predict the future values
forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
#predict the future values
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period)
predictions = predictions.predicted_mean


#add index to the predication
predictions.index=pd.date_range(start=end_date,periods=len(predictions),freq='D')
predictions=pd.DataFrame(predictions)
predictions.insert(0,'Date',predictions.index)
predictions.reset_index(drop=True,inplace=True)
st.write("## Predictions",predictions)
st.write("## Actual Data",data)
st.write("---")

#let ploat the data
fig=go.Figure()
#Actual data to yhe ploat
fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines',name='Actual',line=dict(color='blue')))
#add predicted data to the ploat
fig.add_trace(go.Scatter(x=predictions["Date"],y=predictions["predicted_mean"],mode='lines',name='predicted',line=dict(color='red')))
#set the title and axis labels
fig.update_layout(title='Actual VS Predicted', xaxis_title='Date', yaxis_title='Price', width=900, height=500)
#display the ploat
st.plotly_chart(fig)

## add button to show and hide the seprete ploats
show_plots=False

if st.button('Show Separate Plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"],y=data[column],title='Actual',width=900,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='green'))
        st.write(px.line(x=predictions["Date"],y=predictions["predicted_mean"],title="predicted",width=900,height=400,labels={'x','Date','y','Price'}).update_traces(line_color='Red' , line_dash='dot'))
        show_plots=True
    else:
        show_plots=False
#add hide plots button
hide_plots=False
if st.button("Hide Separate Plots"):
    if not hide_plots:
        hide_plots=True
    else:
        hide_plots=False

st.write("---")


st.write("### About Devloper---")


st.write("## Connect with me on social media")
#add links to my social media
# urls of the images
linkedin_url = "https://img.icons8.com/color/48/000000/linkedin.png"
github_url = "https://img.icons8.com/fluent/48/000000/github.png"
youtube_arl = "https://img.icons8.com/color/48/000000/youtube-play.png"
twitter_url = "https://img.icons8.com/color/48/000000/twitter.png"
facebook_url = "https://img.icons8.com/color/48/000000/facebook-new.png"
instagram_url = "https://img.icons8.com/fluent/48/000000/instagram-new.png"
tiktok_url = "https://img.icons8.com/color/48/000000/tiktok.png"


# redirect urls
linkedin_redirect_url = "https://www.linkedin.com/tejas-dange-2a4b76241/"
github_redirect_url = "https://github.com/TejasDange49"
instagram_redirect_url = "https://www.instagram.com/groups/codanics/tejas_dange_49/"

#add links to the images
st.markdown (f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>'
             f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>'
              f'<a href="{instagram_redirect_url}"><img src="{instagram_url}" width="60" height="60"></a>', unsafe_allow_html=True)

st.write("### Gmail:- dangetejas494@gmail.com")
st.write("### Mobile no:-  +91 9022202938")
st.write("### Education:-  MCA")
st.write("### Location :- TAE Collge Pune")






