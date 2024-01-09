import os
import streamlit as st
import pandas as pd
import streamlit as st
import page1_overview
import page2_forecasting


cwd = os.getcwd()
# Set the page configuration, including the title
st.set_page_config(
    page_icon="📈",
    layout="wide"
)
st.markdown("#")
st.header("Algeria's CPI inflation 2000-2022")
pages = ["Overview", "Forecast"]
    
# Display the selected page
page = st.sidebar.radio("Select a page", pages)

@st.cache_data()
def get_data(start_year,end_year): 
    # Import Data
    df = pd.read_csv('DZ_Consumption_price_index.csv')
    # df = pd.read_csv('/Users/bengherbia/Library/CloudStorage/OneDrive-Personal/Bureau/My_github/Deep-Learning-vs-ARMA-in-forcasting-inflation-/Deep-Learning-vs-ARMA-in-forcasting-inflation--1/streamlit_App/DZ_Consumption_price_index.csv')
    # df = pd.read_csv('https://github.com/hassentchoketch/Deep-Learning-vs-ARMA-in-forcasting-inflation-/blob/master/streamlit_App/DZ_Consumption_price_index.csv')
    df['date'] = pd.to_datetime(df['date'])
    # Set 'date_column' as the index
    df.set_index('date', inplace=True)
    # df = df[df.index.year > start_year-2 ] 
    df = df[(df.index.year > (start_year-1)) & (df.index.year <= (end_year))]

    return df

# Code for selecting periods
years = list(range(2000,2023,1))
start_year = st.sidebar.selectbox(label="Select Start Year", options=years,index=0)
end_year = st.sidebar.selectbox(label ="Select End Year", options=years,index=22)


# Read and Select Data for year
df = get_data(start_year-1,end_year)

# Calculate Inflation Rate
df["Inflation Rate"] = round((df["CPI"] / df["CPI"].shift(12) - 1) * 100,2)

# Calculate rolling mean and standard deviation
window_size = 12  # Adjust the window size as needed
df['Rolling_Mean'] = df['Inflation Rate'].rolling(window=window_size).mean()
df['Rolling_Std'] = df['Inflation Rate'].rolling(window=window_size).std()

if page == "Overview":
   page1_overview.show_overview(df)
if page == 'Forecast':
  page2_forecasting.show_forecasts(df)
