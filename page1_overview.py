### Import Libraries
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

cwd = os.getcwd()

def show_overview(df):
        ### Set Up the Streamlit App
        # st.set_page_config(page_title = " DZ Inflation", page_icon = ":bar_chart:", layout = "wide")
        st.markdown("##")
        st.subheader("CPI inflation : An Overview")
        
        ### Descriptive statistics above the Plots
        st.markdown("####")
        st.subheader("1- Descriptive Statistics")
        metric_col1, metric_col2, metric_col3, metric_col4= st.columns(4)
        # metric_col1.metric(label="Headline", value=str(round(cpi["Inflation Rate"].iloc[-1],1))+" %")
        st.markdown("####")
        metric_col1.metric(label= "Historical Average" , value=str(df["Inflation Rate"].mean().round(1))+" %")
        metric_col2.metric(label="Max", value=str(df["Inflation Rate"].max())+" %")
        metric_col3.metric(label="Min", value=str(df['Inflation Rate'].min())+" %")
        metric_col4.metric(label="Standard deviation", value=str(df['Inflation Rate'].std().round(1))+" %")

        st.markdown("####")
        st.subheader("2- Exploratory Data Analysis")
        # Plot 1: Inflation Rate (line plot)
        ##### Add traces if checkboxes are activated
        fig1 = go.Figure()
        # if inflation:
        fig1.add_trace(go.Scatter(x=df.index, y=df["Inflation Rate"],mode='lines', name='Inflation Rate'))
        fig1.add_trace(go.Scatter(x=df.index, y=df["Rolling_Mean"], mode='lines',line = dict(dict(color='red',width=2,dash ='dash')),name='Rolling_Mean'))
        fig1.add_trace(go.Scatter(x=df.index, y=df["Rolling_Std"],mode='markers',  marker=dict(color='green',size=3),name='Rolling_Std'))

        fig1.update_layout(
        title="Inflation Rate (Monthly, Year-over-Year)",
        yaxis=dict(title_text="Inflation Rate (%)", titlefont=dict(size=12)),
        # xaxis=dict(title_text="Date", titlefont=dict(size=12)),
        legend=dict(x=0, y=1, traceorder='normal', orientation='v'))


        # Plot 2: Inflation Rate (heatmap)
        # Creat month and year columns
        df['months'] = df.index.month
        df['year'] = df.index.year
        # Define a dictionary to map month numbers to month names
        month_mapping = {
            1: 'January',
            2: 'February',
            3: 'March',
            4: 'April',
            5: 'May',
            6: 'June',
            7: 'July',
            8: 'August',
            9: 'September',
            10: 'October',
            11: 'November',
            12: 'December'
        }
        # Create a new column 'month_name' based on the mapping
        df['month_name'] = df['months'].map(month_mapping)
        # Creat cross table
        tab = df.pivot_table('Inflation Rate','month_name','year')
        # Define the custom order of months
        custom_month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        # Reorder the index based on the custom order of months
        tab = tab.reindex(custom_month_order)
        # Create a Plotly Heatmap
        heatmap_trace = go.Heatmap(
            z=tab.values,
            x=tab.columns,
            y=tab.index,
            colorscale='balance',
            # colorbar=dict(title='DZ_Inflation', tickvals=[-1, 0, 1, 2, 3], ticktext=['', '0', '1', '2', '3+']),
            hoverongaps=False)
        # Create a Plotly Figure
        fig2 = go.Figure(data=[heatmap_trace])
        fig2.update_layout(title="Years vs Months Heatmap")
        plot_col1, plot_col2 = st.columns(2)
        plot_col1.plotly_chart(fig1, use_container_width=True)
        plot_col2.plotly_chart(fig2, use_container_width=True)

        # plot 3 :  Boxplot
        # Create a Plotly Box trace
        box_trace = go.Box(
            # x=df['year'],
            y=df['Inflation Rate'],
            boxpoints='all',  # Display individual points
            jitter=0.3,       # Add jitter for better visibility of points
            pointpos=-1.8,     # Adjust position of points
            marker=dict(color='rgba(0,147,148,0.8)', outliercolor='rgba(219, 64, 82, 0.6)', line=dict(outliercolor='rgba(219, 64, 82, 0.9)', outlierwidth=2)),
            line=dict(color='rgba(0,147,148,0.8)')
        )
        # Create a Plotly Figure
        fig3 = go.Figure(data=[box_trace])
        # Update layout for better appearance
        fig3.update_layout(
            title='Boxplot',
            xaxis=dict(title=''),
            yaxis=dict(title='Inflation rate'),
            showlegend=False  # To hide legend
        )



        # plot 4 : 
        # Create a time series with the 'CPI' column
        ts = pd.Series(df['Inflation Rate'])
        # Calculate autocorrelation directly in Plotly
        autocorr_values = [1] + [ts.autocorr(lag=i) for i in range(1, 13)]
        # Create a Plotly bar chart for autocorrelation with values above each bar
        autocorr_trace = go.Bar(x=list(range(13)), y=autocorr_values,
                                marker=dict(color='goldenrod'), opacity=0.7)
        # Create a Plotly layout
        layout = go.Layout(title='Autocorrelation',
                        xaxis=dict(title='Lag'),
                        yaxis=dict(title='Autocorrelation'))
        # Create a Plotly Figure
        fig4 = go.Figure(data=[autocorr_trace], layout=layout)
        # Display the boxplot using Streamlit
        plot_col3, plot_col4 = st.columns(2)
        plot_col3.plotly_chart(fig3, use_container_width=True)
        plot_col4.plotly_chart(fig4,use_container_width=True)



        # plot 5 : Statistical decomposition
        st.markdown("###")
        st.subheader("3- Statistical Decomposition")
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomp_data = df["Inflation Rate"].dropna()
        from plotly.subplots import make_subplots
        from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
        def plot_seasonal_decompose(result:DecomposeResult, dates:pd.Series=None, title:str="Seasonal Decomposition"):
            x_values = dates if dates is not None else np.arange(len(result.observed))
            return (
                make_subplots(
                    rows=4,
                    cols=1,
                    subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
                )
                .add_trace(
                    go.Scatter(x=x_values, y=result.observed, mode="lines", name='Observed'),
                    row=1,
                    col=1,
                )
                .add_trace(
                    go.Scatter(x=x_values, y=result.trend, mode="lines", name='Trend'),
                    row=2,
                    col=1,
                )
                .add_trace(
                    go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Seasonal'),
                    row=3,
                    col=1,
                )
                .add_trace(
                    go.Scatter(x=x_values, y=result.resid, mode="lines", name='Residual'),
                    row=4,
                    col=1,
                )
                .update_layout(
                    height=900, title=f'<b>{title}</b>', margin={'t':100}, title_x=0.5, showlegend=False
                )
            )
        data = decomp_data
        decomposition = seasonal_decompose(decomp_data, model='additive', period=12)
        fig = plot_seasonal_decompose(decomposition, dates=decomp_data.index)
        st.plotly_chart(fig, use_container_width=True)

