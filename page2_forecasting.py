### Import Libraries
import os
import pandas as pd
import numpy as np

import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
import keras

cwd = os.getcwd()

def models_loader (model_path): 
#   model_path = input("Pleas enter the model file name:")
  model = keras.models.load_model(model_path,compile=False )
  return model

def model_forecast(model, series, window_size=12, batch_size=30):

    """Uses an input model to generate predictions on data windows

    Args:
      model (TF Keras Model) - model that accepts data windows
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the window
      batch_size (int) - the batch size

    Returns:
      forecast (numpy array) - array containing predictions
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    # Get predictions on the entire dataset
    forecast = model.predict(dataset)

    return forecast.squeeze()

def loss_comp(y_pred, y_truth, loss= 'mse'):
    """
    y_pred: predicted values.
    y_truth: ground truth.
    loss_func: the function used to calculate loss.
    return loss value.
    """
    if loss == "mse":
        return keras.metrics.mean_squared_error(y_pred, y_truth).numpy()
        # mean_squared_error(y_truth, y_pred)
    elif loss == "mae":
        return keras.metrics.mean_absolute_error(y_pred, y_truth).numpy()
        # r2_score(y_truth, y_pred)
    else:
        if loss != "rmse":
            print(
                "The loss functin is illegal. Turn to default loss function: mse"
            )
        return keras.metrics.RootMeanSquaredError(y_pred, y_truth).numpy

def show_forecasts(df):
    st.markdown("##")
    st.subheader("Forecast using Deep Neural Network Methods")
    types = ['Within_Sample_Forecast', 'Out_of_Sample_Forecast','Future_Forecast']
    forecast_type = st.sidebar.selectbox(label="Forecast Type", options=types, index=0)
    models = ['MPL', 'NN', 'simpl_RNN', 'LSTM', 'BI_LSTM', 'CNN']
    model_ = st.sidebar.selectbox(label="Select Model", options=models, index=0)
    
    # model = models_loader(f'/Users/bengherbia/Library/CloudStorage/OneDrive-Personal/Bureau/My_github/Deep-Learning-vs-ARMA-in-forcasting-inflation-/Deep-Learning-vs-ARMA-in-forcasting-inflation--1/streamlit_App/models/{model_}_model.h5') 
    model = models_loader(f'https://github.com/hassentchoketch/Deep-Learning-vs-ARMA-in-forcasting-inflation-/tree/master/streamlit_App/models/{model_}_model.h5') 
  

    if forecast_type == 'Within_Sample_Forecast':
        forecasts = model_forecast(model=model, series=df['Inflation Rate'].loc[:'2015-12-01'].values)
        date = pd.to_datetime(df['Inflation Rate'].loc[:'2015-12-01'].index)[12 - 1:]
        actual = df['Inflation Rate'].loc[:'2015-12-01'][12 - 1:]
        loss = loss_comp(forecasts, actual.values.reshape(-1))
       
    if forecast_type == 'Out_of_Sample_Forecast':
        forecasts = model_forecast(model=model, series=df['Inflation Rate'].loc['2015-12-01':].values)
        date = pd.to_datetime(df['Inflation Rate'].loc['2015-12-01':].index)[12 - 1:]
        actual = df['Inflation Rate'].loc['2015-12-01':][12 - 1:]
        loss = loss_comp(forecasts, actual.values.reshape(-1))
     
    if forecast_type == 'Future_Forecast':
        num_periods = st.number_input("Number of Periods:", step=1, value=12, format="%d")
        
        # Generate new dates with 0 values
        last_date = df.index[-1]
        new_dates = pd.date_range(last_date + pd.DateOffset(1), periods=num_periods, freq='M')
        
        # Create a DataFrame with 0 values for the new dates
        new_df = pd.DataFrame(np.nan, index=new_dates, columns=df.columns)
        
        # Concatenate the original DataFrame and the new DataFrame
        new_df = pd.concat([df, new_df])
        new_df['forecasts'] =  new_df['Inflation Rate'] 
        for i in range(len(df), len(new_df)):
            # Forecast using the last 12 values
            forecast = model_forecast(model=model, series=new_df.iloc[i-12:i, 4].values)
            # Update the value
            new_df.loc[new_df.index[i],'forecasts'] = forecast
            
        # Get the forecasts, actual values, and dates
        date = pd.to_datetime(new_df['Inflation Rate'].index)
        actual = new_df['Inflation Rate']
        
        # Replace the first 10 values in 'Values' column with None
        new_df.iloc[:len(df), new_df.columns.get_loc('forecasts')] = np.nan
        forecasts = new_df['forecasts']
        # Calculate confidence interval bounds (adjust as needed)

        lower_bound = forecasts -1.96*df['Inflation Rate'].std()  # Adjust the width of the confidence interval
        upper_bound = forecasts + 1.96*df['Inflation Rate'].std()  # Adjust the width of the confidence interval
        new_df['Upper Bound'] = upper_bound
        new_df['Lower Bound'] = lower_bound

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=date, y=actual,mode='lines', name='Actual'))
    if forecast_type == 'Within_Sample_Forecast':
       fig1.add_trace(go.Scatter(x=date, y=forecasts,mode='lines',line = dict(dict(color='red',width=2,dash ='dash')),name='Within_sample_Forecasts *'))
       # Add an annotation for the loss value
       fig1.add_annotation(
       xref='paper', yref='paper',
       x=0.95, y=0.05,  # Adjust these coordinates for the text position
       text=f'Loss(MSE): {loss:.2f}',
       showarrow=False,
       font=dict(size=13))

    if forecast_type == 'Out_of_Sample_Forecast':
       fig1.add_trace(go.Scatter(x=date, y=forecasts,mode='lines',line = dict(dict(color='red',width=2,dash ='dash')),name='Out_of_sample_Forecasts *'))
              # Add an annotation for the loss value
       fig1.add_annotation(
       xref='paper', yref='paper',
       x=0.95, y=0.05,  # Adjust these coordinates for the text position
       text=f'Loss(MSE): {loss:.2f}',
       showarrow=False,
       font=dict(size=13))

    if forecast_type == 'Future_Forecast':
       fig1.add_trace(go.Scatter(x=date, y=forecasts,mode='lines',line = dict(dict(color='red',width=2,dash ='dash')),name='Future_Forecasts *'))
       # Add shaded area for the confidence interval
       fig1.add_trace(go.Scatter(x=date.tolist() + date.tolist()[::-1],
                             y=lower_bound.tolist() + upper_bound.tolist()[::-1],
                             fill='toself',
                             fillcolor='rgba(255, 0, 0, 0.2)',
                             line=dict(color='rgba(255, 255, 255, 0)'),
                             showlegend=True
                             ,name='Confidance interval  of 95%'))
    #    st.area_chart(new_df[['Lower Bound', 'Upper Bound']], use_container_width=True)
    fig1.update_layout(
    # title="Inflation Rate (Monthly, Year-over-Year)",
    yaxis=dict(title_text="Inflation Rate (%)", titlefont=dict(size=12)),
    xaxis=dict(title_text="Date", titlefont=dict(size=12)),
    legend=dict(x=0, y=1, traceorder='normal', orientation='v' ),

    )



    st.plotly_chart(fig1, use_container_width=True)
    
    # Add text to the Streamlit app
    if forecast_type == 'Within_Sample_Forecast':
       text =''' * Within-sample forecasts are often used to assess how well a model  fits the training data.This evaluation helps determine whether,
       the model captures patterns and relationships present in the data.'''
   
    if forecast_type == 'Out_of_Sample_Forecast':
        text =''' * An out-of-sample forecast refers to the process of making predictions on data that the model has not seen during the training phase.'''
   
    if forecast_type == 'Future_Forecast':
        text =''' * A future forecast refers to predicting values beyond the last observation in the available historical data. It is often used in a forward-looking context
        where the goal is to predict future values based on the patterns learned from historical data.'''
        
    # st.text(text)
    # Display the text using st.markdown with the wide CSS style
    st.markdown(f'<div style="width:100%">{text}</div>', unsafe_allow_html=True)
