import streamlit as st
from app.deshboard.page_2 import arima_model_app
import plotly.graph_objects as go
import pandas as pd


def process_and_filter_data(top_n,processed_data=None):
    """
    Helper function to group, process, and filter data for the top N countries.
    """
    medal_trends = processed_data.groupby(['Year', 'Country_x']).size().reset_index(name='Medal Count')
    total_medals = medal_trends.groupby('Country_x')['Medal Count'].sum().reset_index()
    top_countries = total_medals.sort_values(by='Medal Count', ascending=False).head(top_n)['Country_x']
    
    filtered_df = medal_trends[medal_trends['Country_x'].isin(top_countries)]
    
    filtered_df['Year'] = pd.to_datetime(filtered_df['Year'], format='%Y')

    starting_date_of_the_olyampic = filtered_df['Year'].min()
    ending_date_of_the_olyampic = filtered_df['Year'].max()
    all_years = pd.date_range(start=starting_date_of_the_olyampic   , end=ending_date_of_the_olyampic, freq='4YS')
    full_years_df = pd.DataFrame({'Year': all_years})
    
    time_series = pd.merge(full_years_df, filtered_df,  on='Year',how='left')
    
    time_series['Medal Count'] = time_series['Medal Count'].fillna(0)
    time_series['Country_x'] = time_series['Country_x'].fillna(method='ffill')

    df = time_series.copy()
    df = df[['Year','Medal Count','Country_x']]
    
    return df
def handle_forecast_for_testing(model, country_name, method_name):
    """
    Handle forecasting using the selected model.
    """
    st.subheader(f"Forecasting Medal Trends - {method_name}")
    st.write(
        "The model is trained on the historical data of the selected country to forecast future trends."
    )

    forecast_fig = model(country_name=country_name)
    if forecast_fig:
        arima_model_app(fig=forecast_fig)
    else:
        st.error(f"Insufficient historical data for {country_name}. Unable to generate a forecast.")

def handle_forecast_for_prediction(model, country_name, method_name,forecast_horizon):
    """
    Handle forecasting using the selected model.
    """
    st.subheader(f"Forecasting Medal Trends - {method_name}")
    st.write(
        "The model is trained on the historical data of the selected country to forecast future trends."
    )

    forecast_fig = model(country_name=country_name,forecast_horizon=forecast_horizon)
    if forecast_fig:
        arima_model_app(fig=forecast_fig)
    else:
        st.error(f"Insufficient historical data for {country_name}. Unable to generate a forecast.")


def visualize_medal_trends(filtered_df, country_name):
    """
    Helper function to visualize medal trends for the selected country using Plotly.
    """
    st.write(f"### Medal Trends for {country_name}")
    
    # Filter data for the selected country
    country_data = filtered_df[filtered_df['Country_x'] == country_name]
    
    if not country_data.empty:
        # Create an interactive Plotly figure
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=country_data['Year'], 
                y=country_data['Medal Count'], 
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=8, color='orange'),
                name=f"{country_name} Medal Trend"
            )
        )

        # Update layout
        fig.update_layout(
            title=f"Medal Trends for {country_name}",
            xaxis_title="Year",
            yaxis_title="Medal Count",
            template="plotly_white",
            font=dict(size=14),
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {country_name}.")

