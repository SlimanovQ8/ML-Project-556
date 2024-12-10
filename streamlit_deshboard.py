import streamlit as st
from app.deshboard.page_1 import analytics_deshboard_app
from app.configs.const import FILEPATH
from app.data_proprocessing.preprocessing import PREPROCESSINGDATA
from app.models.DecisionTreeBase import DecisionTreeBaseModels
from app.models.xgb_model import XGBMODEL
import matplotlib.pyplot as plt
from app.configs.logging_file import logging
from app.deshboard.utils import process_and_filter_data  ,visualize_medal_trends
from app.deshboard.utils import handle_forecast_for_testing,handle_forecast_for_prediction
import pandas as pd
# Instantiate preprocessing and model objects
_preprocessing_data = PREPROCESSINGDATA(FILEPATH)
processed_data = _preprocessing_data.read_winter_processed_data()

_decisiontreemodels = DecisionTreeBaseModels(FILEPATH)
_xgb_model = XGBMODEL(FILEPATH)

# Extract unique country codes for dropdowns
country_code_list = processed_data['Code'].unique().tolist()

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Olympic Medal Trend Visualization",
    page_icon="🏅",
    layout="wide"
)
# Sidebar for Main Selection
st.sidebar.title("Olympic Medal App")
main_choice = st.sidebar.radio(
    "What are you interested in?",
    ["Dashboard Analytics", "Model Performance", "Future Forecasting"]
)
if main_choice == "Future Forecasting":

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    model_name = st.sidebar.radio(
        "Select a Feature",
        ["Random Forest Model", "Catboost Model", "XGBoost Model"]
    )
    # Select Top N Countries
    top_n = st.sidebar.slider(
        "Select Top N Countries by Medal Count",
        min_value=1,
        max_value=140,
        value=5,
        step=1,
    )
    filtered_data = process_and_filter_data(top_n,processed_data=processed_data)

    # Country Selection
    st.sidebar.subheader("Select a Country for Analysis")
    country_name = st.sidebar.radio("Countries", options=filtered_data['Country_x'].unique())

    # Sidebar Navigation for Forecasting Horizon
    st.sidebar.subheader("Forecasting Options")
    forecast_horizon = st.sidebar.slider(
        "Select Forecasting Horizon (Years)",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )
    logging.info(f'__horizan__ : {forecast_horizon}')
    logging.info(f"__model_name__ : {model_name}")
    if forecast_horizon and model_name and country_name:

        
        # Forecasting
        if model_name == "Random Forest Model":
            handle_forecast_for_prediction(_decisiontreemodels.future_predictions_using_random_forest_model, country_name, "Random Forest Model",forecast_horizon)
        elif model_name == "Catboost Model":
            handle_forecast_for_prediction(_decisiontreemodels.future_predictions_using_catboost_model, country_name, "Catboost Model",forecast_horizon)
        elif model_name == "XGBoost Model":
            handle_forecast_for_prediction(_xgb_model.future_predictions_using_xgb_model, country_name, "XGBoost Model",forecast_horizon)

        else:
            st.write("Please select a valid model.")



elif main_choice=='Dashboard Analytics':
    analytics_deshboard_app()

elif main_choice=='Model Performance':
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    model_choice = st.sidebar.radio(
        "Select a Feature",
        ["Random Forest Model", "Catboost Model", "XGB Model"]
    )
    logging.info(f"Selected Model : {model_choice}")

    if model_choice in ["Random Forest Model", "Catboost Model", "XGB Model"]:
        st.title(f"{model_choice} - Olympic Medal Trends")
        st.write("This section allows you to analyze and forecast Olympic medal trends.")

        # Select Top N Countries
        top_n = st.sidebar.slider(
            "Select Top N Countries by Medal Count",
            min_value=1,
            max_value=140,
            value=5,
            step=1,
        )
        filtered_data = process_and_filter_data(top_n,processed_data=processed_data)


        filtered_data['Year'] = pd.to_datetime(filtered_data['Year'], format='%Y')

        starting_date_of_the_olyampic = filtered_data['Year'].min()
        ending_date_of_the_olyampic = filtered_data['Year'].max()
        all_years = pd.date_range(start=starting_date_of_the_olyampic   , end=ending_date_of_the_olyampic, freq='4YS')
        full_years_df = pd.DataFrame({'Year': all_years})
        
        time_series = pd.merge(full_years_df, filtered_data,  on='Year',how='left')
        
        time_series['Medal Count'] = time_series['Medal Count'].fillna(0)
        time_series['Country_x'] = time_series['Country_x'].fillna(method='ffill')



        # Country Selection
        st.sidebar.subheader("Select a Country for Analysis")
        country_name = st.sidebar.radio("Countries", options=time_series['Country_x'].unique())


        # Forecasting
        if model_choice == "Random Forest Model":
            # visualize_medal_trends(time_series, country_name)
            handle_forecast_for_testing(_decisiontreemodels.train_test_random_forest_model, country_name, "Random Forest Model")
        elif model_choice == "Catboost Model":
            # visualize_medal_trends(time_series, country_name)
            handle_forecast_for_testing(_decisiontreemodels.train_test_catboost_model, country_name, "Catboost Model")
        elif model_choice == "XGB Model":
            # visualize_medal_trends(time_series, country_name)
            handle_forecast_for_testing(_xgb_model.train_test_xgb_model, country_name, "XGB Model")
