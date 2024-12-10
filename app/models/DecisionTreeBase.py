from app.data_proprocessing.preprocessing import PREPROCESSINGDATA
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from app.visualization.data_visualization import Visualization
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
from darts import TimeSeries
from app.configs.logging_file import logging
from app.visualization.data_visualization_plotly import VisualizationPlotly
from darts.models import RandomForest
from darts.models import CatBoostModel


class DecisionTreeBaseModels:

    def __init__(self,FILEPATH=None):
        self.proprocesssing_object = PREPROCESSINGDATA(FILEPATH=FILEPATH)
        self.visualizatione_object = Visualization()
        self.visualizationeplotly_object = VisualizationPlotly()
        self.summar_dataframe = self.proprocesssing_object.read_summar_processed_data()
        self.winter_dataframe = self.proprocesssing_object.read_winter_processed_data()
        self.starting_date_of_the_olyampic = pd.to_datetime('1896-01-01 00:00:00')

    def take_prediction_using_random_forest(self,time_series=None,medal_type=None):
        logging.info('take_prediction_using RandomForest model function called')
        try:
            # Split data into train and test
            logging.info(f"Dates of times series : {time_series.pd_dataframe().reset_index()}")
            train, test = time_series.split_after(0.80)

            random_forest_model = RandomForest(
                        lags=12,
                        # lags_past_covariates=12,
                        # lags_future_covariates=[0,1,2,3,4,5],
                        output_chunk_length=6,
                        n_estimators=200,
                        criterion="absolute_error",
                    )
            random_forest_model.fit(train)
            forecast = random_forest_model.predict(len(test))
            forecast = forecast.pd_dataframe().reset_index()
            
            forecast[f'{medal_type}'] =  forecast[f'{medal_type}'].astype(int)

            # Generate test dataframe for comparison
            test_dataframe = test.pd_dataframe().reset_index()
            forecast_df = pd.DataFrame(
                forecast[medal_type].values,
                index=test_dataframe['Year'],
                columns=['Forecasted_Bronze_Medals']
            )

            # Handle NaN values in the forecast
            if forecast_df['Forecasted_Bronze_Medals'].isna().any():
                logging.warning("NaN values detected in forecasted data. Filling with mean value.")
                forecast_df['Forecasted_Bronze_Medals'].fillna(
                    forecast_df['Forecasted_Bronze_Medals'].mean(), inplace=True
                )

            # Convert train, test, and original time series to DataFrames
            train = train.pd_dataframe().reset_index()
            test = test.pd_dataframe().reset_index()
            dta = time_series.pd_dataframe().reset_index().set_index('Year')
            return train, test, random_forest_model, forecast, forecast_df, dta

        except Exception as e:
            logging.error(f'Error occurred while predicting using Auto ARIMA: {e}',exc_info=True)
            # Return None for outputs in case of error
            return None, None, None, None, None, None

    def take_prediction_using_catboost(self, time_series=None, medal_type=None):
        logging.info(f"__take_prediction_using_catboost__"
                     f"__parameters : Medal type : {medal_type}")
        try:
            # Split data into train and test
            train, test = time_series.split_after(0.80)

            arma_mod30 = CatBoostModel(
                    lags=12,
                    output_chunk_length=len(test)
                )
            arma_mod30.fit(train)
            forecast = arma_mod30.predict(len(test))
            forecast = forecast.pd_dataframe().reset_index()
            forecast[f'{medal_type}'] =  forecast[f'{medal_type}'].astype(int)

            # Generate test dataframe for comparison
            test_dataframe = test.pd_dataframe().reset_index()
            forecast_df = pd.DataFrame(
                forecast[medal_type].values,
                index=test_dataframe['Year'],
                columns=['Forecasted_Bronze_Medals']
            )

            # Handle NaN values in the forecast
            if forecast_df['Forecasted_Bronze_Medals'].isna().any():
                logging.warning("NaN values detected in forecasted data. Filling with mean value.")
                forecast_df['Forecasted_Bronze_Medals'].fillna(
                    forecast_df['Forecasted_Bronze_Medals'].mean(), inplace=True
                )

            # Convert train, test, and original time series to DataFrames
            train = train.pd_dataframe().reset_index()
            test = test.pd_dataframe().reset_index()
            dta = time_series.pd_dataframe().reset_index().set_index('Year')

            return train, test, arma_mod30, forecast, forecast_df, dta

        except Exception as e:
            logging.error(f'Error occurred while predicting using Auto ARIMA: {e}',exc_info=True)
            # Return None for outputs in case of error
            return None, None, None, None, None, None

    # Calculate Errors
    def calculate_errors(self, actual, forecasted, medal_type):
        logging.info(f'calculate_errors function called for {medal_type} medals.')
        try:

            # Ensure both inputs are numpy arrays
            actual = np.array(actual)
            forecasted = np.array(forecasted)

            # Calculate error metrics
            mae = mean_absolute_error(actual, forecasted)
            mape = (np.abs((actual - forecasted) / (actual + 1e-10)))
            mape = np.where(np.isinf(mape), 0, mape).mean() * 100
            smape = (200 * np.abs(actual - forecasted) / (np.abs(actual) + np.abs(forecasted) + 1e-10)).mean()

            if mape>100:
                mape=100
            if smape>200:
                smape=200
            metrics = {
                
                "mae": float(np.round(mae, 2)),
                "mape": float(np.round(mape, 2)),
                "smape": float(np.round(smape, 2))
            }
            return metrics

        except Exception as e:
            logging.error(f"Error occurred while calculating errors: {e}",exc_info=True)
            return None
    def prepare_time_series(self,df=None,full_years_df=None, medal_type=None, frequency='4YS'):
        logging.info(f"prepare_time_series function called  medal_type : {medal_type}")
        try:

            time_series = df.groupby('Year').size().reset_index(name=medal_type)
            time_series = pd.merge(full_years_df, time_series, how='left').fillna(0)
            time_series = self.aligning_dates(df=time_series, medal_type=medal_type)
            time_series = self.filling_missing_dates(df=time_series, medal_type=medal_type)
            return time_series
        except Exception as e:
            logging.error(f"Error occurred while preparing time series: {e}",exc_info=True)
            return None
    def catboost_future_forecast(self,timeseries=None,horizan=10,medal_type=None):
        logging.info(f"auto_arima_future_forecast function called  medal_type : {medal_type}")
        
        # arma_mod30 = AutoARIMA(start_p=8, max_p=12, start_q=1)
        logging.info(f"Hitting the catboost model")
        arma_mod30 = CatBoostModel(
                    lags=16,
                    output_chunk_length=horizan
                )
        arma_mod30.fit(timeseries)
        forecast = arma_mod30.predict(horizan)
        forecast = forecast.pd_dataframe().reset_index()
        forecast[f'{medal_type}'] =  forecast[f'{medal_type}'].astype(int)

        logging.info(f"Forecast dataframe : {forecast}")

        return forecast
    def random_forest_future_forecast(self,timeseries=None,horizan=10,medal_type=None):
        logging.info(f"arima_future_forecast function called  medal_type : {medal_type}")
        
        random_forest_model = RandomForest(
                lags=16,
                output_chunk_length=horizan,
                n_estimators=200,
                criterion="absolute_error")
        random_forest_model.fit(timeseries)
        forecast = random_forest_model.predict(horizan)
        forecast = forecast.pd_dataframe().reset_index()
        forecast[f'{medal_type}'] =  forecast[f'{medal_type}'].astype(int)




        return forecast
    def train_test_random_forest_model(self,country_name='USA'):
        logging.info(f"train_test arima_model function called  country_name : {country_name}")
        try:

            # Group by 'Country_y'
            medal_data = self.summar_dataframe.groupby(['Country_y'])
            file_path = "random_forest_results_metrics.csv"

            for index_name, group_name in medal_data:
                if group_name['Code'].values[0] == country_name:
                    # Extract relevant columns
                    extracted_feature_df = group_name[['Year', 'Code', 'Population', 'GDP per Capita', 'Medal']]

                    extracted_feature_df['Year'] = pd.to_datetime(extracted_feature_df['Year'], format='%Y')
                    
                    
                    ending_date_of_the_olyampic = extracted_feature_df['Year'].max()
                    all_years = pd.date_range(start=self.starting_date_of_the_olyampic    , end=ending_date_of_the_olyampic, freq='4YS')
                    full_years_df = pd.DataFrame({'Year': all_years})

                    # Create a time series for each medal type
                    extracted_feature_df_gold = extracted_feature_df[extracted_feature_df['Medal'] == 'Gold']
                    extracted_feature_df_silver = extracted_feature_df[extracted_feature_df['Medal'] == 'Silver']
                    extracted_feature_df_bronze = extracted_feature_df[extracted_feature_df['Medal'] == 'Bronze']

                    time_series_gold = self.prepare_time_series(df=extracted_feature_df_gold,full_years_df=full_years_df,medal_type='Gold_Medals')
                    time_series_silver = self.prepare_time_series(df=extracted_feature_df_silver,full_years_df=full_years_df,medal_type='Silver_Medals')
                    time_series_bronze = self.prepare_time_series(df=extracted_feature_df_bronze,full_years_df=full_years_df,medal_type='Bronze_Medals')



               


                    # Call the prediction function for each medal type
                    train_gold, test_gold, arma_mod30_gold, forecast_gold, forecast_df_gold, dta_gold = self.take_prediction_using_random_forest(time_series=time_series_gold,
                                                                                                                                            medal_type='Gold_Medals')
                    train_silver, test_silver, arma_mod30_silver, forecast_silver, forecast_df_silver, dta_silver = self.take_prediction_using_random_forest(time_series=time_series_silver,
                                                                                                                                                        medal_type='Silver_Medals')
                    train_bronze, test_bronze, arma_mod30_bronze, forecast_bronze, forecast_df_bronze, dta_bronze = self.take_prediction_using_random_forest(time_series=time_series_bronze,
                                                                                                                                                        medal_type='Bronze_Medals')


                    general_train = train_gold
                    general_train['Total_Medals'] = train_gold['Gold_Medals'] + train_silver['Silver_Medals'] + train_bronze['Bronze_Medals']
                    general_train["Silver_Medals"] = train_silver['Silver_Medals'] 
                    general_train["Bronze_Medals"] = train_bronze['Bronze_Medals'] 
                    general_train["Gold_Medals"] = train_gold['Gold_Medals'] 



                    general_test = test_gold
                    general_test['Total_Medals'] = test_gold['Gold_Medals'] + test_silver['Silver_Medals'] + test_bronze['Bronze_Medals']
                    general_test["Silver_Medals"] = test_silver['Silver_Medals'] 
                    general_test["Bronze_Medals"] = test_bronze['Bronze_Medals'] 
                    general_test["Gold_Medals"] = test_gold['Gold_Medals'] 

                    general_forecast = forecast_df_gold.copy()
                    general_forecast['Total_Medals'] = (forecast_df_gold['Forecasted_Bronze_Medals'] +
                                                        forecast_df_silver['Forecasted_Bronze_Medals'] +
                                                        forecast_df_bronze['Forecasted_Bronze_Medals'])

                    general_forecast["Forecasted_gold_Medals"] = forecast_df_gold["Forecasted_Bronze_Medals"]
                    general_forecast["Forecasted_silver_Medals"] = forecast_df_silver["Forecasted_Bronze_Medals"]
                    general_forecast["Forecasted_bronze_Medals"] = forecast_df_bronze["Forecasted_Bronze_Medals"]

                    # Gold Metrics
                    gold_actual = test_gold['Gold_Medals']
                    gold_forecasted = forecast_df_gold['Forecasted_Bronze_Medals']
                    gold_metrics = self.calculate_errors(gold_actual.values, gold_forecasted.values, "Gold")


                    # Silver Metrics
                    silver_actual = test_silver['Silver_Medals']
                    silver_forecasted = forecast_df_silver['Forecasted_Bronze_Medals']
                    silver_metrics = self.calculate_errors(silver_actual.values, silver_forecasted.values, "Silver")

                    # Bronze Metrics
                    bronze_actual = test_bronze['Bronze_Medals']
                    bronze_forecasted = forecast_df_bronze['Forecasted_Bronze_Medals']
                    bronze_metrics = self.calculate_errors(bronze_actual.values, bronze_forecasted.values, "Bronze")



                    # Bronze Metrics
                    general_forecasted = general_forecast['Total_Medals']

                    general_metrics = self.calculate_errors(general_test['Total_Medals'].values, general_forecasted.values, "General")

            
                            # Create a DataFrame with the current results
                    new_data = pd.DataFrame([{
                        "Country": country_name,
                        "Gold_MAE": gold_metrics['mae'],
                        "Gold_MAPE": gold_metrics['mape'],
                        "Gold_SMAPE": gold_metrics['smape'],
                        "Silver_MAE": silver_metrics['mae'],
                        "Silver_MAPE": silver_metrics['mape'],
                        "Silver_SMAPE": silver_metrics['smape'],
                        "Bronze_MAE": bronze_metrics['mae'],
                        "Bronze_MAPE": bronze_metrics['mape'],
                        "Bronze_SMAPE": bronze_metrics['smape'],
                        "General_MAE": general_metrics['mae'],
                        "General_MAPE": general_metrics['mape'],
                        "General_SMAPE": general_metrics['smape']
                    }])

                    # Check if the file already exists
                    if os.path.exists(file_path):
                        # Load the existing data
                        existing_data = pd.read_csv(file_path)

                        # Filter out the row with the same country name
                        existing_data = existing_data[existing_data["Country"] != country_name]

                        # Concatenate the new data with the remaining data
                        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                    else:
                        # If the file doesn't exist, just use the new data
                        updated_data = new_data

                    # Save the updated DataFrame back to the file
                    updated_data.to_csv(file_path, index=False)
                    logging.info(f"Results saved to {file_path}")
            
                    

                    fig = self.visualizationeplotly_object.plots_individual_plots_and_general_trend(
                                                    train_gold=train_gold.set_index("Year"),
                                                    train_silver=train_silver.set_index("Year"),
                                                    train_bronze=train_bronze.set_index("Year"),
                                                    test_gold=test_gold.set_index("Year"),
                                                    test_silver=test_silver.set_index("Year"),
                                                    test_bronze=test_bronze.set_index("Year"),
                                                    forecast_df_silver=forecast_df_silver,
                                                    forecast_df_bronze=forecast_df_bronze,
                                                    forecast_df_gold=forecast_df_gold,
                                                    dta_gold=dta_gold,
                                                    dta_bronze=dta_bronze,
                                                    dta_silver=dta_silver,
                                                    general_train=general_train.set_index('Year'),
                                                    general_test=general_test.set_index('Year'),
                                                    general_forecast=general_forecast,
                                                    gold_metrics=gold_metrics,
                                                    silver_metrics=silver_metrics,
                                                    bronze_metrics=bronze_metrics,
                                                    general_metrics=general_metrics
                                                    )
                    
                    

                    return fig
        except Exception as e:
            logging.error(f"Error occurred while processing country: {country_name}, error: {str(e)}",exc_info=True)
            return None            
    def filling_missing_dates(self,df=None,medal_type=None):
        olyampic_data_time_series = TimeSeries.from_dataframe(
            df,
            'Year',
            medal_type,
            freq='4YS',
            fill_missing_dates=True,
            fillna_value=0
        )
        return olyampic_data_time_series
    
    def aligning_dates(self,df=None,medal_type=None):
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')
        all_years = pd.date_range(
            start=df['Year'].min(), end=df['Year'].max(), freq='4YS')
        full_years_df = pd.DataFrame({'Year': all_years})

        time_series_aligned_dates = pd.merge(
            full_years_df, df, on='Year', how='left')
        df[medal_type].fillna(0, inplace=True)

        return time_series_aligned_dates
            
    def train_test_catboost_model(self,country_name='USA'):
        logging.info(f"__train_test_catboost_model__: country_name : {country_name}")
        try:

            # Group by 'Country_y'
            medal_data = self.summar_dataframe.groupby(['Country_y'])
            file_path = "catboost_results_metrics.csv"

            for index_name, group_name in medal_data:
                if group_name['Code'].values[0] == country_name:
                    # Extract relevant columns
                    extracted_feature_df = group_name[['Year', 'Code', 'Population', 'GDP per Capita', 'Medal']]

                    extracted_feature_df['Year'] = pd.to_datetime(extracted_feature_df['Year'], format='%Y')
                    

                    ending_date_of_the_olyampic = extracted_feature_df['Year'].max()
                    all_years = pd.date_range(start=self.starting_date_of_the_olyampic   , end=ending_date_of_the_olyampic, freq='4YS')
                    full_years_df = pd.DataFrame({'Year': all_years})

                    # Create a time series for each medal type
                    extracted_feature_df_gold = extracted_feature_df[extracted_feature_df['Medal'] == 'Gold']
                    extracted_feature_df_silver = extracted_feature_df[extracted_feature_df['Medal'] == 'Silver']
                    extracted_feature_df_bronze = extracted_feature_df[extracted_feature_df['Medal'] == 'Bronze']


                    # Count the number of medals per year for each medal type

                    time_series_gold = self.prepare_time_series(df=extracted_feature_df_gold,full_years_df=full_years_df,medal_type='Gold_Medals')
                    time_series_silver = self.prepare_time_series(df=extracted_feature_df_silver,full_years_df=full_years_df,medal_type='Silver_Medals')
                    time_series_bronze = self.prepare_time_series(df=extracted_feature_df_bronze,full_years_df=full_years_df,medal_type='Bronze_Medals')

       
                    # Call the prediction function for each medal type
                    train_gold, test_gold, arma_mod30_gold, forecast_gold, forecast_df_gold, dta_gold = self.take_prediction_using_catboost(time_series=time_series_gold,
                                                                                                                                            medal_type='Gold_Medals')
                    train_silver, test_silver, arma_mod30_silver, forecast_silver, forecast_df_silver, dta_silver = self.take_prediction_using_catboost(time_series=time_series_silver,
                                                                                                                                                        medal_type='Silver_Medals')
                    train_bronze, test_bronze, arma_mod30_bronze, forecast_bronze, forecast_df_bronze, dta_bronze = self.take_prediction_using_catboost(time_series=time_series_bronze,
                                                                                                                                                        medal_type='Bronze_Medals')


                    general_train = train_gold
                    general_train['Total_Medals'] = train_gold['Gold_Medals'] + train_silver['Silver_Medals'] + train_bronze['Bronze_Medals']
                    general_train["Silver_Medals"] = train_silver['Silver_Medals'] 
                    general_train["Bronze_Medals"] = train_bronze['Bronze_Medals'] 
                    general_train["Gold_Medals"] = train_gold['Gold_Medals'] 
                    
                    
                    
                    general_test = test_gold
                    general_test['Total_Medals'] = test_gold['Gold_Medals'] + test_silver['Silver_Medals'] + test_bronze['Bronze_Medals']
                    general_test["Silver_Medals"] = test_silver['Silver_Medals'] 
                    general_test["Bronze_Medals"] = test_bronze['Bronze_Medals'] 
                    general_test["Gold_Medals"] = test_gold['Gold_Medals'] 

                    # logging.info(f"2: General testing dataset : {general_test}")

                    general_forecast = forecast_df_gold.copy()
                    general_forecast['Total_Medals'] = (forecast_df_gold['Forecasted_Bronze_Medals'] +
                                                        forecast_df_silver['Forecasted_Bronze_Medals'] +
                                                        forecast_df_bronze['Forecasted_Bronze_Medals'])
                    general_forecast["Forecasted_gold_Medals"] = forecast_df_gold["Forecasted_Bronze_Medals"]
                    general_forecast["Forecasted_silver_Medals"] = forecast_df_silver["Forecasted_Bronze_Medals"]
                    general_forecast["Forecasted_bronze_Medals"] = forecast_df_bronze["Forecasted_Bronze_Medals"]



                    # Gold Metrics
                    gold_actual = test_gold['Gold_Medals']
                    gold_forecasted = forecast_df_gold['Forecasted_Bronze_Medals']
                    gold_metrics = self.calculate_errors(gold_actual.values, gold_forecasted.values, "Gold")


                    # Silver Metrics
                    silver_actual = test_silver['Silver_Medals']
                    silver_forecasted = forecast_df_silver['Forecasted_Bronze_Medals']
                    silver_metrics = self.calculate_errors(silver_actual.values, silver_forecasted.values, "Silver")

                    # Bronze Metrics
                    bronze_actual = test_bronze['Bronze_Medals']
                    bronze_forecasted = forecast_df_bronze['Forecasted_Bronze_Medals']
                    bronze_metrics = self.calculate_errors(bronze_actual.values, bronze_forecasted.values, "Bronze")



                    # Bronze Metrics
                    general_forecasted = general_forecast['Total_Medals']

                    general_metrics = self.calculate_errors(general_test['Total_Medals'].values, general_forecasted.values, "General")

                    logging.info(f"Error Calculated Metrics: {general_metrics}")
            
                            # Create a DataFrame with the current results
                    new_data = pd.DataFrame([{
                        "Country": country_name,
                        "Gold_MAE": gold_metrics['mae'],
                        "Gold_MAPE": gold_metrics['mape'],
                        "Gold_SMAPE": gold_metrics['smape'],
                        "Silver_MAE": silver_metrics['mae'],
                        "Silver_MAPE": silver_metrics['mape'],
                        "Silver_SMAPE": silver_metrics['smape'],
                        "Bronze_MAE": bronze_metrics['mae'],
                        "Bronze_MAPE": bronze_metrics['mape'],
                        "Bronze_SMAPE": bronze_metrics['smape'],
                        "General_MAE": general_metrics['mae'],
                        "General_MAPE": general_metrics['mape'],
                        "General_SMAPE": general_metrics['smape']
                    }])

                    # Check if the file already exists
                    if os.path.exists(file_path):
                        # Load the existing data
                        existing_data = pd.read_csv(file_path)

                        existing_data = existing_data[existing_data["Country"] != country_name]

                        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                    else:
                        updated_data = new_data

                    # Save the updated DataFrame back to the file
                    updated_data.to_csv(file_path, index=False)
                    logging.info(f"Results saved to {file_path}")

                    fig = self.visualizationeplotly_object.plots_individual_plots_and_general_trend(
                                                    train_gold=train_gold.set_index("Year"),
                                                    train_silver=train_silver.set_index("Year"),
                                                    train_bronze=train_bronze.set_index("Year"),
                                                    test_gold=test_gold.set_index("Year"),
                                                    test_silver=test_silver.set_index("Year"),
                                                    test_bronze=test_bronze.set_index("Year"),
                                                    forecast_df_silver=forecast_df_silver,
                                                    forecast_df_bronze=forecast_df_bronze,
                                                    forecast_df_gold=forecast_df_gold,
                                                    dta_gold=dta_gold,
                                                    dta_bronze=dta_bronze,
                                                    dta_silver=dta_silver,
                                                    general_train=general_train.set_index('Year'),
                                                    general_test=general_test.set_index('Year'),
                                                    general_forecast=general_forecast,
                                                    gold_metrics=gold_metrics,
                                                    silver_metrics=silver_metrics,
                                                    bronze_metrics=bronze_metrics,
                                                    general_metrics=general_metrics
                                                    )
                    
                    

                    return fig
        except Exception as e:
            logging.error(f"Error occurred while processing country: {country_name}, error: {str(e)}",exc_info=True)
            return None
    def future_predictions_using_catboost_model(self,country_name=None,forecast_horizon=10):
        logging.info(f"__future_predictions_using_catboost_model__: "
                     f"country_name : {country_name}" , 
                     f"forecast_horizon : {forecast_horizon}")
        medal_data = self.summar_dataframe.groupby(['Country_y'])
        
        for index_name, group_name in medal_data:
            if group_name['Code'].values[0] == country_name:
                extracted_feature_df = group_name[['Year', 'Code', 'Population', 'GDP per Capita', 'Medal']]
                extracted_feature_df['Year'] = pd.to_datetime(extracted_feature_df['Year'], format='%Y')
                
            
                ending_date_of_the_olyampic = extracted_feature_df['Year'].max()
                
                all_years = pd.date_range(start=self.starting_date_of_the_olyampic   , end=ending_date_of_the_olyampic, freq='4YS')
                full_years_df = pd.DataFrame({'Year': all_years})

                # Create a time series for each medal type
                extracted_feature_df_gold = extracted_feature_df[extracted_feature_df['Medal'] == 'Gold']
                extracted_feature_df_silver = extracted_feature_df[extracted_feature_df['Medal'] == 'Silver']
                extracted_feature_df_bronze = extracted_feature_df[extracted_feature_df['Medal'] == 'Bronze']

                time_series_gold = self.prepare_time_series(df=extracted_feature_df_gold,full_years_df=full_years_df,medal_type='Gold_Medals')
                time_series_silver = self.prepare_time_series(df=extracted_feature_df_silver,full_years_df=full_years_df,medal_type='Silver_Medals')
                time_series_bronze = self.prepare_time_series(df=extracted_feature_df_bronze,full_years_df=full_years_df,medal_type='Bronze_Medals')

                gold_forecasted_df = self.catboost_future_forecast(timeseries=time_series_gold,horizan=forecast_horizon,medal_type="Gold_Medals")
                silver_forecasted_df = self.catboost_future_forecast(timeseries=time_series_silver,horizan=forecast_horizon,medal_type="Silver_Medals")
                bronze_forecasted_df = self.catboost_future_forecast(timeseries=time_series_bronze,horizan=forecast_horizon,medal_type="Bronze_Medals")

                time_series_gold_df = time_series_gold.pd_dataframe().reset_index()
                time_series_silver_df = time_series_silver.pd_dataframe().reset_index()
                time_series_bronze_df = time_series_bronze.pd_dataframe().reset_index()
                
                
                historical_data = time_series_gold_df
                historical_data['Total_Medals'] = time_series_gold_df['Gold_Medals'] + time_series_silver_df['Silver_Medals'] + time_series_bronze_df['Bronze_Medals']
                historical_data["Silver_Medals"] = time_series_silver_df['Silver_Medals'] 
                historical_data["Bronze_Medals"] = time_series_bronze_df['Bronze_Medals'] 
                historical_data["Gold_Medals"] = time_series_gold_df['Gold_Medals'] 

                forecasted_data = gold_forecasted_df
                forecasted_data['Total_Medals'] = gold_forecasted_df['Gold_Medals'] + silver_forecasted_df['Silver_Medals'] + bronze_forecasted_df['Bronze_Medals']



                fig = self.visualizationeplotly_object.plot_forecasting_values_only(
                    time_series_gold=time_series_gold.pd_dataframe().reset_index(),
                    time_series_silver=time_series_silver.pd_dataframe().reset_index(),
                    time_series_bronze=time_series_bronze.pd_dataframe().reset_index(),
                    gold_forecasted_df=gold_forecasted_df,
                    silver_forecasted_df=silver_forecasted_df,
                    bronze_forecasted_df=bronze_forecasted_df,
                    country_name=country_name,
                    historical_data=historical_data,
                    forecasted_data=forecasted_data
                )
                return fig
            
    def future_predictions_using_random_forest_model(self,country_name=None,forecast_horizon=10):
        try:

            logging.info("function future_predictions_using_arima_model called")
            medal_data = self.summar_dataframe.groupby(['Country_y'])
            
            for index_name, group_name in medal_data:
                if group_name['Code'].values[0] == country_name:
                    extracted_feature_df = group_name[['Year', 'Code', 'Population', 'GDP per Capita', 'Medal']]
                    extracted_feature_df['Year'] = pd.to_datetime(extracted_feature_df['Year'], format='%Y')
                    

                    ending_date_of_the_olyampic = extracted_feature_df['Year'].max()
                    
                    all_years = pd.date_range(start=self.starting_date_of_the_olyampic   , end=ending_date_of_the_olyampic, freq='4YS')
                    full_years_df = pd.DataFrame({'Year': all_years})

                    # Create a time series for each medal type
                    extracted_feature_df_gold = extracted_feature_df[extracted_feature_df['Medal'] == 'Gold']
                    extracted_feature_df_silver = extracted_feature_df[extracted_feature_df['Medal'] == 'Silver']
                    extracted_feature_df_bronze = extracted_feature_df[extracted_feature_df['Medal'] == 'Bronze']

                    time_series_gold = self.prepare_time_series(df=extracted_feature_df_gold,full_years_df=full_years_df,medal_type='Gold_Medals')
                    time_series_silver = self.prepare_time_series(df=extracted_feature_df_silver,full_years_df=full_years_df,medal_type='Silver_Medals')
                    time_series_bronze = self.prepare_time_series(df=extracted_feature_df_bronze,full_years_df=full_years_df,medal_type='Bronze_Medals')

                    gold_forecasted_df = self.random_forest_future_forecast(timeseries=time_series_gold,horizan=forecast_horizon,medal_type="Gold_Medals")
                    silver_forecasted_df = self.random_forest_future_forecast(timeseries=time_series_silver,horizan=forecast_horizon,medal_type="Silver_Medals")
                    bronze_forecasted_df = self.random_forest_future_forecast(timeseries=time_series_bronze,horizan=forecast_horizon,medal_type="Bronze_Medals")

                    time_series_gold_df = time_series_gold.pd_dataframe().reset_index()
                    time_series_silver_df = time_series_silver.pd_dataframe().reset_index()
                    time_series_bronze_df = time_series_bronze.pd_dataframe().reset_index()
                    
                    
                    historical_data = time_series_gold_df
                    historical_data['Total_Medals'] = time_series_gold_df['Gold_Medals'] + time_series_silver_df['Silver_Medals'] + time_series_bronze_df['Bronze_Medals']
                    historical_data["Silver_Medals"] = time_series_silver_df['Silver_Medals'] 
                    historical_data["Bronze_Medals"] = time_series_bronze_df['Bronze_Medals'] 
                    historical_data["Gold_Medals"] = time_series_gold_df['Gold_Medals'] 

                    forecasted_data = gold_forecasted_df
                    forecasted_data['Total_Medals'] = gold_forecasted_df['Gold_Medals'] + silver_forecasted_df['Silver_Medals'] + bronze_forecasted_df['Bronze_Medals']



                    fig = self.visualizationeplotly_object.plot_forecasting_values_only(
                        time_series_gold=time_series_gold.pd_dataframe().reset_index(),
                        time_series_silver=time_series_silver.pd_dataframe().reset_index(),
                        time_series_bronze=time_series_bronze.pd_dataframe().reset_index(),
                        gold_forecasted_df=gold_forecasted_df,
                        silver_forecasted_df=silver_forecasted_df,
                        bronze_forecasted_df=bronze_forecasted_df,
                        country_name=country_name,
                        historical_data=historical_data,
                        forecasted_data=forecasted_data
                    )
                    return fig
        except Exception as e:
            logging.error(f"Error in future_predictions_using_random_forest_model: {str(e)}",exc_info=True)
            return None