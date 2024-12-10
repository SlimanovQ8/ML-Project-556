from app.configs.const import FILEPATH
from app.data_proprocessing.preprocessing import PREPROCESSINGDATA
from app.models.arima_model import ARIMAMODEL
import pandas as pd

from darts import TimeSeries
from darts.models import AutoARIMA
from app.configs.logging_file import logging
from app.models.xgb_model import  XGBMODEL

logging.info("Starting the model")
_proprocessingdata = PREPROCESSINGDATA(FILEPATH)
proprocessed_data = _proprocessingdata.read_winter_processed_data()

country_code_list = proprocessed_data['Code'].tolist()


# Group the data for medal trends
medal_trends = proprocessed_data.groupby(
    ['Year', 'Country_x']).size().reset_index(name='Medal Count')

# Calculate total medals for each country
total_medals = medal_trends.groupby(
    'Country_x')['Medal Count'].sum().reset_index()

# Sort countries by total medal count and select the top N (e.g., top 10 countries)
top_countries = total_medals.sort_values(
    by='Medal Count', ascending=False).head(10)['Country_x']

# Filter the original data to include only top countries
filtered_df = medal_trends[medal_trends['Country_x'].isin(top_countries)]


# arima_model_object = ARIMAMODEL(FILEPATH=FILEPATH)
# arima_model_object.train_test_auto_arima_model(country_name="SUI")
xgb_model_object = XGBMODEL(FILEPATH=FILEPATH)
xgb_model_object.future_predictions_using_xgb_model(country_name="USA")