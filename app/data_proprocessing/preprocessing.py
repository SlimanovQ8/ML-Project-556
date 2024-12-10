import pandas as pd


class PREPROCESSINGDATA:

    def __init__(self,FILEPATH=None):
        try:
            self.winter_raw_data_path = FILEPATH['winter_data']
            self.summar_raw_data_path = FILEPATH['summar_data']
            self.gdp_data_raw_data_path = FILEPATH['dictionary_data']
        except KeyError as e:
            raise KeyError(f"Missing key in FILEPATH: {e}") from e

    def read_winter_data(self):
        try:
            df = pd.read_csv(self.winter_raw_data_path)
            return df
        except FileNotFoundError as e:
            print(f"Error: Winter data file not found at {self.winter_raw_data_path}.")
            raise e
        except pd.errors.ParserError as e:
            print(f"Error: Unable to parse winter data file at {self.winter_raw_data_path}.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred while reading winter data: {e}")
            raise e

    def read_summar_data(self):
        try:
            df = pd.read_csv(self.summar_raw_data_path)
            return df
        except FileNotFoundError as e:
            print(f"Error: Summer data file not found at {self.summar_raw_data_path}.")
            raise e
        except pd.errors.ParserError as e:
            print(f"Error: Unable to parse summer data file at {self.summar_raw_data_path}.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred while reading summer data: {e}")
            raise e

    def read_gdp_data(self):
        try:
            df = pd.read_csv(self.gdp_data_raw_data_path)
            return df
        except FileNotFoundError as e:
            print(f"Error: GDP data file not found at {self.gdp_data_raw_data_path}.")
            raise e
        except pd.errors.ParserError as e:
            print(f"Error: Unable to parse GDP data file at {self.gdp_data_raw_data_path}.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred while reading GDP data: {e}")
            raise e

    def read_winter_processed_data(self): 
        try:
            df_winter = self.read_winter_data()
            df_gdp = self.read_gdp_data()
            df = pd.merge(df_winter, df_gdp, left_on='Country', right_on='Code')
            return df
        except KeyError as e:
            print(f"Error: Missing column during winter data merging: {e}")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred while processing winter data: {e}")
            raise e

    def read_summar_processed_data(self): 
        try:
            df_summer = self.read_summar_data()
            df_gdp = self.read_gdp_data()
            df = pd.merge(df_summer, df_gdp, left_on='Country', right_on='Code')
            return df
        except KeyError as e:
            print(f"Error: Missing column during summer data merging: {e}")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred while processing summer data: {e}")
            raise e
