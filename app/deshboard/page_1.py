import streamlit as st
import pandas as pd
import plotly.express as px
from app.configs.const import FILEPATH
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import numpy as np
from app.configs.logging_file import logging



def analytics_deshboard_app():
    print("called analytics_deshboard_app app")
    logging.info("called analytics_deshboard_app app")
    st.write("# Olympic Medal Trend Visualization")

    # Load and merge data with exception handling
    try:
        # Attempt to load the data
        winter_data_df = pd.read_csv(FILEPATH['winter_data'])
        summar_data_df = pd.read_csv(FILEPATH['summar_data'])
        dictionary_data_df = pd.read_csv(FILEPATH['dictionary_data'])

        # Merge dataframes
        winter_data_merged_df = pd.merge(winter_data_df, dictionary_data_df, left_on='Country', right_on='Code')
        summar_data_merged_df = pd.merge(summar_data_df, dictionary_data_df, left_on='Country', right_on='Code')
        # Categorize GDP per capita
        bins_gdp = [-float('inf'), 10000, 30000, float('inf')]
        labels_gdp = ['Low Income', 'Middle Income', 'High Income']
        summar_data_merged_df['GDP_Category'] = pd.cut(
            summar_data_merged_df['GDP per Capita'], bins=bins_gdp, labels=labels_gdp
        )

        # Categorize Population
        bins_pop = [0, 1e7, 5e7, float('inf')]
        labels_pop = ['Small Population', 'Medium Population', 'Large Population']
        summar_data_merged_df['Population_Category'] = pd.cut(
            summar_data_merged_df['Population'], bins=bins_pop, labels=labels_pop
        )
        df = summar_data_merged_df  # Use the merged dataframe

        # Define available grouping features
        features = [None,"Medal", 'City', 'Sport', 'Discipline', 'Gender', 'Event']
        # AgGrid(df)

        
    except FileNotFoundError as e:
        st.error(f"One or more data files are missing: {e}")
        logging.error(f"One or more data files are missing: {e}")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("One or more data files are empty or corrupted.")
        logging.error("One or more data files are empty or corrupted.")
        st.stop()

    except Exception as e:
        st.error(f"An unexpected error occurred while loading or merging data: {e}")
        logging.error(f"An unexpected error occurred while loading or merging data: {e}",exc_info=True)
        st.stop()

    # Sidebar options for visualization type
    visualization_type = st.sidebar.selectbox(
        "Select Visualization Type",
        [None,"Line Chart"],  # Options
        index=0          # Default selection (0 corresponds to "Line Chart")
    )


    if visualization_type == "Line Chart":
        # Line chart - Medal Counts Over the Years
        st.subheader("Line Chart - Medal Counts Over the Years")
        top_n = st.sidebar.slider("Select Top N Countries", 1, 140, 10, key="top_n_1")

        try:
            medal_trends = df.groupby(['Year', 'Country_x']).size().reset_index(name='Medal Count')
            total_medals = medal_trends.groupby('Country_x')['Medal Count'].sum().reset_index()
            top_countries = total_medals.sort_values(by='Medal Count', ascending=False).head(top_n)['Country_x']
            filtered_df = medal_trends[medal_trends['Country_x'].isin(top_countries)]

            filtered_df['Year'] = pd.to_datetime(filtered_df['Year'], format='%Y')

            # starting_date_of_the_olyampic = filtered_df['Year'].min()
            starting_date_of_the_olyampic = pd.to_datetime('1800-01-01 00:00:00')
            ending_date_of_the_olyampic = filtered_df['Year'].max()

            all_years = pd.date_range(start=starting_date_of_the_olyampic, 
                                      end=ending_date_of_the_olyampic, 
                                      freq='4YS')
            
            full_years_df = pd.DataFrame({'Year': all_years})
            
            time_series = pd.merge(full_years_df, filtered_df,  on='Year',how='left')
            
            time_series['Medal Count'] = time_series['Medal Count'].fillna(0)
            # time_series['Country_x'] = time_series['Country_x'].fillna(method='ffill')
            # time_series['Country_x'] = time_series['Country_x'].fillna(method='bfill')

            df_filtered = time_series.copy()
            
            logging.info(f"df_filtered : {df_filtered}")# Create the line chart
            fig = px.line(
                df_filtered,
                x='Year',
                y='Medal Count',
                color='Country_x',
                title=f"Medal Counts Over the Years (Top {top_n} Countries)",
                markers=True
            )
            st.plotly_chart(fig, key="line_chart_1")

        except KeyError as e:
            st.error(f"Data grouping error: {e}")
            logging.error(f"Data grouping error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred while generating the line chart: {e}")
            logging.error(f"An unexpected error occurred while loading or merging data: {e}",exc_info=True)
            logging.error(f"filtered_df columns : {list(filtered_df)}")
            logging.error(f"filtered_df : {filtered_df}")


        st.subheader("Grouping Visualization")
        st.write(
            """
            **Instructions:**
            - Use the dropdown menus below to select multiple grouping levels.
            - You can group the data by features such as City, Sport, Discipline, Gender, and Event.
            - The chart will update dynamically based on the selected grouping levels.
            - This feature is useful to explore how different grouping levels impact medal counts across countries.

            **Grouping Levels:**
            - **Level 1** to **Level 5** allow you to hierarchically group the data by different attributes.
            - For example, you can start by grouping by 'Sport', then further group by 'Gender' and 'Event' to see the breakdown of medal counts in a specific sport by gender and event.
            """
        )

        # Sidebar options for grouping levels
        col1, col2, col3 = st.columns(3)
        with col1:
            level_1 = st.selectbox("Select Grouping Level 1", features, key='level_1', index=0)
            level_2 = st.selectbox("Select Grouping Level 2", features, key='level_2', index=0)

        with col2:
            level_3 = st.selectbox("Select Grouping Level 3", features, key='level_3', index=0)
            level_4 = st.selectbox("Select Grouping Level 4", features, key='level_4', index=0)

        with col3:
            level_5 = st.selectbox("Select Grouping Level 5", features, key='level_5', index=0)

        try:
            
            # Group dynamically based on selected features
            grouping_columns = [level_1, level_2, level_3, level_4, level_5]

            # Filter out any empty selections
            grouping_columns = [col for col in grouping_columns if col]
            
            # # Group by the selected columns dynamically
            # medal_trends = df.groupby(['Year', 'Country_x'] + grouping_columns).size().reset_index(name='Medal Count')
            medal_trends = df.groupby(['Year', 'Country_x']).size().reset_index(name='Medal Count')

            # # Calculate total medals for each country and grouping
            # total_medals = medal_trends.groupby(['Country_x'] + grouping_columns)['Medal Count'].sum().reset_index()

            # # Sort by total medals and select the top N countries
            top_countries = total_medals.sort_values(by='Medal Count', ascending=False).head(top_n)['Country_x']

            # # Filter the original data to include only the top N countries
            filtered_df = df[df['Country_x'].isin(top_countries)]
    
            # # Country Selection
            # st.sidebar.subheader("Select a Country for Analysis")
            # country_name = st.sidebar.radio("Countries", options=filtered_df['Country_x'].unique())
            # filtered_df = filtered_df[filtered_df['Country_x']==country_name]

            # GDP Country Filter
            st.sidebar.subheader("Select a Country for Analysis")
            country_options = ["None"] + list(filtered_df['Country_x'].unique())
            country_name = st.sidebar.radio("Countries", options=country_options, index=0)
            if country_name != "None":
                filtered_df = filtered_df[filtered_df['Country_x'] == country_name]

            
            # GDP Category Filter
            st.sidebar.subheader("Select a GDP_Category for Analysis")
            gdp_options = ["None"] + list(filtered_df['GDP_Category'].unique())
            gdp_category = st.sidebar.radio("GDP Category", options=gdp_options, index=0)
            if gdp_category != "None":
                filtered_df = filtered_df[filtered_df['GDP_Category'] == gdp_category]


            # Population Category Filter
            st.sidebar.subheader("Select a Population_Category for Analysis")
            population_options = ["None"] + list(filtered_df['Population_Category'].unique())
            population_category = st.sidebar.radio("Population Category", options=population_options, index=0)
            if population_category != "None":
                filtered_df = filtered_df[filtered_df['Population_Category'] == population_category]

            # Ensure filtered_df is not empty after filtering
            if filtered_df.empty:
                st.warning("No data available for the selected filters.")
            else:
                # Regroup filtered data
                filtered_df = filtered_df.groupby(['Year', 'Country_x'] + grouping_columns).size().reset_index(name='Medal Count')

            logging.info(f"top_countries : {top_countries}")
            logging.info(f"df : {list(df)}")
            logging.info(f"filtered_df : {list(filtered_df)}")
            logging.info(f"grouping_columns : {grouping_columns}")


            # Create the bar chart based on the grouping level selections
            fig = px.bar(
                filtered_df,
                x=grouping_columns[-1],  # Use dynamic grouping columns
                y='Medal Count',
                color='Year',
                title=f"Medal Counts by {level_1} -> {level_2} -> {level_3} -> {level_4} -> {level_5}",
                barmode="group"
            )

            # Render the chart
            st.plotly_chart(fig, key="line_chart_4")

        except KeyError as e:
            st.error(f"Data grouping error: {e}")
            logging.error(f"Data grouping error: {e}",exc_info=True)
        except Exception as e:
            st.error(f"Please select features first")
            logging.error(f"An unexpected error occurred while generating the bar chart: {e}",exc_info=True)
            
