import pandas as pd
import numpy as np
import seaborn as sns
from app.configs.const import FILEPATH
import pandas as pd
import numpy
import os
import matplotlib_inline
import matplotlib.pyplot as plt

# !pip install -r requirements.txt


if __name__=='__main__':
    winter_data_df = pd.read_csv(FILEPATH['winter_data'])
    summar_data_df = pd.read_csv(FILEPATH['summar_data'])
    dictionary_data_df = pd.read_csv(FILEPATH['dictionary_data'])
    winter_data_merged_df = pd.merge(winter_data_df, dictionary_data_df, left_on='Country', right_on='Code')
    summar_data_merged_df = pd.merge(summar_data_df, dictionary_data_df, left_on='Country', right_on='Code')

    import streamlit as st
    import pandas as pd
    import plotly.express as px

    # Example DataFrame
    data = {
        'Year': [1924, 1924, 1924, 1924, 1924, 2014, 2014, 2014, 2014, 2014],
        'Country_x': ['AUT', 'BEL', 'CAN', 'FIN', 'FRA', 'SUI', 'SVK', 'SWE', 'UKR', 'USA'],
        'Medal Count': [4, 5, 9, 15, 12, 32, 1, 55, 5, 50]
    }
    df = pd.DataFrame(data)

    # Streamlit App
    st.title("Olympic Medal Count Visualization")

    # Sidebar options
    visualization_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Line Chart", "Heatmap", "Bar Chart", "Choropleth Map"]
    )

    # Line Chart (Top N Countries)
    if visualization_type == "Line Chart":
        st.subheader("Line Chart - Medal Counts Over the Years")
        top_n = st.sidebar.slider("Select Top N Countries", 1, 20, 10)
        top_countries = df.groupby('Country_x')['Medal Count'].sum().nlargest(top_n).index
        filtered_df = df[df['Country_x'].isin(top_countries)]
        fig = px.line(
            filtered_df,
            x='Year',
            y='Medal Count',
            color='Country_x',
            title=f"Medal Counts Over the Years (Top {top_n} Countries)",
            markers=True
        )
        st.plotly_chart(fig)

    # Heatmap
    elif visualization_type == "Heatmap":
        st.subheader("Heatmap - Medal Counts by Country and Year")
        heatmap_df = df.pivot(index='Country_x', columns='Year', values='Medal Count').fillna(0)
        fig = px.imshow(
            heatmap_df,
            labels={'color': 'Medal Count'},
            x=heatmap_df.columns,
            y=heatmap_df.index,
            color_continuous_scale="Viridis",
            title="Medal Counts Heatmap"
        )
        st.plotly_chart(fig)

    # Bar Chart
    elif visualization_type == "Bar Chart":
        st.subheader("Bar Chart - Medal Counts Over the Years")
        group_by = st.sidebar.radio("Group Bars By", ["Country", "Year"])
        if group_by == "Country":
            fig = px.bar(
                df,
                x='Country_x',
                y='Medal Count',
                color='Year',
                title="Medal Counts by Country",
                barmode="group"
            )
        else:
            fig = px.bar(
                df,
                x='Year',
                y='Medal Count',
                color='Country_x',
                title="Medal Counts by Year",
                barmode="group"
            )
        st.plotly_chart(fig)

    # Choropleth Map
    elif visualization_type == "Choropleth Map":
        st.subheader("Choropleth Map - Geographic Visualization of Medal Counts")
        fig = px.choropleth(
            df,
            locations='Country_x',
            locationmode='ISO-3',
            color='Medal Count',
            hover_name='Country_x',
            animation_frame='Year',
            title="Medal Counts Over the Years (Choropleth Map)"
        )
        st.plotly_chart(fig)



