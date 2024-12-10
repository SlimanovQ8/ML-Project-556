import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.configs.logging_file import logging

class VisualizationPlotly:

    def __init__(self):
        pass

    def plots_individual_plots_and_general_trend(
        self,
        train_gold=None,
        train_silver=None,
        train_bronze=None,
        test_gold=None,
        test_silver=None,
        test_bronze=None,
        forecast_df_silver=None,
        forecast_df_bronze=None,
        forecast_df_gold=None,
        dta_gold=None,
        dta_bronze=None,
        dta_silver=None,
        general_train=None,
        general_test=None,
        general_forecast=None,
        gold_metrics=None,
        silver_metrics=None,
        bronze_metrics=None,
        general_metrics=None
    ):
        try:
            # Create subplots: 5 rows and 1 column
            fig = make_subplots(rows=5, cols=1, subplot_titles=[
                f'Gold Medals: Observed vs Forecasted : {gold_metrics}',
                f'Silver Medals: Observed vs Forecasted : {silver_metrics}',
                f'Bronze Medals: Observed vs Forecasted : {bronze_metrics}',
                f'General Trend: Total Medals Trend : {general_metrics}',
                'Medals Breakdown per Year'
            ])

            
            # Create subplots: 5 rows and 1 column
            fig = make_subplots(rows=5, cols=1, subplot_titles=[
                f'Gold Medals: Observed vs Forecasted : {gold_metrics}',
                f'Silver Medals: Observed vs Forecasted : {silver_metrics}',
                f'Bronze Medals: Observed vs Forecasted : {bronze_metrics}',
                f'General Trend: Total Medals Trend : {general_metrics}',
                'Medals Breakdown per Year'
            ])

            # Gold Medals
            fig.add_trace(go.Scatter(x=train_gold.index, y=train_gold['Gold_Medals'], mode='lines+markers',
                                    name='Gold Train Data', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=test_gold.index, y=test_gold['Gold_Medals'], mode='lines+markers',
                                    name='Gold Test Data', line=dict(color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast_df_gold.index, y=forecast_df_gold['Forecasted_Bronze_Medals'], mode='lines+markers',
                                    name='Gold Forecasted', line=dict(color='orange')), row=1, col=1)

            # Silver Medals
            fig.add_trace(go.Scatter(x=train_silver.index, y=train_silver['Silver_Medals'], mode='lines+markers',
                                    name='Silver Train Data', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=test_silver.index, y=test_silver['Silver_Medals'], mode='lines+markers',
                                    name='Silver Test Data', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=forecast_df_silver.index, y=forecast_df_silver['Forecasted_Bronze_Medals'], mode='lines+markers',
                                    name='Silver Forecasted', line=dict(color='orange')), row=2, col=1)

            # Bronze Medals
            fig.add_trace(go.Scatter(x=train_bronze.index, y=train_bronze['Bronze_Medals'], mode='lines+markers',
                                    name='Bronze Train Data', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=test_bronze.index, y=test_bronze['Bronze_Medals'], mode='lines+markers',
                                    name='Bronze Test Data', line=dict(color='green')), row=3, col=1)
            fig.add_trace(go.Scatter(x=forecast_df_bronze.index, y=forecast_df_bronze['Forecasted_Bronze_Medals'], mode='lines+markers',
                                    name='Bronze Forecasted', line=dict(color='orange')), row=3, col=1)

            # General Trend
            fig.add_trace(go.Scatter(x=general_train.index, y=general_train['Total_Medals'], mode='lines+markers',
                                    name='General Train Data', line=dict(color='blue')), row=4, col=1)
            fig.add_trace(go.Scatter(x=general_test.index, y=general_test['Total_Medals'], mode='lines+markers',
                                    name='General Test Data', line=dict(color='silver')), row=4, col=1)
            fig.add_trace(go.Scatter(x=general_forecast.index, y=general_forecast['Total_Medals'], mode='lines+markers',
                                    name='General Forecasted', line=dict(color='brown')), row=4, col=1)

            # Medals Breakdown per Year (Bar Plot)
            medals_data_train = general_train.copy()  # Combine general_train and general_test if needed
            medals_data_test = general_test.copy()  # Combine general_train and general_test if needed
            logging.info(f"columns : {list(general_forecast)}")
            # Medals Breakdown per Year (Bar Plot)
            fig.add_trace(go.Bar(
                x=general_train.index,
                y=general_train['Gold_Medals'],
                name='Gold Medals (Train)',
                marker_color='gold'
            ), row=5, col=1)
            fig.add_trace(go.Bar(
                x=general_train.index,
                y=general_train['Silver_Medals'],
                name='Silver Medals (Train)',
                marker_color='silver'
            ), row=5, col=1)
            fig.add_trace(go.Bar(
                x=general_train.index,
                y=general_train['Bronze_Medals'],
                name='Bronze Medals (Train)',
                marker_color='brown'
            ), row=5, col=1)
            fig.add_trace(go.Bar(
                x=general_test.index,
                y=general_test['Gold_Medals'],
                name='Gold Medals (Test)',
                marker_color='gold'
            ), row=5, col=1)
            fig.add_trace(go.Bar(
                x=general_test.index,
                y=general_test['Silver_Medals'],
                name='Silver Medals (Test)',
                marker_color='silver'
            ), row=5, col=1)
            fig.add_trace(go.Bar(
                x=general_test.index,
                y=general_test['Bronze_Medals'],
                name='Bronze Medals (Test)',
                marker_color='brown'
            ), row=5, col=1)
            fig.add_trace(go.Bar(
                x=general_forecast.index,
                y=  ['Forecasted_gold_Medals'],
                name='Predicted Gold Medals',
                marker_color='gold'
            ), row=5, col=1)
            fig.add_trace(go.Bar(
                x=general_forecast.index,
                y=general_forecast['Forecasted_silver_Medals'],
                name='Predicted Silver Medals',
                marker_color='silver'
            ), row=5, col=1)
            fig.add_trace(go.Bar(
                x=general_forecast.index,
                y=general_forecast['Forecasted_bronze_Medals'],
                name='Predicted Bronze Medals',
                marker_color='brown'
            ), row=5, col=1)

            # Update layout
            fig.update_layout(
                height=1500,  # Adjust plot height
                width=1000,  # Adjust plot width
                title_text="Medals Visualization with Breakdown",  # Main title
                showlegend=True,  # Show legends
                barmode='stack'  # Stack bar charts
            )

            # Update axes labels
            for i in range(1, 5):
                fig.update_xaxes(title_text="Year", row=i, col=1)
                fig.update_yaxes(title_text="Number of Medals", row=i, col=1)

            fig.update_xaxes(title_text="Year", row=5, col=1)
            fig.update_yaxes(title_text="Medals Count", row=5, col=1)

            # Save the plot as an HTML file
            fig.write_html("interactive_medals_plot_with_bars.html")
            return fig

        except Exception as e:
            logging.error(f"\nError in `plots_individual_plots_and_general_trend`:\n{e}\n", exc_info=True)
            return None
        
    def plot_forecasting_values_only(self,
                                 time_series_gold=None,
                                 time_series_silver=None,
                                 time_series_bronze=None,
                                 gold_forecasted_df=None,
                                 silver_forecasted_df=None,
                                 bronze_forecasted_df=None,
                                 country_name=None,
                                 historical_data=None,
                                 forecasted_data=None):
        try:
            # Create subplots: 5 rows and 1 column
            fig = make_subplots(rows=5, cols=1, subplot_titles=[
                f'Gold Medals: Observed vs Forecasted',
                f'Silver Medals: Observed vs Forecasted',
                f'Bronze Medals: Observed vs Forecasted',
                f'General Trend: Total Medals',
                'Medals Breakdown per Year'
            ])
            
            logging.info(f"time_series_bronze : columns  : {list(time_series_bronze)}")
            logging.info(f"time_series_bronze : columns  : {list(time_series_bronze)}")

            # Gold Medals
            fig.add_trace(go.Scatter(
                x=time_series_gold['Year'], 
                y=time_series_gold['Gold_Medals'], 
                mode='lines+markers',
                name='Gold Observed', 
                line=dict(color='blue')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=gold_forecasted_df['Year'], 
                y=gold_forecasted_df['Gold_Medals'], 
                mode='lines+markers',
                name='Gold Forecasted', 
                line=dict(color='orange')
            ), row=1, col=1)

            # Silver Medals
            fig.add_trace(go.Scatter(
                x=time_series_silver['Year'], 
                y=time_series_silver['Silver_Medals'], 
                mode='lines+markers',
                name='Silver Observed', 
                line=dict(color='blue')
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=silver_forecasted_df['Year'], 
                y=silver_forecasted_df['Silver_Medals'], 
                mode='lines+markers',
                name='Silver Forecasted', 
                line=dict(color='orange')
            ), row=2, col=1)

            # Bronze Medals
            fig.add_trace(go.Scatter(
                x=time_series_bronze['Year'], 
                y=time_series_bronze['Bronze_Medals'], 
                mode='lines+markers',
                name='Bronze Observed', 
                line=dict(color='blue')
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=bronze_forecasted_df['Year'], 
                y=bronze_forecasted_df['Bronze_Medals'], 
                mode='lines+markers',
                name='Bronze Forecasted', 
                line=dict(color='orange')
            ), row=3, col=1)


            fig.add_trace(go.Scatter(
                x=historical_data['Year'], 
                y=historical_data['Total_Medals'], 
                mode='lines+markers',
                name='Total Observed', 
                line=dict(color='blue')
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=forecasted_data['Year'], 
                y=forecasted_data['Total_Medals'],
                mode='lines+markers',
                name='Total Forecasted', 
                line=dict(color='orange')
            ), row=4, col=1)

            # Medals Breakdown per Year (Bar Plot)
            # for forecast_df, medal, color in zip(
            #     [gold_forecasted_df, silver_forecasted_df, bronze_forecasted_df],
            #     ['Gold', 'Silver', 'Bronze'],
            #     ['gold', 'silver', 'brown']
            # ):
            #     fig.add_trace(go.Bar(
            #         x=forecast_df.index,
            #         y=forecast_df[f'Forecasted_{medal}_Medals'],
            #         name=f'Forecasted {medal} Medals',
            #         marker_color=color
            #     ), row=5, col=1)

            # Update layout
            fig.update_layout(
                height=1500,
                width=1000,
                title_text=f"Medals Visualization for {country_name}",
                showlegend=True,
                barmode='stack'
            )

            # Update axes labels
            for i in range(1, 6):
                fig.update_xaxes(title_text="Year", row=i, col=1)
                fig.update_yaxes(title_text="Number of Medals", row=i, col=1)

            # Save the plot as an HTML file
            filename = f"{country_name}_medals_forecast.html" if country_name else "interactive_medals_plot_with_bars.html"
            fig.write_html(filename)

            return fig

        except Exception as e:
            logging.error(f"Error while plotting forecasting values: {e}")
            return None

       



    def __call__(self):
        pass
