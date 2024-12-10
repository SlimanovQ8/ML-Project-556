from typing import Any
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})  # Set the font size globally to 12


class Visualization:

    def __init__(self) -> None:
        pass

    def plots_individual_plots_and_general_trend(self,
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
        fig = plt.figure(figsize=(20, 30))  # Adjust size for a taller figure
        gs = gridspec.GridSpec(4, 1, figure=fig)  # 4 rows, 1 column

        # Define subplots
        ax1 = fig.add_subplot(gs[0, 0])  # First row
        ax2 = fig.add_subplot(gs[1, 0])  # Second row
        ax3 = fig.add_subplot(gs[2, 0])  # Third row
        ax4 = fig.add_subplot(gs[3, 0])  # Fourth row

        # Plot Gold Medals
        ax1.plot(train_gold.index, train_gold['Gold_Medals'], label='Train Data', color='blue', marker='o')
        ax1.plot(test_gold.index, test_gold['Gold_Medals'], label='Test Data', color='green', marker='o')
        ax1.plot(forecast_df_gold.index, forecast_df_gold.reset_index()['Forecasted_Bronze_Medals'], label='Forecasted', color='orange', marker='o')
        ax1.set_title(f'Gold Medals: Observed vs Forecasted : {gold_metrics}')
        ax1.set_xticklabels(dta_gold.index, rotation=90)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Medals')
        ax1.legend()

        # Plot Silver Medals
        ax2.plot(train_silver.index, train_silver['Silver_Medals'], label='Train Data', color='blue', marker='o')
        ax2.plot(test_silver.index, test_silver['Silver_Medals'], label='Test Data', color='green', marker='o')
        ax2.plot(forecast_df_silver.index, forecast_df_silver.reset_index()['Forecasted_Bronze_Medals'], label='Forecasted', color='orange', marker='o')
        ax2.set_title(f'Silver Medals: Observed vs Forecasted : {silver_metrics}')
        ax2.set_xticklabels(dta_silver.index, rotation=90)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Medals')
        ax2.legend()

        # Plot Bronze Medals
        ax3.plot(train_bronze.index, train_bronze['Bronze_Medals'], label='Train Data', color='blue', marker='o')
        ax3.plot(test_bronze.index, test_bronze['Bronze_Medals'], label='Test Data', color='green', marker='o')
        ax3.plot(forecast_df_bronze.index, forecast_df_bronze.reset_index()['Forecasted_Bronze_Medals'], label='Forecasted', color='orange', marker='o')
        ax3.set_title(f'Bronze Medals: Observed vs Forecasted : {bronze_metrics}')
        ax3.set_xticklabels(dta_bronze.index, rotation=90)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Number of Medals')
        ax3.legend()

        # General Trend Plot
        ax4.plot(general_train.index, general_train.reset_index()['Total_Medals'], label='Train Medals Count', color='blue', marker='o')
        ax4.plot(general_test.index, general_test.reset_index()['Total_Medals'], label='Test Medal Count', color='silver', marker='o')
        ax4.plot(general_forecast.index, general_forecast.reset_index()['Total_Medals'], label='Forecasted Medal Count', color='brown', marker='o')
        ax4.set_title(f'General Trend: Total Medals Trend : {general_metrics}')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of Medals')
        ax4.legend()

        plt.tight_layout()
        plt.savefig("sample.png")
        return fig


    def __call__(self,) -> Any:
        pass
        