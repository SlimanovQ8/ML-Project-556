import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def arima_model_app(fig=None):
    # Page Title and Header
    st.title("🏅 Olympic Medal Trend Visualization")
    st.write(
        """
        Welcome to the Olympic Medal Trend Analysis! 
        This section visualizes historical trends and forecasts in Olympic medal counts using DecisionTree Base Models.
        """
    )
    
    # Content Display
    if fig is None:
        st.subheader("No Forecast Data Available")
        st.warning("No Model forecast provided for visualization.")
        
        # st.info(
        #     """
        #     To explore medal trends:
        #     - Navigate to the **ARIMA Model** section.
        #     - Select a country to generate forecasts.
            
        #     **Note:** This app is a demo and currently does not display actual data or models.
        #     """
        # )
       # st.image(
        #     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Olympic_rings_without_rims.svg/512px-Olympic_rings_without_rims.svg.png",
        #     width=200,
        # )  # st.image(
        #     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Olympic_rings_without_rims.svg/512px-Olympic_rings_without_rims.svg.png",
        #     width=200,
        # )

    else:
        st.subheader("Medal Forecast Visualization")
        st.write("Here is the Selected Model forecast for the selected country:")
        # st.pyplot(fig)
        st.plotly_chart(fig, key="line_chart_1")

        # Placeholder for further insights
        st.write("---")
        st.info(
            """
            **Next Steps:**
            - Analyze additional countries in the **Model** section.
            - Compare forecasts with actual trends to identify key insights.
            """
        )
