import numpy as np
import joblib
import streamlit as st

# Load the saved model and scaler
model = joblib.load('best_model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# Define the Streamlit app
def run_streamlit_app():
    st.title('Flood Prediction')
    st.subheader('Enter the details below to predict flood probability')

    # Create four columns for layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        monsoon_intensity = st.slider('Monsoon Intensity', 0, 100, 50)
        deforestation = st.slider('Deforestation', 0, 100, 50)
        river_water_flow = st.slider('River Water Flow', 0, 100, 50)
        soil_type = st.slider('Soil Type', 0, 100, 50)
        vegetation_cover = st.slider('Vegetation Cover', 0, 100, 50)

    with col2:
        topography_drainage = st.slider('Topography Drainage', 0, 100, 50)
        urbanization = st.slider('Urbanization', 0, 100, 50)
        land_use_change = st.slider('Land Use Change', 0, 100, 50)
        elevation = st.slider('Elevation', 0, 100, 50)
        temperature = st.slider('Temperature', 0, 100, 50)

    with col3:
        river_management = st.slider('River Management', 0, 100, 50)
        climate_change = st.slider('Climate Change', 0, 100, 50)
        population_density = st.slider('Population Density', 0, 100, 50)
        slope = st.slider('Slope', 0, 100, 50)
        humidity = st.slider('Humidity', 0, 100, 50)

    with col4:
        precipitation = st.slider('Precipitation', 0, 100, 50)
        flood_control_measures = st.slider('Flood Control Measures', 0, 100, 50)
        drainage_density = st.slider('Drainage Density', 0, 100, 50)
        watershed_area = st.slider('Watershed Area', 0, 100, 50)
        wind_speed = st.slider('Wind Speed', 0, 100, 50)

    if st.button('Predict'):
        # Collect input data and scale it using the scaler
        input_data = np.array([[monsoon_intensity, topography_drainage, river_management, deforestation, urbanization, climate_change,
                                precipitation, river_water_flow, land_use_change, population_density, flood_control_measures,
                                soil_type, elevation, slope, drainage_density, watershed_area, vegetation_cover, temperature,
                                humidity, wind_speed]])
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        
        # Display the result
        st.write(f'Predicted Flood Probability: {prediction[0]}')

if __name__ == '__main__':
    run_streamlit_app()
