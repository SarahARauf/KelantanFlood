import streamlit as st
import streamlit.components.v1 as components

import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

import folium
from streamlit_folium import st_folium

import json



with open('Flood_Prediction_model.pkl', 'rb') as f:
    model = joblib.load('Flood_Prediction_model.pkl')

with open('minmax_flood.pkl', 'rb') as s:
    scaler = joblib.load('minmax_flood.pkl')


with open('description.json', 'r') as f:
        weather_data = json.load(f)

def obtain_latlong(selected_town):
    marker_locations = {
        'Pasir Mas': [6.0493, 102.1399],
        'Kota Bharu': [6.1236, 102.2433],
        'Bachok': [6.0667, 102.4],
        'Kuala Krai': [5.5313, 102.1993],
        'Pasir Puteh': [5.8333, 102.4],
        'Tumpat': [6.1978, 102.171],
        'Gua Musang': [4.8844, 101.9686],
        'Machang': [5.767933, 102.215385],
        'Tanah Merah': [5.8, 102.15],
        'Jeli': [5.7007, 101.8432]
    }
    return marker_locations.get(selected_town)

def fetch_weather_icon(weather_code, is_day):
    # Load the weather data from the JSON file

    # Get the weather information based on the weather code
    weather_info = weather_data[str(weather_code)]
    print(weather_info)

    if weather_info:
        # Get the day or night description based on the time of day
        weather_description = weather_info.get('day' if is_day else 'night')
        if weather_description:
            return weather_description.get('image'), weather_description.get('description')
    return None

def display_image_url(image_url):
    html_code = f'<img src="{image_url}" alt="Weather Icon">'
    st.markdown(html_code, unsafe_allow_html=True)

def display_map(selected_town):

  latlong = obtain_latlong(selected_town)



  map=folium.Map(location=[5.342013,102.035270], zoom_start=8.4)


  folium.Marker(location=latlong,
  tooltip=selected_town,
  tiles='Stamen Toner').add_to(map)

  st_map = st_folium(map, width=700, height=450)




def preprocess_input(hourly_dataframe, daily_dataframe):

    daily_dataframe['date'] = daily_dataframe['date'].dt.strftime('%m/%d/%Y')
    hourly_dataframe['new_date']=hourly_dataframe['date'].dt.strftime('%m/%d/%Y')
    hourly_dataframe['new_time']=hourly_dataframe['date'].dt.strftime('%H:%M')
    hourly_dataframe['date'] = hourly_dataframe['date'].dt.strftime('%m/%d/%Y %H:%M')
    merged_df = pd.merge(hourly_dataframe, daily_dataframe, left_on='new_date', right_on='date', how='left')



    columns_name = [col for col in merged_df.columns if col not in ['index', 'time', 'Flood occurrence', 'Geo Locations', 'date_x', 'date_y', 'new_date', 'new_time', "rain", "precipitation_probability", "weather_code"]]
    normalized_df = scaler.transform(merged_df[columns_name])
    normalized_df = pd.DataFrame(normalized_df, index = merged_df.index, columns=columns_name)
    normalized_df = pd.merge(normalized_df, merged_df[['date_x', 'new_date']], left_on=normalized_df.index, right_on=merged_df.index, how='left', left_index=False, right_index=False)

    normalized_df=normalized_df.drop(columns=['key_0'])
    normalized_df=normalized_df.rename(columns={'date_x':'date', 'new_date':'only_date'})

    dict_normname = {col:'Normalized_'+col for col in normalized_df.columns if col not in ['date', 'only_date']}
    normalized_df = normalized_df.rename(columns=dict_normname)
    
    return normalized_df, merged_df

def predict(input_data):
    prediction = model.predict(input_data.drop(['date', 'only_date'], axis=1))
    return prediction

def fetch_API(selected_town):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)


    latlong = obtain_latlong(selected_town)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latlong[0],
        "longitude": latlong[1],
        "hourly": ["temperature_2m", "relative_humidity_2m", "cloud_cover", "rain", "precipitation_probability", "weather_code"],
        "daily": ["temperature_2m_max", "rain_sum", "precipitation_hours"],
	"forecast_days": 7
 #        "start_date": "2023-12-17",
	# "end_date": "2023-12-28"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation_probability = hourly.Variables(2).ValuesAsNumpy()
    hourly_rain = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(5).ValuesAsNumpy()


    hourly_data = {"date": pd.date_range(
      start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
      end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
      freq = pd.Timedelta(seconds = hourly.Interval()),
      inclusive = "left"
    )}
    hourly_data["temperature_2m (°C)"] = hourly_temperature_2m
    hourly_data["relativehumidity_2m (%)"] = hourly_relative_humidity_2m
    hourly_data["cloudcover (%)"] = hourly_cloud_cover
    hourly_data["precipitation_probability"] = hourly_precipitation_probability
    hourly_data["rain"] = hourly_rain
    hourly_data["weather_code"] = hourly_weather_code


    hourly_dataframe = pd.DataFrame(data = hourly_data)
    #print(hourly_dataframe)

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(2).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["temperature_2m_max (°C)"] = daily_temperature_2m_max
    daily_data["rain_sum (mm)"] = daily_rain_sum
    daily_data["precipitation_hours (h)"] = daily_precipitation_hours

    daily_dataframe = pd.DataFrame(data = daily_data)
    #print(daily_dataframe)

    return hourly_dataframe, daily_dataframe

# Streamlit app
def main():
    st.title("⛈️ Kelantan Flood Prediction")

    selected_town = st.sidebar.selectbox('Town in Kelantan:', ["Kota Bharu", "Kuala Krai", "Pasir Mas", "Pasir Puteh", "Tumpat", "Gua Musang", "Machang", "Tanah Merah", "Bachok", "Jeli"])

    display_map(selected_town)

    st.write(f"Results For: {selected_town}, Kelantan")

    hourly_dataframe, daily_dataframe = fetch_API(selected_town)

    #input_df = pd.DataFrame([input_dict])

    input_data, display_data = preprocess_input(hourly_dataframe, daily_dataframe)
    #print(input_data.columns)

    unique_dates = sorted(set(input_data['only_date']))
    selected_date = st.sidebar.selectbox('Select a Date:', [date for date in unique_dates])

    prediction = predict(input_data)

    #st.write(type(prediction))

    prediction_fordate = []

    for idx, row in display_data[display_data['new_date']==selected_date].iterrows():
      prediction_fordate.append(prediction[idx])

    if 1 in prediction_fordate:
      st.write("Flood Caution")

    


    with st.container():
        for idx, row in display_data[display_data['new_date']==selected_date].iterrows():
            st.markdown(f"<h3>Date and Time: {row['date_x']}</h3>", unsafe_allow_html=True)


            hour = int(row['new_time'].split(':')[0])
            is_day = (7 < hour and hour < 19)

            current_icon, current_desc = fetch_weather_icon(int(row['weather_code']), is_day)

            display_image_url(current_icon)
            st.write(f"{current_desc}")

            st.write(f"Temperature: {round(row['temperature_2m (°C)'], 1)}°C")
            st.write(f"Relative Humidity: {row['relativehumidity_2m (%)']}%")
            st.write(f"Precipitation Probability: {row['precipitation_probability']}%")
            st.write(f"Precipitation: {round(row['rain'],1)}mm")
            st.write(f"Cloud Cover: {row['cloudcover (%)']}%")
            st.write(f"Prediction: {'Flood' if prediction[idx] == 1 else 'No Flood'}")


            #st.write(f"{row['weather_code']}")
            
            # st.write(int(row['new_time'].split(':')[0]))
            # st.write(7 < hour and hour < 19)



            st.markdown("<hr>", unsafe_allow_html=True)






    #st.subheader('Prediction:')

    # st.write(input_data)
    # st.write(display_data)

    # st.write(prediction.tolist())


if __name__ == '__main__':
    main()

