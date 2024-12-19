import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

casualty_data = pd.read_csv(r"C:\Users\Darvesh Valecha\Desktop\assignment topics final\dft-road-casualty-statistics-casualty-2023.csv", low_memory = False )
collision_data = pd.read_csv(r"C:\Users\Darvesh Valecha\Desktop\assignment topics final\dft-road-casualty-statistics-collision-2023.csv", low_memory = False)
vehicle_data = pd.read_csv(r"C:\Users\Darvesh Valecha\Desktop\assignment topics final\dft-road-casualty-statistics-vehicle-2023.csv", low_memory = False)

collision_casualty_data = pd.merge(collision_data, casualty_data, on="accident_index", how="inner")
combined_data = pd.merge(collision_casualty_data, vehicle_data, on="accident_index", how="inner")
urban_data = combined_data[combined_data['urban_or_rural_area'] == 1]

columns_to_keep = [
    'accident_index', 'accident_severity', 'date', 'time','day_of_week',
    'weather_conditions', 'light_conditions', 'road_surface_conditions',
    'latitude', 'longitude', 'number_of_vehicles', 'vehicle_type','sex_of_driver', 'age_of_driver'
]

final_data = urban_data[columns_to_keep]
cleaned_file_path = r"C:\Users\Darvesh Valecha\Desktop\assignment topics final\final_data_urban.csv"
final_data.to_csv(cleaned_file_path,index = False)

# Load Data
final_data = pd.read_csv(r"C:\Users\Darvesh Valecha\Desktop\assignment topics final\final_data_urban.csv")

# Map Categorical Variables
weather_map = {
    1: 'Fine no high winds',
    2: 'Raining no high winds',
    3: 'Snowing no high winds',
    4: 'Fine high winds',
    5: 'Raining high winds',
    6: 'Snowing high winds',
    7: 'Fog or mist',
    8: 'Other',
    -1: 'Unknown'
}
final_data['weather_conditions'] = final_data['weather_conditions'].map(weather_map)

road_surface_map = {
    1: 'Dry',
    2: 'Wet or damp',
    3: 'Snow',
    4: 'Frost or ice',
    5: 'Flood over 3cm deep',
    -1: 'Unknown'
}
final_data['road_surface_conditions'] = final_data['road_surface_conditions'].map(road_surface_map)

light_map = {
    1: 'Daylight',
    4: 'Darkness - lights lit',
    5: 'Darkness - lights unlit',
    6: 'Darkness - no lighting',
    7: 'Darkness - lighting unknown',
    -1: 'Unknown'
}
final_data['light_conditions'] = final_data['light_conditions'].map(light_map)


final_data['hour_of_day'] = pd.to_datetime(final_data['time'], format='%H:%M', errors='coerce').dt.hour
bins = [0, 18, 30, 50, 70, 100]
labels = ['<18', '18-30', '31-50', '51-70', '>70']
final_data['age_group'] = pd.cut(final_data['age_of_driver'], bins=bins, labels=labels, right=False)


map = folium.Map(location=[final_data['latitude'].mean(), final_data['longitude'].mean()], zoom_start=10)
HeatMap(data=final_data[['latitude', 'longitude']].dropna(), radius=10).add_to(map)
map.save(r"C:\Users\Darvesh Valecha\Desktop\assignment topics final\heatmap.html")


final_data.to_csv(r"C:\Users\Darvesh Valecha\Desktop\assignment topics final\processed_data.csv", index=False)









