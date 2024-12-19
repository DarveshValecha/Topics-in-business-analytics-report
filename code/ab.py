import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from folium import Map
from folium.plugins import HeatMap


file_path = r"C:\Users\Darvesh Valecha\Desktop\assignment topics final\final_data_urban.csv"  
data = pd.read_csv(file_path)

data = data.dropna(subset=['weather_conditions', 'road_surface_conditions', 'latitude', 'longitude'])


data['hour_of_day'] = pd.to_datetime(data['time'], format='%H:%M', errors='coerce').dt.hour
bins = [0, 18, 30, 50, 70, 100]
labels = ['<18', '18-30', '31-50', '51-70', '>70']
data['age_group'] = pd.cut(data['age_of_driver'], bins=bins, labels=labels, right=False)


temporal_pivot = data.pivot_table(index='day_of_week', columns='hour_of_day', values='accident_severity', aggfunc='count', fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(temporal_pivot, cmap='coolwarm', annot=False)
plt.title('Heatmap of Accidents by Day and Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Day of the Week')
plt.xticks(rotation=45)
plt.yticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=0)
plt.tight_layout()
plt.show()


severity_distribution = data['accident_severity'].value_counts(normalize=True)
plt.figure(figsize=(8, 5))
severity_distribution.plot(kind='bar', color=['blue', 'orange', 'red'])
plt.title('Accident Severity Distribution')
plt.xlabel('Severity')
plt.ylabel('Proportion')
plt.xticks([0, 1, 2], ['Slight', 'Serious', 'Fatal'])
plt.show()

map = Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=10)
HeatMap(data=data[['latitude', 'longitude']].dropna(), radius=10).add_to(map)
map.save("heatmap.html")  


X = pd.get_dummies(data[['hour_of_day', 'weather_conditions', 'road_surface_conditions']], drop_first=True)
y = data['accident_severity']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)



print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logistic))

conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
disp_logistic = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_logistic, display_labels=['Slight', 'Serious', 'Fatal'])
disp_logistic.plot(cmap='Blues')
plt.title('Confusion Matrix: Logistic Regression')
plt.show()

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=['Slight', 'Serious', 'Fatal'])
disp_rf.plot(cmap='Greens')
plt.title('Confusion Matrix: Random Forest')
plt.show()


feature_importances = rf_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.title('Feature Importance: Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()





pairplot_data = data[['weather_conditions', 'road_surface_conditions', 
                      'light_conditions', 'accident_severity']]


sns.set(style="ticks", font_scale=1.2)
pair_plot = sns.pairplot(pairplot_data, hue='accident_severity', diag_kind='kde', palette='viridis')
pair_plot.fig.suptitle("Pair Plot of Environmental Factors and Accident Severity", y=1.02)
plt.savefig("pair_plot_environmental_factors.png", dpi=300)
plt.show()
