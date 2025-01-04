import pandas as pd

data = pd.read_csv('lambertian.csv')
print(data.columns)

data_shifted_plus = data.copy()
data_shifted_plus['x'] += 45

data_shifted_minus = data.copy()
data_shifted_minus['x'] -= 45

combined_data = data.copy()
combined_data['y'] += data_shifted_plus['y'] + data_shifted_minus['y']

combined_data.to_csv('combined_lambertian.csv', index=False)