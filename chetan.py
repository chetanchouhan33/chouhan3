import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load data (using a sample dataset)
melbourne_file_path = 'melb_data.csv'
data = pd.read_csv(melbourne_file_path)

# Drop missing values
data = data.dropna(axis=0)

# Select target and features
y = data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]

# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define Model
model = DecisionTreeRegressor(random_state=1)

# Fit Model
model.fit(train_X, train_y)

# Predict
predictions = model.predict(val_X)
print(mean_absolute_error(val_y, predictions))
