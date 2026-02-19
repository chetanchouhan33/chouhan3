from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load built-in dataset
data = load_iris()

X = data.data      # features
y = data.target    # labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Check accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
