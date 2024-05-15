import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
data = pd.read_csv("MediCure_BigDataSet.csv")

X = data.drop(columns=['Disease'])
y = data['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier()
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Training {name} model...")

@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.json['symptoms']
    print("Symptoms:", symptoms)

    # Convert symptoms to a list
    symptoms_list = [symptoms['Fever'], symptoms['Cough'], symptoms['Headache'], symptoms['Sense_of_smell'], symptoms['Vomiting']]
    print("Symptoms List:", symptoms_list)

    predictions = {}
    accuracies = {}
    for name, model in models.items():
        # Predict disease for new symptoms
        predicted_disease = model.predict([symptoms_list])
        predictions[name] = predicted_disease[0]
        print(f"Predicted disease with {name} model:", predicted_disease[0])

        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        print(f"{name} Accuracy:", accuracy)

    # Find the model with the highest accuracy
    best_model = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model]
    print("Best Model:", best_model)
    print("Best Accuracy:", best_accuracy)

    return jsonify({
        "predictions": predictions,
        "accuracies": accuracies,
        "best_model": best_model,
        "best_accuracy": best_accuracy
    })

if __name__ == "__main__":
    app.run(debug=True)
