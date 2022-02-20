from sklearn.preprocessing import StandardScaler
import pickle

def predict(arr):
    # Load the model
    with open('boston_model.sav', 'rb') as f:
        model = pickle.load(f)
    with open('input_scaler.sav', 'rb') as f:
        scaler = pickle.load(f)
    print(arr)
    scaled_input = scaler.transform([arr])
    print(scaled_input)
    preds = model.predict(scaled_input)
    return preds