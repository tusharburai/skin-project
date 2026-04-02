import numpy as np
from backend.model_loader import load_model
from utils.preprocess import preprocess_image
from utils.labels import labels
from backend.model_loader import load_model
def predict_image(image):
    model = load_model()
    processed = preprocess_image(image)

    prediction = model.predict(processed)
    class_index = np.argmax(prediction)

    return labels[class_index]