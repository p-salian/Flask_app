from flask import Flask, request, render_template, redirect, url_for
import pickle
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model_2.pkl', 'rb') as file:
    model_2 = pickle.load(file)


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


label_map = {
    0: "None",
    1: "Building",
    2: "Forest",
    3: "Glacier",
    4: "Sea",
    5: "Mountain",
    6: "Street"
}

def preprocess_image(image_path):
    # Preprocess the image (example: resize and flatten)
    image = Image.open(image_path).convert('L')
    image = image.resize((128, 128))
    image_array = np.array(image).flatten()
    return image_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            
            image_array = preprocess_image(file_path).reshape(1, -1)

            
            prediction_1 = model.predict(image_array)
            prediction_2 = model_2.predict(image_array)

            
            prediction_1_name = label_map[prediction_1[0]]
            prediction_2_name = label_map[prediction_2[0]]

            return render_template('index.html', prediction_1=prediction_1_name, prediction_2=prediction_2_name)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
