 # FastAPI/Flask app
from flask import Flask, render_template, request, redirect, url_for
import os
from inference import load_model, predict_image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model()
#{'no_damage': 0, 'visible_damage': 1}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            upload_dir = os.path.join('static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)  # ✅ Creates directory if missing
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)

            label, conf = predict_image(model, filepath)
            prediction = "Damage" if label == 1 else "No Damage"
            confidence = f"{conf:.2f}"
            image_url = filepath

    return render_template('index.html', prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)