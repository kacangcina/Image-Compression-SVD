from flask import Flask, render_template, request, send_from_directory, url_for
import os, time
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.float32) / info.max

def compress_image(in_path, rate, out_path):
    img = cv2.imread(in_path)
    if img is None:
        raise ValueError("Invalid image")

    b, g, r = [im2double(img[:, :, i]) for i in range(3)]

    def compress_channel(channel):
        u, s, vt = np.linalg.svd(channel, full_matrices=False)
        k = max(1, int(len(s) * (rate / 100.0)))
        s_k = np.diag(s[:k])
        u_k = u[:, :k]
        vt_k = vt[:k, :]
        compressed = np.dot(u_k, np.dot(s_k, vt_k))
        return np.clip(compressed * 255, 0, 255).astype(np.uint8)

    b_comp = compress_channel(b)
    g_comp = compress_channel(g)
    r_comp = compress_channel(r)

    compressed_img = cv2.merge((b_comp, g_comp, r_comp))
    cv2.imwrite(out_path, compressed_img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress():
    start = time.time()
    file = request.files['image']
    rate = int(request.form['rate'])
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    out_filename = f"compressed_{filename}"
    output_path = os.path.join(app.config['STATIC_FOLDER'], out_filename)
    compress_image(upload_path, rate, output_path)

    elapsed = round(time.time() - start, 2)
    diff = 100 - rate

    return render_template("index.html",
                           before_filename=filename,
                           after_filename=out_filename,
                           diff=diff,
                           time=elapsed)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
