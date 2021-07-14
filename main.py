import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from source.utils import read_img, prepare_image, draw_rectangles
from source.face_recognition import detect_faces, recognize_faces, training_face

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/train')
def train():
	return render_template('train.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'image' not in request.files:
		flash('No file part')
		return redirect(request.url)
	
	file = request.files['image']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	
	if file and allowed_file(file.filename):
		# Read image
		image = read_img(file)
		faces = recognize_faces(image)
		draw_rectangles(image, faces)
		# Prepare image for html
		to_send = prepare_image(image)

		return render_template('index.html', 
			face_recognized=len(faces)>0, num_faces=len(faces), image_to_show=to_send, init=True)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/train', methods=['POST'])
def train_image():
	if 'image' not in request.files:
		flash('No file part')
		return redirect(request.url)
	
	file = request.files['image']
	name = request.form['name']

	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if name == '':
		flash('Enter name of candidate')
		return redirect(request.url)
	
	if file and allowed_file(file.filename):
		# Read image
		image = read_img(file)
		# Detect image
		faces_pos = detect_faces(image)
		if len(faces_pos)==0:
			flash('No face detected')
			return redirect(request.url)
		if len(faces_pos)>1:
			flash('You should choose image with one face')
			return redirect(request.url)

		# Train model
		faces = training_face(image, name)
		print(faces)
		draw_rectangles(image, faces)
		# Prepare image for html
		to_send = prepare_image(image)

		return render_template('train.html', 
			face_recognized=len(faces)>0, num_faces=len(faces), image_to_show=to_send, init=True)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)