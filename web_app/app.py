#Import necessary libraries
import os
import pandas     as pd
import numpy      as np
import tensorflow as tf

from flask                     import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Create flask instance
app = Flask(__name__) # current file

#Set Max size of file as 5 MB.
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
allowed_ext = ['png', 'jpg', 'jpeg']

# Classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_ext



def init():
	tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
	
	global model
	model = tf.keras.models.load_model(
		            os.path.join('static/models', 'model.h5')
		            )




# Function to load and prepare the image in right shape
def read_image(filename):
    # Load image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    
    # Convert the image to array
    img = img_to_array(img)
    
    # Reshape the image into a sample of 1 channel, input_shape for the keras model
    img = img.reshape(1, 28, 28, 1)

    # Pixel data
    img = img.astype('float32')
    img = img / 255.
    
    return img





@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')





@app.route('/predict', methods = ['GET','POST'])
def predict():
	if request.method == 'POST':
		file = request.files['file']
		try:
			if file and allowed_file(file.filename):
				filename  = file.filename
				file_path = os.path.join('static/images', filename)
				
				file.save(file_path)
				img = read_image(file_path)


				# Prediction
				pred_label = model.predict_classes(img) # e.g: pred_label = [7], -> index = pred_label[0] = 7

				# Map apparel category with the numerical class
				product = class_names[pred_label[0]]

				return render_template('predict.html', product = product, user_image = file_path)
		except Exception as e:
			return 'Error! Please check the file extension.'

	return render_template('predict.html')





if __name__ == "__main__":
    init()
    app.run() # args: port 5000
