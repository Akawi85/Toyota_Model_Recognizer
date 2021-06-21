"""
Author: Akawi Ifeanyi Courage
E-mail address: Ifeanyi.akawi85@gmail.com
Purpose of program: Predict Toyota model
"""
# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
from keras.preprocessing import image
import tensorflow as tf

# load model
interpreter_file = tf.lite.Interpreter(model_path="./model_dir/10x_tflite.tflite")
interpreter_file.allocate_tensors()

print('@@ Model loaded')

# -----------------------------------------------------------------------------------
def predict_image(interpreter, imagefile):
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_image = image.load_img(imagefile, target_size = (224, 224))
    test_image = image.img_to_array(test_image)/255
    test_image = np.expand_dims(test_image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_details[0]['index'])
    digit = np.argmax(output()[0])
    print('--->>>', digit)
    
    pred_class = {
               0: 'Toyota 4runner', 1: 'Toyota Avalon',
               2: 'Toyota_Camry', 3: 'Toyota Corolla',
               4: 'Toyota FJCruiser', 5: 'Toyota Hiace',
               6: 'Toyota Hilux', 7: 'Toyota LandCruiser',
               8: 'Toyota Rav4', 9: 'Toyota Sienna'
               }
    return 'The Car Model Is {}!'.format(pred_class.get(digit))

# ------------>>Toyota model prediction<<--end

# Create flask instance
app = Flask(__name__)


#----------------------------------------------------------------
#render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    """ Display the home page"""

    return render_template('./index.html')


# ---------------------------------------------
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    """
    Save the uploaded image, predict the uploaded image class
    And render the ./predict.html page
    """

    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename
        print("@@ Input posted = ", filename)
        file_path = './static/user_uploaded/'+str(filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred_file = predict_image(interpreter=interpreter_file, imagefile=file_path)

        return render_template('./predict.html',
                               pred_output=pred_file, user_image=file_path)
        
# For local system
# write the main function
if __name__ == '__main__':
    app.run(debug=True)
