# [Toyota_Model_Recognizer](https://toyota-model-recognizer.herokuapp.com/)
This Web App uses the [EfficientNet_Lite0](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite) model to build an API that predicts the model of toyota vehicles.

### Steps
The `scrape_car_images_using_selenium.ipynb` notebook was executed locally to scrape one thousand, one hundred (1100) images of toyota vehicles from google. The scrapper only scrapped for 10 vehicle classes, namely:
- `toyota_4runner`
- `toyota_avalon`
- `toyota_camry`
- `toyota_corolla`
- `toyota_fjcruiser`
- `toyota_hiace`
- `toyota_hilux`
- `toyota_landcruiser`
- `toyota_rav4`
- `toyota_sienna`

These classes were selected on the basis of brand model popularity within the Nigerian context. The selection followed no particular hierarchy. All selected vehicle classes had 110 images each in a class-separated folder.

Link to the dataset can be found [here](https://www.dropbox.com/sh/3ivwc3mhdvxp3sy/AACs-jcAJqE2E3sPyK2DYq5Za?dl=0)

### [DataLoader](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/image_classifier/DataLoader)
The `Image_Classification_with_TensorFlow_Lite_Model_Maker.ipynb` notebook was executed in google colab for efficeincy and speed. The images from the folders were loaded using the `DataLoader` function from `tflite_model_maker.image_classifier` class. The `DataLoader` function together with the `from_folder` method was used to load images from subdirectories, identifying the subdirectory names as the class labels.

### Model Training
The default `EfficientNet_Lite0` model from `tflite_model_maker.image_classifier.image_classifier.create` class was used to train the images for 50 epochs achieving an accuracy score of `72.7%` on the validation dataset. This is considered a good score considering the very small volume of data at the model's disposal.

### Model Compression
A pre-trained [VGG19](https://keras.io/api/applications/vgg/) model was first used to train the images as seen in the `keras_vgg19_toyota_model_recognizer.ipynb` notebook. The top layer was removed to accomodate for a fully connected Dense Layer with `1024` neurons, a `relu` activation function, batchnormalization and dropout layers. `Softmax` activation function was applied at the output layer for multiclass classification (in this case 10 classes). The size of the model after training was 347mb. This large file size caused redundancies in deployment, as github does not permit pushing of files greater than 100mb.  
For this reason the model was compressed by following the steps outlined in [this official tensorflow doc](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras) for optimization and quantization of models. This helped reduce the model size x10, to about 43mb, while maintaining the model's predictive accuracy.  
The `compress_keras_model.ipynb` notebook shows the code for this.  The backlog of this compression was that the compressed model took about 1 min 20 seconds to perform a single prediction, which resulted in very bad user experience.

To further reduce the model size the [tflite_model_maker](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification) package was used to build a `.tflite` model and achieved an even better efficiency (model accuracy of `0.727` and model portability of `3.8mb`) compared to the baseline model which had an accuracy score of `0.70` with a size of `370.7 mb` and the compressed model which had an accuracy score of `0.67` with a size of `47mb`.

The `tflite_model_maker` model is stored in the `model_dir` folder as `model.tflite`.  
The fully trained VGG19 model together with the compressed model is found [here](https://drive.google.com/drive/folders/1ADfccMceNSsrVBJxOhUlA9e2plaa_Cmu?usp=sharing)

# Running the service...
- Click [here](https://toyota-model-recognizer.herokuapp.com/) to run a prediction  
- Click on `Choose File` to select an image
- Select a toyota image from your local machine
- Click on the `Predict` button to the right
- Wait a few seconds for the system to process and predict the image class.
- Viola!!! Here you have your prediction.

# Snapshot of Web App  
#### (The home page)  
![Home page of web app](./static/home.png)  

#### The prediction Page
![Prediction page of web app](./static/predict.png)  

# Further Steps
- [ ] Create a more sophisticated web interface
- [ ] scrape more image dataset to create an even better model
- [x] Train model using Transfer learning and compare performance 
- [ ] Include more classes

*PS:* *The `Image_Classification_with_TensorFlow_Lite_Model_Maker.ipynb` notebook was executed with the help of [this tensorflow tutorial](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification#scrollTo=zNDBP2qA54aK)*

