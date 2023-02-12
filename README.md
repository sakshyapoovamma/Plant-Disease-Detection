# Plants-Disease-Detection-using-Tensorflow-and-OpenCV
Implemented Machine Learning and Artificial Intelligence model to detect the different types of diseases seen on plant leaves using the images.

# Technology Stack
- Python
- Convolutional Neural Networks (CNN)
- Machine Learning
- Flask

# VS Code Execution - Flask Application
 -Create a virtual environment
 - 'virtualenv -p python3.10 env'
 
 - note: Only python version 3.10 and below support tensorflow
 
 - Activate the virtualenv
 - '.\env\Scripts\activate'
 
 -If activated then install the required dependencies
 
 -To install tensorflow run command
 - 'pip install tensorflow'
 
 -To install cv2 run command
 - 'pip install opencv-python'
 
 - Now open the directory which contains the flask application python file(In our case it is the Deployment directory
 - Run command
 - 'cd 'Deployment' '
 
 -Now to run the flask app(app.py) run command
 - 'python app.py'
 
 -Click on the address of the local server where the application is hosted (https://127.0.0.1:5000) which redirects you to the server or copy the link and run it in any browser
 
# Fetching Data from Kaggle on Google Colab
 - Open the Kaggle fetch python notebook on Google Colab
 - Mount your Google Drive on colab
 - Give access to Colab when prompted to mount your google drive
 - Download dataset from Kaggle's Plant Village directory
 - Unzip the file onto your google drive
 
# Building,Training and testing your ML model
 - Open plant-disease-detection python notebook on Google Colab
 - Mount your google drive
 - Build your model
 - Perform train-test split of 80%-20%
 - Train your model
 - Test the accuracy of the model
 - Dump the model in the project folder using pickle
 - Train your model
 
