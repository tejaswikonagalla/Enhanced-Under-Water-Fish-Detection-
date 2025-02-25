**PROJECT DETAILS :**

<div align="justify">This project focuses on developing an advanced system for underwater fish detection and species recognition using deep learning techniques. Identifying and classifying fish species in underwater environments presents significant challenges due to poor visibility, varying lighting conditions and the presence of particulate matter. To overcome these issues, this project uses efficient neural network architectures and image processing techniques to enhance detection accuracy and performance.The primary objective of this project is to enable real-time and accurate detection of fish species(Fish, jellyfish, penguin, puffin, shark, starfish and sting ray) in underwater environments. The system utilizes convolutional neural networks (CNNs) to extract meaningful features from images and classify different fish species. By leveraging optimized deep learning models and adaptive image enhancement techniques, the project aims to improve the reliability of underwater fish detection, supporting marine research, environmental monitoring, and sustainable fisheries management.To evaluate the effectiveness of different deep learning models, we trained and compared the performance of two distinct architectures: Faster R-CNN with MobileNet and YOLOv9 in the backend. The results of both models were analyzed and we came to a conclusion that Faster RCNN is the best model to be used for underwater fish detection.</div><br/>

**FEATURES :**

<div align="justify">

<b>1)User-Friendly Interface :</b> Designed for ease of use, allowing researchers and conservationists to utilize the system effectively.

<b>2)High Accuracy : </b> Utilizes advanced deep learning models to enhance the accuracy of fish species recognition.

<b>3)Robust to Underwater Conditions :</b> Incorporates image enhancement techniques to mitigate challenges like poor visibility, lighting variations and particulate matter common in underwater environments. 

<b>4)Enhanced Adaptability to Varied Species :</b> The combination of MobileNet and Faster R-CNN improves the system's ability to accurately identify and classify a wide range of fish species. 

<b>5)Better Handling of Occlusions and Overlaps :</b> The system includes strategies to better manage scenarios where fish are overlapping or partially occluded. This capability is crucial in densely populated underwater environments.

<b>6)Scalability and Flexibility :</b> The lightweight nature of the proposed system, powered by MobileNet, offers scalability and flexibility, allowing it to be deployed on a variety of platforms, including underwater vehicles and embedded systems. </div><br/>

**DATASET DETAILS :**

<div align="justify">The dataset is from Roboflow and the dataset used is <b>“Aquarium Combined”</b>.The dataset utilized for this project comprises a diverse collection of underwater images, specifically curated to represent seven distinct marine species: Fish, jellyfish, penguin, puffin, shark, starfish and sting ray.It helps to make the model more versatile during training, thus providing for correct identification of these water creatures in their natural environments.The dataset comprises of 638 images which are further splitted into 448 (70%) train, 127(20%) valid, 63(10%) test images.</div><br/>
The dataset can be accessed by using the link : https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/6 <br/><br/>

<br/>

**ALGORITHMS AND TECHNOLOGIES :**

<div align="justify">
  
<b>1)Faster R-CNN : </b> An advanced object detection framework that consists of a Region Proposal Network (RPN) and a classification network. It enables precise localization and classification of fish species in underwater images.

<b>2)YOLOv9 :</b> A state-of-the-art object detection model that optimizes speed and accuracy by directly predicting bounding boxes and class probabilities. It is particularly useful for real-time detection.

<b>3)MobileNet :</b> A lightweight deep learning architecture designed for efficient feature extraction while maintaining accuracy. It is used as the backbone for the Faster R-CNN model to improve performance and reduce computational complexity.

<b>4)TensorFlow & Keras :</b> Popular deep learning frameworks used to build, train, and deploy neural network models efficiently.

<b>5)Region Proposal Network (RPN) :</b> A key component of Faster R-CNN that suggests candidate regions in images where objects are likely to be present, improving detection efficiency.

<b>6)HTML, CSS, and JavaScript :</b> Used to develop a web-based platform where users can upload fish images, register, and sign in to interact with the detection system.</div><br/>

**PREREQUISITES :**

The installations must be done in <b>Windows PowerShell</b> using the following commands:

1) choco install xampp-72

2) choco install miniconda3

3) choco install vscode

<br/>

**STEPS FOR PROJECT EXECUTION :**

1)Download all the files.

2)Open XAMPP control panel and start Apache and MySQL services.

3)In XAMPP control panel, navigate to Admin Action in MySQL, then to import section and choose db.sql file from FRONT END folder and click IMPORT to connect the database.

4)Open Command Prompt and navigate to FRONT END folder that is downloaded.

5)Upon successful navigation, type "code ." to open all the codes from that folder.

6)The codes will be opened in Visual Studio Code.

7)Now in Visual Studio Code open a new terminal and run the command "python app.py".

8)Now a http link will be generated and opening that link will lead us to UI of the project.

9)Complete the registration and login.

10)Upload any image that is downloaded from internet or train/test/valid images from dataset to get an output that has bounding boxes for fish detecction and species classification with confidence score.

<br/>

**PROJECT WORKFLOW :**

<div align="justify">

<b>1)Image Acquisition :</b> The system captures underwater images containing fish using pre-collected datasets.

<b>2)Preprocessing :</b> The images undergo preprocessing steps such as noise reduction, color correction, and contrast enhancement to improve clarity and visibility.

<b>3)Feature Extraction :</b> A deep learning model (CNN-based) extracts important features from the images, such as shape, texture, and patterns of fish.

<b>4)Region Proposal & Object Detection :</b> The system identifies regions within the image that may contain fish and applies bounding boxes around detected objects.

<b>5)Classification :</b> Using trained models (Faster R-CNN with MobileNet and YOLOv9), the detected objects are classified into different fish species based on learned features.

<b>6)Performance Comparison :</b> The results of both Faster R-CNN with MobileNet and YOLOv9 are compared based on accuracy, speed, and robustness in underwater conditions.

<b>7)Determining the best model for detection :</b> Based on evaluation metrics obtained after training both the models in the backend, Faster RCNN is choosed as the best model for detection.So Faster RCNN is used in the front end part for giving the output.

<b>8)Output & Visualization :</b> The system displays the detected fish species with bounding boxes and confidence scores, helping users to analyze the results.</div>
