from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
import mysql.connector, os
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='fish'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.lower() == "admin@gmail.com":
            if password.lower() == "admin":
                return redirect("/admin_home")
            else:
                return render_template('login.html', message= "Invalid Password!!")
        
        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('home.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('static/upload_images/', fn)
        myfile.save(mypath)

        # Load the Pre-trained Model with Custom Number of Classes
        def load_pretrained_model(num_classes, model_path):
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            # Adjust the classifier for the number of classes
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

            # Load the trained model weights
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            return model

        # Preprocess the Input Image
        def preprocess_image(image_path):
            image = Image.open(image_path).convert("RGB")
            transform = T.Compose([
                T.Resize((600, 600)),
                T.ToTensor(),
            ])
            image = transform(image)
            return image

        # Visualization Function
        def visualize_prediction(image, prediction, threshold=0.8, output_path='static/prediction.png'):
            image = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            ax = plt.gca()

            for box, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
                if score > threshold:
                    x, y, xmax, ymax = box
                    width, height = xmax - x, ymax - y
                    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    label_name = class_names[label]  # Map label ID to class name
                    plt.text(x, y, f'{label_name}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
            plt.savefig(output_path)
            plt.close()

        # Class Names List
        class_names = [
            'background',  # Assuming a background class
            'fish',
            'jellyfish',
            'penguin',
            'puffin',
            'shark',
            'starfish',
            'stingray'
        ]

        # Model Parameters
        num_classes = 7  # Number of classes excluding background
        model_path = "best_model_epoch_18.pth"  # Path to the model weights
        image_path = mypath

        # Load the model
        model = load_pretrained_model(num_classes, model_path)

        # Preprocess the image
        img = preprocess_image(image_path)

        # Perform detection
        with torch.no_grad():
            prediction = model([img])

        # Visualize the predictions and save the plot
        output_path = 'static/prediction.png'
        visualize_prediction(img, prediction[0], output_path=output_path)

        return render_template('upload.html', path=mypath, prediction_image=output_path)
    return render_template('upload.html')



if __name__ == '__main__':
    app.run()