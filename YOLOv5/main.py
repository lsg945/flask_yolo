import pathlib
from hmac import compare_digest
from flask import Flask
from flask_jwt import *
from werkzeug.utils import secure_filename
import hashlib
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth

import torch
import cv2
from PIL import Image, ExifTags

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression

class User(object):
    def __init__(self, id:int, username:str, password:str) -> None:
        self.id = id
        self.username = username
        self.password = password
    def __str__(self) -> str:
        return "{}".format(self.id)

mainFolder = pathlib.Path(__file__).parent.parent.absolute()
securityFolder = os.path.join(mainFolder, "security")
uploadsFolder = os.path.join(mainFolder, "uploads")
bestWeightFile = os.path.join(mainFolder, "YOLOv5", "runs", "train", "yolov5s_results", "weights", "best.pt")

confThres = 0.4
iouThres = 0.45
imgsz = 416
device = "cpu"
model = attempt_load(bestWeightFile, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

allowdExtensions = {"jpg"}

username_table = {}
userid_table = {}

try:
    users = []
    credentials = credentials.Certificate(os.path.join(securityFolder, "firebase_admin.json"))
    firebase_admin.initialize_app(credentials)
    page = auth.list_users()
    index = 0
    while page:
        for user in page.users:
            users.append(
                User(
                    index, 
                    hashlib.sha3_256(user.email.encode("utf-8")).hexdigest(), 
                    hashlib.sha3_256(user.uid.encode("utf-8")).hexdigest()
                )
            )
            index += 1
        page = page.get_next_page()
    username_table = {u.username: u for u in users}
    userid_table  = {u.id: u for u in users}
except Exception as e:
    print(e)

if not os.path.exists(uploadsFolder):
    os.makedirs(uploadsFolder)

class User(object):
    def __init__(self, id: int, username: str, password: str) -> None:
        self.id = id
        self.username = username
        self.password = password

    def __str__(self) -> str:
        return "{}".format(self.id)


def allowed_file(filename: str):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowdExtensions


def remove(filePath: str):
    try:
        if os.path.exists(filePath):
            os.remove(filePath)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def inference(filePath: str):
    try:
        dataset = LoadImages(filePath, img_size=imgsz, stride=stride)
        # Run inference
        for _, img, _, _ in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, confThres, iouThres, classes=None, agnostic=False)

            y1ClassList = list()
            digits = ""

            for x in pred[0]:
                y1ClassList.append((x[0].item(), int(x[5].item())))

            y1ClassList.sort(key=lambda x: x[0])

            for y1Class in y1ClassList:
                digits = "{}{}".format(digits, y1Class[1])

        return {"response": digits}, 200
    except Exception as e:
        print(e)
        return {"response": "Internal Server Error"}, 500
    finally:
        remove(filePath)

def auth(username: str, password: str):
    uHash = hashlib.sha3_256(username.encode("utf-8")).hexdigest()
    pHash = hashlib.sha3_256(password.encode("utf-8")).hexdigest()
    user = username_table.get(uHash, None)
    if user and compare_digest(user.password.encode("utf-8"), pHash.encode("utf-8")):
        return user


def identity(payload):
    userId = payload['identity']
    return userid_table.get(userId, None)

app = Flask(__name__)
app.config["SECRET_KEY"] = "" #RANDOM
jwt = JWT(app, authentication_handler=auth, identity_handler=identity)


@app.route("/yolo", methods=["POST"])
@jwt_required()
def yolo():
    if request.method == "POST":
        if "file" not in request.files:
            return {"response": "No file part"}, 400

        file = request.files["file"]

        if file.filename == "":
            return {"response": "No file selected"}, 400

        if file and allowed_file(file.filename):
            fileName = secure_filename(file.filename)
            fileName = "{}_{}.{}".format(current_identity, datetime.today().strftime(
                '%Y%m%d%H%M%S'), fileName.rsplit('.', 1)[1].lower())
            fileName = secure_filename(fileName)
            filePath = os.path.join(uploadsFolder, fileName)
            if not os.path.exists(filePath):
                file.save(filePath)
                #preprocess(filePath)
                return inference(filePath)
            else:
                return {"response": "File already exists"}, 500
        else:
            return {"response": "File format not allowed"}, 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50002, debug=True, ssl_context=(os.path.join(securityFolder, "cert.pem"), os.path.join(securityFolder, "key.pem")))
