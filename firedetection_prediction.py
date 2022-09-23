"""
Inference script for Fire Detection TFLite model

Author: Venkat Rebba, rebba498@gmail.com
"""

from statistics import mode
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter 
import cv2
# import tensorflow as tf
import time
import argparse


class FireDetectionModel:
    def __init__(self, modelPath):
        self.modelPAth = modelPath
        self.IMG_SIZE = 256 

        self.interpreter = Interpreter(model_path=modelPath)
        # self.interpreter = tf.lite.Interpreter(model_path=modelPath)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.classes = ['Not Fire', 'Fire']

    def segmentation(self, arr):
            for i in range(self.IMG_SIZE):
                for j in range(self.IMG_SIZE):
                    r=arr[i,j,0]
                    g=arr[i,j,1]
                    b=arr[i,j,2]
                    if(r>g and g>b and r>200):
                        pass
                    else:
                        arr[i,j]=[0,0,0]
            return arr
        
    def preProcessImg(self, imgPath):
        img = Image.open(imgPath)
        img = img.resize((self.IMG_SIZE, self.IMG_SIZE))
        img = img.convert('RGB')
        img=  np.array(img, dtype=np.float32)
        img = self.segmentation(img)
        img = np.expand_dims(img, axis=0)
        return img


    def preProcessImg2(self, imgPath):
        img1 = cv2.imread(imgPath)
        img1 = cv2.resize(img1, (self.IMG_SIZE, self.IMG_SIZE)) 
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = np.array(img1, dtype=np.float32)
        img1 = self.segmentation(img1)
        img1 = np.expand_dims(img1, axis=0)
        return img1


    def predictTestRepo(self, repo):
        images = [os.path.join(repo, fi) for fi in os.listdir(repo)]
        labels = []
        for imgPath in images:
            img = self.preProcessImg(imgPath=imgPath)
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            result = output_data[0]
            answer = np.argmax(result)
            label = self.classes[answer]
            labels.append(label)
        
        return labels


    def liveCameraPredictions(self, cameraId, timeout):

        # cameraId= "Screen Recording 2022-04-08 at 2.38.43 PM.mov"
        start_time = time.time()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
        out = cv2.VideoWriter(f'predictions_{time.time()}.avi', fourcc, 1, (self.IMG_SIZE,self.IMG_SIZE))
        cap = cv2.VideoCapture(cameraId);
        labels = []
        while cap.isOpened():
            if time.time() - start_time >= timeout:
                print("Timeout")
                break

            ret, frame = cap.read()
            if not ret:
                print("Error reading frame..Exiting")
                break

            img = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE)) 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_fi = np.array(img_rgb, dtype=np.float32)
            img_se = self.segmentation(img_fi)
            img_exp = np.expand_dims(img_se, axis=0)

            st = time.time()
            label = self.predict_model(img_exp)

            print(label)
            print(f"prediction time: {time.time() - st}")
            labels.append(label)

            cv2.putText(img, label, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            out.write(img)

        #cap.release()
        #out.release()
        return labels


    def predictVideo(self, videoName):

        # videoName = f'record_{time.time()}.avi'
        print("video", videoName)
        predVideo = videoName + "_pred.avi"
        print("prediction video", predVideo)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        predOut = cv2.VideoWriter(predVideo, fourcc, 1, (self.IMG_SIZE,self.IMG_SIZE))

        cap = cv2.VideoCapture(videoName, cv2.CAP_FFMPEG)
        labels = []

        print("video reading ", cap.isOpened())

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                print("Error reading frame..Exiting")
                break

            frame = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_fi = np.array(img_rgb, dtype=np.float32)
            img_se = self.segmentation(img_fi)
            img_exp = np.expand_dims(img_se, axis=0)

            st = time.time()
            label = self.predict_model(img_exp)

            print(label)
            print(f"prediction time: {time.time() - st}")
            labels.append(label)

            cv2.putText(frame, label, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            predOut.write(frame)

        cap.release()
        predOut.release()

        return labels


    def recordAndPredict(self, cameraId, timeout):

        start_time = time.time()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoName = f'record_{time.time()}.avi'
        print("reoced video", videoName)
        out = cv2.VideoWriter(videoName, fourcc, 1, (self.IMG_SIZE,self.IMG_SIZE))
        cap = cv2.VideoCapture(cameraId);
        labels = []
        while cap.isOpened():
            
            if time.time() - start_time >= timeout:
                print("Timeout")
                break

            ret, frame = cap.read()
            if not ret:
                print("Error reading frame..Exiting")
                break

            img = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE))
            out.write(img)

        time.sleep(30)
        cap.release()
        out.release()

        return self.predictVideo(videoName)
        
    def predict_model(self, img):
        # Load TFLite model and allocate tensors.
        # img = self.preProcessImg(imgPath=imgPath)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        result = output_data[0]
        answer = np.argmax(result)
        label = self.classes[answer]
        
        return label


if __name__ == "__main__":


    # Create the parser
    my_parser = argparse.ArgumentParser(description='Fire detection params')

    # Add the arguments
    my_parser.add_argument('-t', '--timeout',
                        type=int,
                        default=15,
                        help='Timeout for capturing')

    my_parser.add_argument('-c', '--camera',
                        type=int,
                        default=0,
                        help='Camera number')

    my_parser.add_argument('-m', '--model_path',
                        type=str,
                        default="fireClassification.tflite",
                        help='Model path to load')

    my_parser.add_argument('-v', '--video_path',
                        type=str,
                        help='Video path to load')


    # Execute the parse_args() method
    args = my_parser.parse_args()
    print("args", args)

    timeout = args.timeout
    modelPath = args.model_path
    cameraid = args.camera
    videoPath = args.video_path

    model = FireDetectionModel(modelPath=modelPath)
    labels = model.recordAndPredict(cameraid, timeout)

    firePer = 0
    if 'Fire' in labels:
        firePer = labels.count('Fire')*100/len(labels)

    print(f"Fire frames {firePer}%, out of total frames: {len(labels)}")
