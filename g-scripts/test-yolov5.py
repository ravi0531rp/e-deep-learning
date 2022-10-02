import torch
import cv2
import numpy as np

class Detector(object):
    def __init__(self , model , nmsConf = 0.25 , iou = 0.45 , max_det = 20 , device='cpu' , onlyHumans = True, verbose = True):
        self.model = model
        self.model.nmsConf = nmsConf
        self.model.iou = iou
        self.model.max_det = max_det

        self.detections = []
        self.detectedImage = None
        self.onlyHumans = onlyHumans
        self.verbose = verbose
        self.device = device
        if device == 'cpu':
            self.model.cpu()
        else:
            if torch.cuda.is_available():
                print("Cuda is available!")
                self.model.cuda()
                self.device = 'gpu'
            else:
                self.device = 'cpu'

    def detect(self,frame):
        self.detectedImage = frame
        if self.device == 'cpu':
            detections = self.model(frame).xyxy[0].numpy()
        else:
            detections = self.model(frame).xyxy[0].cpu().numpy()
        if self.onlyHumans:
            detections = detections[detections[:,5]==0]
        self.detections = detections
        if self.verbose:
            print(self.detections)
    
    def drawBoundingBoxes(self):
        for box in self.detections:
            self.detectedImage = cv2.rectangle(self.detectedImage , (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])) , 
            (0,0,255) , 1)
        
    def showDetectorOutput(self):
        cv2.imshow("frame" , detector.detectedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    detector = Detector(model = model , device = 'gpu')
    image = cv2.imread("images/car.jpg")
    detector.detect(image)
    detector.drawBoundingBoxes()
    detector.showDetectorOutput()
    print("Done processing ....")