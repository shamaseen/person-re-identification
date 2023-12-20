import torch
import numpy as np
import cv2
import random

class preprocessing_results:
    def __init__(self, name:str) -> None:
        self.model_name=name
    def combine_results(self):
        if "pose" in self.model_name:
            bboxes_xyxy = torch.from_numpy(abs(self.detect.prediction.bboxes_xyxy)).tolist()
            confidence = torch.from_numpy(self.detect.prediction.scores).tolist()
            labels = [0]*len(confidence)
            # Combine the bounding box coordinates and confidence scores into a single list
            concate = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
            # Combine the concatenated list with the class labels into a final prediction list
            self.final_prediction = [sublist + [element] for sublist, element in zip(concate, labels)]
            # get pose
            self.pose=torch.from_numpy(self.detect.prediction.poses).numpy()[:,:,2]
        else:
            bboxes_xyxy = torch.from_numpy(abs(self.detect.prediction.bboxes_xyxy)).tolist()
            confidence = torch.from_numpy(self.detect.prediction.confidence).tolist()
            labels = torch.from_numpy(self.detect.prediction.labels).tolist()
            # Combine the bounding box coordinates and confidence scores into a single list
            concate = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
            # Combine the concatenated list with the class labels into a final prediction list
            self.final_prediction = [sublist + [element] for sublist, element in zip(concate, labels)]

    def filter_results(self):
        results = []
        # Loop over the detections
        for count,data in enumerate(self.final_prediction):
            # Extract the confidence (i.e., probability) associated with the detection
            confidence = data[4]
            # Filter out weak detections by ensuring the confidence is greater than the minimum confidence and with the class_id
            if self.classes_ids_fillter== None:
                if float(confidence) < self.conf:
                    continue
            else:
                if ((int(data[5] != self.classes_ids_fillter)) or (float(confidence) < self.conf)):
                    continue
            # filler the pose
            if "pose" in self.model_name:
                # check the  conf of 17 point if 40% of values is less than 0.5 ignone the person
                pred_pose_conf=(np.count_nonzero(self.pose[count] > self.pose_point_conf) / len(self.pose[count]))
                if pred_pose_conf <self.pose_conf:
                    continue
            # If the confidence is greater than the minimum confidence, draw the bounding box on the frame
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            # Add the bounding box (x, y, w, h), confidence, and class ID to the results list
            results.append([xmin, ymin, xmax, ymax, confidence, class_id])
        return results
    def __call__(self,detect,classes_ids_fillter:int,pose_conf:float,conf:float,pose_point_conf:float):
        self.classes_ids_fillter=classes_ids_fillter # defult is None
        self.pose_conf=pose_conf # conf for pose
        self.pose_point_conf=pose_point_conf # conf for each 17 point
        """
        Step 1: Filtering by Confidence: It iterates through all 17 pose points
        and filters those with a confidence score (pose_point_conf) exceeding 0.5.
        In this example, 14 points meet this threshold.

        Step 2: Acceptance Rate Calculation: The number of accepted points (13 in this case)
        is divided by the total number of points (17), resulting in an acceptance rate of 0.764 (76.4%).

        Step 3: Comparison with Reference Confidence: This acceptance rate
        is then compared to a predefined reference confidence value (represented by "pose_conf" in your code).

        """
        self.conf=conf # conf for detected bbox
        self.detect=detect # detection results

        #
        self.combine_results()
        return self.filter_results()

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1]-2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # cv2.putText(img, label, (c1[0], c1[1]+int(c2[1]/2)), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
# if __name__=='__main__':
#     filter_results=preprocessing_results("yolo_nas_pose_l")
#     results=filter_results(detect=detect,pose_conf=0.6,pose_point_conf=0.4,conf=0.3,classes_ids_fillter=0)