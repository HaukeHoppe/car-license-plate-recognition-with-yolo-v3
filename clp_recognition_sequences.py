# ############################## learnopencv.com ##############################
import cv2 as cv
import numpy as np
import os
from os import walk
import sys
from PIL import Image, ImageFilter
import pytesseract
import re
import operator
import jellyfish

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes
classesFile = "/home/yolo/darknet/data/clp.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the
# network using them.
modelConfiguration = "/home/yolo/darknet/cfg/yolov3-c1.cfg"
modelWeights = "/home/yolo/darknet/yolov3-c1_50024.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)  # Run on GPU

outputFile = "/home/yolo/darknet/yolo_out_py.avi"


class args:
    video = "/home/yolo/darknet/VID_20181024_110614.mp4"
    image = False

# ---------------------------------Hauke Hoppe---------------------------------


class car_license_plate:

    def __init__(self, license_plate_truth, frame_length):
        self.license_plate_truth = license_plate_truth
        self.clp_prediction = {'': 0}
        self.label = ""
        self.frameCount = 0
        self.firstFrame = ""
        self.locked = False
        self.frame_length = frame_length

    def flatten(self, license_plate, frame_nr):
        clp_count = self.clp_prediction.get(license_plate, False)
        if not clp_count:
            self.clp_prediction[license_plate] = 1
        else:
            new_count = clp_count + 1
            self.clp_prediction[license_plate] = new_count
        b_label = max(
            self.clp_prediction.items(),
            key=operator.itemgetter(1))[0]
        self.label = b_label
        if self.license_plate_truth == self.label and not self.locked:
            self.firstFrame = frame_nr
            self.locked = True  # save first frame_nr of correct detection
            print(
                "locked! T:" +
                self.license_plate_truth +
                " L:" +
                self.label +
                " frn:" +
                frame_nr)
        self.frameCount = self.frameCount + 1
        return b_label

    # return flatten_sorted_keys
    def getData(self):
        flatten_sorted_keys = sorted(
            self.clp_prediction,
            key=self.clp_prediction.get,
            reverse=True)
        # string = ""
        print("Truth: " + self.license_plate_truth)
        for r in flatten_sorted_keys:
            print("(" + r + ")/", self.clp_prediction[r])

    def getMaxLabel(self):
        return str(max(self.clp_prediction.items(),
                   key=operator.itemgetter(1))[0])

    def accuracy(self):
        maxLabel = self.getMaxLabel()
        x = self.clp_prediction[maxLabel] / self.frameCount
        return str(x)


flatten_arr = []
# -----------------------------------------------------------------------------


def postprocess(frame, outs, current, log_information):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep
    # only the ones with high confidence scores.
    # Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                right = int(center_x + width / 2)
                bottom = int(center_y + height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                # -------------------------Hauke Hoppe-------------------------
                cv2_im = cv.cvtColor(frame, cv.COLOR_RGBA2GRAY)
                cv2_im = cv2_im[top:bottom, left:right]
                ret, cv2_im = cv.threshold(cv2_im, 95, 255, cv.THRESH_BINARY)

                label2 = ""
                if cv2_im is not None:
                    pil_im = Image.fromarray(cv2_im)
                    target = pytesseract.image_to_string(
                        pil_im, lang='deu', config='--psm 7')
                    label2 = re.sub("[^A-Z0-9]+", ' ', target)
                    label2 = label2.replace(" ", "")
                    fl_label = ""
                    if len(label2) != 0:
                        fl_label = flatten_arr[-1].flatten(label2, current)
                    print(
                        log_information +
                        " Label:" +
                        label2 +
                        " Flatten: " +
                        str(fl_label))
                    pil_im.save(
                        "/home/yolo/darknet/python_res/" +
                        label2 +
                        "_" +
                        current +
                        ".png")
                    label2 = flatten_arr[-1].getMaxLabel()
                # -------------------------------------------------------------

    # Perform non maximum suppression to eliminate redundant overlapping
    # boxes with lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(
            classIds[i],
            confidences[i],
            left,
            top,
            left + width,
            top + height,
            label2)

# Draw the predicted bounding box


def drawPred(classId, conf, left, top, right, bottom, label2):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(
        label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.putText(frame, label2, (left, top),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

# Get the names of the output layers


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# ---------------------------------Hauke Hoppe---------------------------------


def getSequences(mypath):
    file_names = []
    license_plates_truth = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        file_names.extend(filenames)
        break
    myarray = np.asarray(filenames)
    for i in myarray:
        license_plates_truth.append(i.split(".")[0])
    return file_names, license_plates_truth, mypath


filenames, license_plate_truth, mypath = getSequences(
    "/home/yolo/Videos/final2")
for index, sequence in enumerate(filenames, start=1):
    args.video = mypath + "/" + sequence
    if (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        # cap.set(cv.CAP_PROP_POS_FRAMES,99)
        outputFile = mypath + '/_predictions/' + sequence + '_prediction.avi'
    else:
        # Webcam input
        cap = cv.VideoCapture(0)

    flatten_arr.append(car_license_plate(
        license_plate_truth[index - 1],
        str(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))))
# -----------------------------------------------------------------------------
    # Get the video writer initialized to save the output video
    if (not args.image):
        vid_writer = cv.VideoWriter(
            outputFile, cv.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), 30, (round(
                    cap.get(
                        cv.CAP_PROP_FRAME_WIDTH)), round(
                    cap.get(
                        cv.CAP_PROP_FRAME_HEIGHT))))

    while cv.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()
        # -----------------------------Hauke Hoppe-----------------------------
        length = str(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
        current = str(int(cap.get(cv.CAP_PROP_POS_FRAMES)))
        log_information = "File: (" + str(index) + "/" + \
            str(len(filenames)) + ") Frame: " + current + "/" + length
        # ---------------------------------------------------------------------

        # Stop the program if reached end of video
        if current == length:
            # if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(
            frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs, current, log_information)

        # Put efficiency information. The function getPerfProfile returns the
        # overall time for inference(t) and the timings for each of the
        # layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (
            t * 1000.0 / cv.getTickFrequency())
        # cv.putText(frame, label, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
        #            (0, 255, 0), 3)

        # Write the frame with the detection boxes
        if (args.image):
            cv.imwrite(outputFile, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))
###############################################################################
# ---------------------------------Hauke Hoppe---------------------------------
# get flatten list of element in array


def measure_performance(flatten_arr):
    for clp in flatten_arr:
        clp.getLabelCount(clp.getMaxLabel())


# get first frame of right predicted label
right_prediction = 0
avg_right_frame_nr = 0
for clp in flatten_arr:
    if clp.label == clp.license_plate_truth:
        right_prediction = right_prediction + 1
        print(clp.firstFrame)

# calculate average string distance
avg_distance = 0
for clp in flatten_arr:
    distance = jellyfish.damerau_levenshtein_distance(
        clp.license_plate_truth, clp.label)
    print(
        "truth:" +
        clp.license_plate_truth +
        " Predicted:" +
        clp.label +
        " Distance:" +
        str(distance))
    avg_distance = avg_distance + distance
print("Average distance: " + str((avg_distance) / len(flatten_arr)))

# remove misspelling at the beginng due to country code in license plate
for clp in flatten_arr:
    label = clp.label
    target = label[2:len(label)]
    if target == clp.license_plate_truth:
        print(target)

# get total framelength of sequence
for clp in flatten_arr:
    label = clp.label
    target = label[0:len(label)]
    if target == clp.license_plate_truth:
        print(clp.frame_length)
# -----------------------------------------------------------------------------
