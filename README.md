# Car License Plate Recognition (YOLOv3 + Tesseract)

Detects car license plates in video sequences with a custom-trained YOLOv3 model, reads the plate text with Tesseract OCR, and evaluates the recognition against the ground truth taken from each video's file name.

## How it works

For each video in the input folder:

1. **Detection:** every frame is passed through a YOLOv3 network (416×416 input, one class: `license_plate`) via OpenCV's DNN module. Boxes below the confidence threshold (0.5) are dropped and overlapping boxes are merged with non-maximum suppression (0.4).
2. **OCR:** each detected plate is cropped, converted to grayscale, binarized (threshold 95) and read by Tesseract (`lang='deu'`, `--psm 7` = single text line). Everything except `A-Z` and `0-9` is removed.
3. **Flattening over frames:** the `car_license_plate` class counts how often each OCR reading occurs across the sequence. The most frequent reading becomes the predicted label. The first frame where the prediction matches the truth is recorded.
4. **Output:** an annotated video (bounding box plus predicted plate text) is written per sequence, and every plate crop is saved as a PNG.

After all sequences are processed, the script prints:

- the first frame with a correct prediction for each correctly recognized plate
- the Damerau-Levenshtein distance between the truth and the prediction per sequence, plus the average
- predictions that match once the first two characters are stripped (to handle a misread country code on the plate)
- the total frame length of correctly recognized sequences

## Files

| File | Purpose |
|---|---|
| `clp_recognition_sequences.py` | Main script: detection, OCR and evaluation |
| `yolov3-c1.cfg` | YOLOv3 network configuration (1 class) |
| `clp.names` | Class names (`license_plate`) |
| `yolov3-c1_50024.weights` | Trained weights after 50,024 iterations (~246 MB, **not tracked in git**, see below) |

## Requirements

- Python 3
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) with the German language pack (`deu`)
- Python packages:

```
pip install opencv-python numpy pillow pytesseract jellyfish
```

## Setup

### Model weights

The weights file is excluded through `.gitignore` because it's too large for a normal git repository (GitHub rejects files over 100 MB). Get `yolov3-c1_50024.weights` separately and put it next to the config file.

### Paths

All paths in `clp_recognition_sequences.py` are hard-coded for the original Linux machine. Change them before running:

| Line | Variable | Original value |
|---|---|---|
| 20 | `classesFile` | `/home/yolo/darknet/data/clp.names` |
| 26 | `modelConfiguration` | `/home/yolo/darknet/cfg/yolov3-c1.cfg` |
| 27 | `modelWeights` | `/home/yolo/darknet/yolov3-c1_50024.weights` |
| 126 | plate crop output | `/home/yolo/darknet/python_res/` |
| 177 | input video folder | `/home/yolo/Videos/final2` |

The script also expects these folders to exist:

- `<video folder>/_predictions/`, where the annotated videos are written
- the plate crop folder (`python_res/`)

### Input videos

Name each video after the plate it shows, since the file name (without extension) is the ground truth. For example, `HHAB1234.mp4` is evaluated against `HHAB1234`.

## Usage

```
python clp_recognition_sequences.py
```

## Known issues

- **Newer OpenCV versions:** `i[0]` in the NMS loop and in `getOutputsNames` fails on OpenCV 4.5.4 and later, because those versions return flat arrays. Either pin an older `opencv-python` or remove the `[0]` indexing.
- `measure_performance()` calls `getLabelCount()`, which doesn't exist. The function is never called, so the script still runs.
- The DNN target is set to OpenCL (`DNN_TARGET_OPENCL`). Without an OpenCL device, OpenCV falls back to the CPU.

## Credits

The detection pipeline is based on the YOLOv3 OpenCV tutorial from [learnopencv.com](https://learnopencv.com). The OCR, flattening and evaluation parts were written by Hauke Hoppe.
