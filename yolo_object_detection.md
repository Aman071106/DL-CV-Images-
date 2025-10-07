## Darknet / YOLO – Canva-Style Guide
(Can change backbone from darknet)
---

### 1️⃣ Installation

**Commands:**

* `make clean` → Remove previous builds
* `make` → Compile C/CUDA code into **darknet** binary

**Key Components:**

| Component | Role                     |
| --------- | ------------------------ |
| Makefile  | Compilation instructions |
| gcc, nvcc | Compilers                |
| darknet   | Final binary             |

**Makefile Tweaks:**

* GPU=1
* CUDNN=1
* OPENCV=1

---

### 2️⃣ Download Pre-trained Weights

```bash
wget https://pjreddie.com/media/files/imagenet.weights
```

**Notes:**

* YOLO weights are essential for detection.
* Pre-trained weights save training time.

---

### 3️⃣ Concepts

| Task                 | What it Does                      | Example Output                 |
| -------------------- | --------------------------------- | ------------------------------ |
| Image Classification | Classify entire image             | “Cat”                          |
| Object Detection     | Identify objects + bounding boxes | Cat at (x=50,y=30,w=100,h=120) |
| Image Segmentation   | Pixel-level classification        | Mask highlighting cat pixels   |

**Traditional Networks:**

* R-CNN, Fast R-CNN, Faster R-CNN → Look twice at image (region proposals + classification)

**YOLO / Detectron2:**

* Look **once** → Predict bounding boxes + class in a single pass
* Uses **IoU loss**, NMS filtering, and thresholding

---

### 4️⃣ YOLO Pipeline

1. **Preprocessing** → Resize image, normalize
2. **CNN Backbone** → Extract feature maps
3. **Bounding Box Prediction** → Predict tx, ty, tw, th for each anchor box
4. **Filter Boxes** → Apply confidence threshold
5. **Non-Max Suppression (NMS)** → Remove overlapping boxes

---

### 5️⃣ Predict Command

**Image Classification:**

```bash
./darknet classifier predict <data_file> <cfg_file> <weights_file> <input_image>
```
```bash
 ./darknet detector test cfg/coco.data cfg/yolov3.cfg  yolov3.weights data/dog.jpg
```

* Example:

```bash
./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg darknet19.weights data/dog.jpg
```

* Output: Top class predictions with probabilities.

**Object Detection:**

```bash
./darknet detector test <data_file> <cfg_file> <weights_file> <input_image>
```

* Example:

```bash
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
```

* Output: Image with bounding boxes and class labels.
* Options: `-thresh 0.25` (confidence threshold), `-ext_output` (detailed output in console)

---

### 6️⃣ Example (Step-by-Step Calculations)

**Scenario:** Image with 2 objects (dog, cat)

* Image divided into 3x3 grid (S=3)
* 2 anchors per cell (B=2)
* Classes: 2 (dog, cat)

**Predictions (tx, ty, tw, th, confidence, class scores):**

| Cell  | Anchor | tx  | ty  | tw  | th  | Conf | Class Probabilities |
| ----- | ------ | --- | --- | --- | --- | ---- | ------------------- |
| (1,1) | 1      | 0.5 | 0.5 | 0.8 | 0.6 | 0.9  | [0.1,0.9]           |
| (1,1) | 2      | 0.4 | 0.6 | 0.7 | 0.5 | 0.3  | [0.8,0.2]           |
| (2,2) | 1      | 0.3 | 0.7 | 0.6 | 0.8 | 0.8  | [0.9,0.1]           |

**Step 1: Convert tx, ty, tw, th to box coordinates**

* bx = (sigmoid(tx) + cx)/S, by = (sigmoid(ty)+cy)/S
* bw = pw * exp(tw), bh = ph * exp(th)

**Step 2: Compute final box confidence**

* Box_confidence = Conf * class_prob

**Step 3: Apply threshold**

* Keep boxes with confidence > 0.5

**Step 4: Apply NMS**
example
![Non max Suppression](https://github.com/Aman071106/DL-CV-Images-/blob/main/nms.png)


* Remove overlapping boxes with IoU > 0.5

**Result:**

* Box 1: Cat, confidence 0.81, coordinates (x1,y1,x2,y2)
* Box 2: Dog, confidence 0.72, coordinates (x1,y1,x2,y2)
