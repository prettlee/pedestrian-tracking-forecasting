# pedestrian-tracking-forecasting
Detect and track pedestrians in video, with a study of how detector size, input resolution, and tracker design affect results. Trajectory forecasting with uncertainty estimates is in progress.

Built on the public MOT17 pedestrian tracking benchmark using PyTorch, Ultralytics YOLOv8, and a tracker I implemented from scratch (Kalman filter + Hungarian matching).

Status
Stage	Status
Detection baseline and benchmark (model size vs. input resolution)	Done
SORT-style tracker with parameter study	Done
ByteTrack-style two-stage tracker, tested on 3 sequences	Done
Trajectory forecasting (ETH/UCY, constant-velocity baseline, then a learned model)	In progress
Uncertainty estimates for forecasts	Planned
ONNX export and edge-style benchmarking	Planned
Pipeline
Detect. A pretrained YOLOv8 model (COCO weights, person class only) runs on every frame. Detections are saved with a low confidence cutoff (0.1) so the tracker can use low-score boxes.
Track. A constant-velocity Kalman filter predicts each person's box, and the Hungarian algorithm matches predictions to detections by IoU.
Compare. A plain SORT-style tracker (SortTracker) is compared against a ByteTrack-style tracker (ByteTracker) that makes a second matching pass using low-confidence detections.
Evaluate. Detection: precision and recall at IoU 0.5. Tracking: MOTA, IDF1, ID switches, and fragmentations via motmetrics.

ByteTracker is my own simplified re-implementation of the two-stage idea from the ByteTrack paper. It is not the official implementation and should not be expected to match its published numbers.

Evaluation setup
Detection: MOT17-02, every 10th frame of the first 600 frames (60 frames), IoU ≥ 0.5, confidence threshold 0.25. Ground truth is limited to pedestrians with visibility ≥ 0.3 unless stated otherwise.
Tracking: full sequences, detections from yolov8n at 1280 px input with confidence ≥ 0.1 saved. Scoring uses ground-truth pedestrians with visibility ≥ 0.3 and IoU 0.5 matching.
Data: the FRCNN-detector variant of each sequence (images and ground truth are identical across the DPM, FRCNN, and SDP variants, only the provided detections differ, and I use my own detector instead).
Results
1. Detection: how visibility affects recall (yolov8n @ 640, MOT17-02)
Min. visibility in ground truth	Precision	Recall
0.0	0.73	0.23
0.3	0.69	0.45
0.6	0.62	0.56

Recall rises as heavily occluded people are removed from the ground truth, so much of the low recall at 0.0 reflects people the detector could not reasonably see. Precision falls slightly because unlabeled or partly visible people the detector finds count as false positives.

2. Detection: model size vs. input resolution (MOT17-02, min_vis = 0.3)

NVIDIA T4 GPU. Time per frame includes image loading and decoding, and the first row includes GPU warm-up.

Model	Input size	Precision	Recall	ms/frame
yolov8n	640	0.69	0.45	34
yolov8n	1280	0.65	0.66	29
yolov8s	640	0.72	0.57	29
yolov8s	1280	0.62	0.69	49
yolov8m	640	0.70	0.62	37
yolov8m	1280	0.65	0.72	94

CPU (earlier run, same setup):

Model	Input size	Precision	Recall	ms/frame
yolov8n	640	0.69	0.45	153
yolov8n	1280	0.65	0.66	504
yolov8s	640	0.72	0.57	459
yolov8s	1280	0.62	0.69	1473
yolov8m	640	0.70	0.62	947

Takeaway: input resolution mattered more than model size on this 1920×1080 footage, which contains many small, distant pedestrians. yolov8n at 1280 reached recall 0.66 at about 29 ms/frame, beating yolov8m at 640 (recall 0.62). I used yolov8n at 1280 as the default detector for tracking.

3. Tracking: SORT-style parameter study (MOT17-02)

Defaults: det_thr=0.3, max_age=15, iou_thr=0.3, min_hits=3. One parameter changed per row. There are 62 people in the ground truth.

Setting changed	Tracks	MOTA	IDF1	Precision	Recall	ID switches	Fragmentations
baseline	204	0.401	0.505	0.751	0.616	93	202
det_thr=0.15	450	0.123	0.406	0.557	0.718	208	237
det_thr=0.5	95	0.386	0.505	0.867	0.461	42	147
max_age=30	172	0.400	0.519	0.747	0.619	87	206
max_age=60	155	0.396	0.510	0.743	0.620	85	205
iou_thr=0.2	193	0.397	0.502	0.747	0.616	93	198
iou_thr=0.5	229	0.405	0.476	0.757	0.612	99	197
min_hits=1	282	0.368	0.494	0.717	0.628	110	202

Lowering the detection threshold to 0.15 raises recall but creates many false tracks, so MOTA collapses. This motivates a two-stage design that uses low-confidence boxes only to extend existing tracks. Fragmentations stay near 200 across most settings, which suggests the parameters are not the main cause.

4. Tracking: ByteTrack-style variants (MOT17-02)
Method	Tracks	MOTA	IDF1	Precision	Recall	ID switches	Fragmentations
SORT (max_age=30)	172	0.400	0.519	0.747	0.619	87	206
ByteTrack-style, high=0.5	92	0.362	0.537	0.704	0.633	40	73
ByteTrack-style, high=0.6	68	0.313	0.513	0.721	0.516	25	30
ByteTrack-style, high=0.4	126	0.277	0.505	0.631	0.683	66	112
ByteTrack-style, high=0.5, low_iou=0.5	94	0.376	0.538	0.722	0.620	43	58
5. Tracking: SORT vs. ByteTrack-style on three sequences

I chose the ByteTrack-style settings (high_thr=0.5, low_iou=0.5, max_age=30) using MOT17-02, then ran them unchanged on MOT17-09 and MOT17-04.

Sequence	Method	Tracks (true people)	MOTA	IDF1	ID switches	Fragmentations
MOT17-02 (tuned on)	SORT	172 (62)	0.400	0.519	87	206
MOT17-02 (tuned on)	ByteTrack-style	94	0.376	0.538	43	58
MOT17-09	SORT	113 (26)	0.457	0.530	57	28
MOT17-09	ByteTrack-style	85	0.430	0.541	36	20
MOT17-04	SORT	251 (83)	0.593	0.584	180	703
MOT17-04	ByteTrack-style	111	0.619	0.660	61	291
Key findings
The ByteTrack-style second pass reduced ID switches and fragmentations on all three sequences (switches by roughly 40 to 66%, fragmentations by roughly 29 to 72%), and track counts moved closer to the true number of people.
IDF1 improved on all three, but only clearly on MOT17-04 (0.584 to 0.660). The gains on MOT17-02 and MOT17-09 (about 0.01 to 0.02) are within what I would expect from noise on single sequences.
MOTA did not consistently improve. It fell on MOT17-02 and MOT17-09 and rose on MOT17-04. I have not investigated why, and I have not tested my guess that low-confidence matches keep some tracks alive on background clutter.
Detection quality limits MOTA more than tracking does. Tracker recall stays close to detector recall, so most remaining errors come from missed detections, especially of occluded people.
Limitations
Only three of the seven MOT17 training sequences were evaluated (02, 04, 09), each run once, with one detector configuration and no error bars.
Tracker parameters and the detector configuration were selected using MOT17-02, so that sequence's numbers are optimistic.
Detection precision is pessimistic: unlabeled people the detector finds are counted as false positives.
Detection timings include image loading and were measured on a shared cloud GPU, so they are rough. They do not represent edge-device performance.
No appearance features are used for matching, so people who cross paths can swap identities.
The detector is a pretrained COCO model and was not fine-tuned on MOT17.
Reproducing

I run everything in Google Colab with a T4 GPU.

bash
pip install ultralytics opencv-python pandas scipy matplotlib motmetrics

motmetrics uses a NumPy function removed in NumPy 2.0. If you hit an np.asfarray error, add this before scoring:

python
import numpy as np
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

Data. The dataset is not included in this repo. Download MOT17 from https://motchallenge.net (about 5.5 GB; check the dataset's license and terms), and extract the train split so the folders look like MOT17/train/MOT17-02-FRCNN/img1/ and .../gt/gt.txt.

Run. Open notebooks/01_detection_tracking.ipynb and run the cells in order. Do not use "Run all" in a fresh session, since the download and benchmark cells take several minutes.

Repo structure
notebooks/    Colab notebook (detection, tracking, evaluation)
results/      Small result files (CSV tables, saved detections)
README.md
requirements.txt
.gitignore    (excludes data/, *.zip, *.tar)
Roadmap
Trajectory forecasting. Predict 12 future steps from 8 observed steps on the ETH/UCY pedestrian datasets. Constant-velocity baseline first, then a small LSTM or Transformer, reported with ADE and FDE.
Uncertainty. Predict a distribution (Gaussian output or sampled trajectories) and test whether uncertainty is higher in crowded scenes or at sudden turns.
Efficiency. Export the detector to ONNX and measure latency and accuracy trade-offs under tighter compute budgets.
More sequences and, if time allows, a proper held-out evaluation of all tracker settings.
Acknowledgments
MOT17 benchmark (Milan et al.).
Ultralytics YOLOv8 for the pretrained detectors.
motmetrics for MOTA and IDF1.
SORT (Bewley et al.) and ByteTrack (Zhang et al.) for the tracking designs this project builds on.
