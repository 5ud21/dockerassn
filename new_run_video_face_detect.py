import argparse
import sys
import cv2

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(description='detect_video')
parser.add_argument('--net_type', default="RFB", type=str, help='The network architecture, optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=480, type=int, help='define network input size, default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.7, type=float, help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int, help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str, help='imgs dir')
parser.add_argument('--test_device', default="cuda:0", type=str, help='cuda:0 or cpu')
parser.add_argument('--video_path', default="iiserb_vid\Coldplay.mp4", type=str, help='path of video')
parser.add_argument('--output_path', default="/workspace/home-start/output_video.mp4", type=str, help='path of output video')
args = parser.parse_args()

input_img_size = args.input_size
define_img_size(input_img_size)

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "./models/voc-model-labels.txt"

net_type = args.net_type

cap = cv2.VideoCapture(args.video_path)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = args.test_device

candidate_size = args.candidate_size
threshold = args.threshold

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

timer = Timer()
sum = 0
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    interval = timer.end()
    print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f" {probs[i]:.2f}"
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)

    # Write the frame to the output video
    out.write(orig_image)

    sum += boxes.size(0)

cap.release()
out.release()
print("all face num:{}".format(sum))
