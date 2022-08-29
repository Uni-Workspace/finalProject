import cv2
import socket 
import torch
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.plots import plot_one_box
from utils.general import scale_coords, xyxy2xywh, non_max_suppression

UDP_IP, UDP_PORT = "127.0.0.1", 8888
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

weights, device ='face_detection.pt', "cpu"
model = attempt_load(weights, map_location=device)

source, imgsz, stride = "0", 416, 64
dataset = LoadStreams(source, img_size=imgsz, stride=stride)

for path, img, im0s, vid_cap in dataset:

    img = torch.from_numpy(img.copy()).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)


    for i, det in enumerate(pred):  # detections per image
            
        im0 = im0s[i].copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0  # for save_crop
        
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        output_dict = {}
        for *xyxy, conf, cls in reversed(det):

            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()                    
            output = (cls, *xywh, conf) 
            data = xywh.copy()
            data.append(conf)
            output_dict[output[1]] = data                                                               

            im0 = plot_one_box(xyxy, im0, label="test")
                                                                    

        output_dict = dict(sorted(output_dict.items()))         
        output_list = [output_dict[d] for d in output_dict.keys()]
        print(output_list)
        
        x = str(xywh[0])[2:4]
        y = str(xywh[1])[2:4]
        MESSAGE = x + y
        MESSAGE = MESSAGE.encode()
        sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

        cv2.imshow("test", im0)
    if cv2.waitKey(1) == ord('q'):
        break 


