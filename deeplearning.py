# import required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en',rec_algorithm='CRNN')

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

# LOAD YOLO MODEL
INPUT_WIDTH =  640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best2.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

def get_detections(img,net):
    # CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    detections = detections.reshape((-1,6))
    return input_image, detections
    
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

def non_maximum_supression(input_image,detections):
    # FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    # NMS
    # index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    index = np.array(index)
    index = index.flatten()
    
    return boxes_np, confidences_np, index

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
   
    resize = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(resize, (5,5), 0)
    mblur = cv2.medianBlur(blur, 3)

    ret, thresh = cv2.threshold(mblur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    text =ocr.ocr(dilation, cls=False, det=False)
    a=text[0]
    text=list(a)
    return text

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

def drawings(image,boxes_np,confidences_np,index):
    # drawings
    text_list = []
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text = extract_text(image,boxes_np[ind])
        licence =license_text[0]

        
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(0,0,0),-1)

        # confidence text
        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        # license text
        cv2.putText(image,licence,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
        
        text_list.append(license_text)

    return image,  text_list

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

# predictions
def yolo_predictions(img,net):
    ## step-1: detections
    input_image, detections = get_detections(img,net)
    ## step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    ## step-3: Drawings
    result_img, text = drawings(img,boxes_np,confidences_np,index)
    
    return result_img, text

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

# for image detection
def object_detection(path,filename):
    # read image
    image = cv2.imread(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    result_img, text_list = yolo_predictions(image,net)
    #cv2.imwrite('./static/predict/output.jpeg',result_img)
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    text = text_list[0][0]
    # print(text)
    # cv2.namedWindow('results',cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('results',result_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
  
    return text

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

# for video detection
def video(path,filename):
    lst =[]
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc('V','P','8','0') ##('XVID')
   #out = cv2.VideoWriter('./static/predict/output2.mp4', codec, fps , (width, height))
    out = cv2.VideoWriter('./static/predict/{}'.format(filename), codec, fps , (width, height))

    while True:
        ret, frame = cap.read()
        
        if ret == False:
            print('unable to read video')
            break


        results, text_list = yolo_predictions(frame,net)
        lst.append(text_list)

        print(fps)
        out.write(results)

        # cv2.namedWindow('YOLO',cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('YOLO', results)
        # if cv2.waitKey(1) == 27:
        #     break

    # extracting best output which have maximum confidence level
    
    a=list()
    large=0
    best=0
    for i in lst:
        if len(i)>0:
            a.append(i)

    for j in a:
        for k in j:
            for m in k:
                if k[1] > large:
                    large = k[1]
                    best = k[0]

    print(best)

    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    return best
#--------------------------------------------------------------------------------------------------------------------------------------

# if __name__ =="__main__":
    # for video
    # video(r"C:\Users\Admin\CDAC Python\Deployment\Pro\static\uploads\N66.mp4")

    # for image
    # object_detection(r"C:\Users\Admin\CDAC Python\Deployment\Pro\static\uploads\N66.jpeg")
