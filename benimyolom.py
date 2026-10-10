import tensorflow as tf
import keras
import keras.layers as kl
import numpy as np
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import read_classes , read_anchors , preprocess_image
from PIL import ImageDraw , ImageFont

class_names = read_classes('model_data/coco_classes.txt')
anchors = read_anchors("model_data/yolo_anchors.txt")

#print(class_names[2])

print("------------------")


#print(anchors[0])



yolo_model = kl.TFSMLayer("model_data/" , call_endpoint="serving_default")

test_input = np.zeros((1,608,608,3) , dtype="float32")
image , image_data = preprocess_image("images/0018.jpg" , model_image_size=(608,608))
out = yolo_model(image_data)
#print("OUT YAZILMASI GEREK**********************")
#print(out['conv2d_22'])


#print("*-*-*-*-*-*--*-**--*-*-*-*-*-*-*-*-*-*-*-*-**-")

box_xy , box_wh , box_confidence , box_class_probs = yolo_head(out["conv2d_22"] , anchors , len(class_names))
print(box_xy.shape ,
        box_wh.shape ,
            box_confidence.shape ,
                box_class_probs.shape)


mask1 = box_confidence >= 0.1
print(mask1.shape)
mask1 = tf.squeeze(mask1)
print(mask1.shape) # her bir griddeki 5 box type a göre true mı false mı
squezzed_box_xy = tf.squeeze(box_xy)
filtered1 = tf.boolean_mask(squezzed_box_xy , mask1) # MERKEZ KOORDİNATLAR 
box_number , trash = filtered1.shape
print(filtered1)
squezzed_box_wh = tf.squeeze(box_wh)
filtered2 = tf.boolean_mask(squezzed_box_wh , mask1) # WİDTH VE HEİGHTLAR
print("------------------------")
print(filtered2)
squezzed_box_class_probs = tf.squeeze(box_class_probs)

filtered3 = tf.boolean_mask(squezzed_box_class_probs , mask1) # CLASS PROBS
print("****************************" , filtered3.shape)


alper = list()
for i in range(0,box_number):
    en_buyuk = 0
    for j in range(0,80):
        temp = filtered3[i][j]
        if temp > en_buyuk:
            en_buyuk = temp
            indeks = j
    alper.append([indeks , en_buyuk])

squezzed_box_confidence = tf.squeeze(box_confidence)

filtered4 = tf.boolean_mask(squezzed_box_confidence, mask1)


for i in range(0,box_number):
    alper[i] =  (alper[i][0] , alper[i][1] * filtered4[i])

print(alper)
print("3 TANE BOXIMIZ VA RVE BUNLARIN MERKEZ KOORDİNATLARI" , filtered1)

font1 = ImageFont.truetype("font/FiraMono-Medium.otf", 16)

nonmaxsupression = list()

for i in range(0,box_number):
    x,y = filtered1[i]
    x = x*1280
    y = y*720

    width , height = filtered2[i]

    width = width*1280
    height = height*720

    x_min = x - width/2
    x_max = x + width/2
    y_min = y - height/2
    y_max = y + height/2
    kutu = {"skor" : float(alper[i][1]) , "sınıf" : alper[i][0] , "x_min" :   float(x_min) , "y_min" :  float(y_min) ,"x_max" :  float(x_max) , "y_max" : float(y_max)}
    nonmaxsupression.append(kutu)

    #draw = ImageDraw.Draw(image)
    #draw.rectangle([x_min , y_min , x_max , y_max] , outline='red' , width=3)
    #draw.text((x_min , y_min) , text= class_names[alper[i][0]] + str(alper[i][1]) , fill='red')
    #draw.text((x_min , y_min) , text= f"{class_names[alper[i][0]]} {alper[i][1].numpy():.2f}" , fill='red' , font=font1)
image.save("out/alperdeneme.jpg")


print(nonmaxsupression)


def iou(kutu1 , kutu2):
    #kutu şekillerimiz bir dict ve sırasıyla skor ,sınıf , 4 tane koordinat için float
    if kutu1["x_min"] >= kutu2["x_min"]:
        iou_x_min = kutu1["x_min"]
    else :
        iou_x_min = kutu2["x_min"]
    #************************************
    if kutu1["x_max"] >= kutu2["x_max"]:
        iou_x_max = kutu2["x_max"]
    else:
        iou_x_max = kutu1["x_max"]
    #************************************
    if kutu1["y_min"] >= kutu2["y_min"]:
        iou_y_min = kutu1["y_min"]
    else:
        iou_y_min = kutu2["y_min"]
    #*************************************
    if kutu1["y_max"] >= kutu2["y_max"]:
        iou_y_max = kutu2["y_max"]
    else:
        iou_y_max = kutu1["y_max"]
    genislik = max(0 , (iou_x_max - iou_x_min))
    yukseklik = max(0 , (iou_y_max - iou_y_min))
    intersection_area = genislik * yukseklik

    union = ((kutu1["x_max"] - kutu1["x_min"]) * (kutu1["y_max"] - kutu1["y_min"])) +  ((kutu2["x_max"] - kutu2["x_min"]) * (kutu2["y_max"] - kutu2["y_min"])) - intersection_area

    return intersection_area / union

nonmaxsupression.sort(key= lambda kutu : kutu["skor"] , reverse=True)
kalanlar = list()
while nonmaxsupression:
    best = nonmaxsupression.pop(0)
    kalanlar.append(best)

    for i in nonmaxsupression[:]:
        if i["sınıf"] == best["sınıf"] and iou(i , best) > 0.6:
            nonmaxsupression.remove(i)


box_number = len(kalanlar)

for i in range(0 , box_number):
    draw = ImageDraw.Draw(image)
    draw.rectangle([kalanlar[i]["x_min"],kalanlar[i]["y_min"],kalanlar[i]["x_max"],kalanlar[i]["y_max"]] , width=3 ,outline='red')

image.save("out/alperdeneme.jpg")
    
    







            












    





#filtered1 = tf.boolean_mask(box_xy , mask1)





