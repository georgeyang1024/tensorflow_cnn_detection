import time
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import numpy as np
import os
import cv2

#cv 初始化
import cnnNet

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.1_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
font = ImageFont.truetype("JingDianXiYuanJian-1.ttf", 20, encoding="utf-8")


#数据集地址
path='/Volumes/SDCard/img/sex/'
#模型保存地址
model_path = path + "model.ckpt"


#检测模型
from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader(model_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in sorted(var_to_shape_map):
    print("tensor_name: ", key)


flower_dict = {0:'男',1:'女'}

w=100
h=100
c=3

x_=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = cnnNet.inference(x_,False,regularizer)
# with tf.name_scope('loss'):
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_, name='loss')

# with tf.name_scope('Adam_optimizer'):
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy",)
#---------------------------网络结束---------------------------

lst_vars = []
for v in tf.global_variables():
    lst_vars.append(v)
    print(v.name, '....')

saver = tf.train.Saver(var_list=lst_vars)


with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())

    print(os.listdir(path))
    lastCheckPoint = tf.train.latest_checkpoint(path+"/")
    if not lastCheckPoint:
        lastCheckPoint = model_path + ".meta"
    print(lastCheckPoint)
    saver.restore(sess,lastCheckPoint)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 设置分辨率
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        faceRects = face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(32, 32)
        )

        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                # print(faceRect)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 3)  # 5控制绿色框的粗细
                img = Image.fromarray(frame[y:y + h, x:x + w])
                imgData = img.resize((100, 100), Image.ANTIALIAS)
                # data = [np.array(imgData).astype('float')]
                data = [np.asarray(imgData,dtype=np.float32)]
                # cv2.imshow("frame", data)

                feed_dict = {x_: data}
                classification_result = sess.run(logits, feed_dict)

                # print(classification_result)
                # print(tf.argmax(classification_result, 1).eval())
                output = []
                output = tf.argmax(classification_result, 1).eval()
                if output[0]<1 :
                    print(classification_result)
                # print(loss.eval({x_: data, y_: [output[0]]}))

                text = ""
                for i in range(len(output)):
                    text = flower_dict[output[i]]

                pil_im = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_im)

                fontSize = font.getsize(text)
                draw.text((x + w / 2 - fontSize[0] / 2, y), text, (0, 0, 255), font=font)
                frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGBA2RGB)
            cv2.imshow("test", frame)
        else:
            cv2.imshow('test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


