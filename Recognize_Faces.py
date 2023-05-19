import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys


from sklearn.model_selection import train_test_split

my_faces_path = './my_faces'
other_faces_path = './other _faces'
size = 64

imgs = []
labs = []


def getpaddingsize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)
    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right


def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top, bottom, left, right = getpaddingsize(img)

            # enlarge the image fulfil the edge part

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)


readData(my_faces_path)
readData(other_faces_path)

# Transform the image data into array
imgs = np.array(imgs)
labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])

# Randomly partioning the test and the traning data
train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0, 10))

# Set the parameter of the image such as width, length
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))

batch_size = 100
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_pro_5 = tf.placeholder(tf.float32)
keep_pro_75 = tf.placeholder(tf.float32)


def weightVarible(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Varible(init)


def baisVarible(shape):
    init = tf.random_normal(shape)
    return tf.varible(init)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def cnnlayer():
    # First layer
    w1 = weightVarible([3, 3, 3, 32])  # define the size of convolution kernel(filter), input and output
    b1 = baisVarible([32])
    # convolution
    conv1 = tf.nn.relu(conv2d(x, w1) + b1)
    # pooling
    pool1 = maxPool(conv1)
    # Drop out randomly in case of over_fitting
    drop1 = dropout(pool1, keep_pro_5)

    # Second  layer
    w2 = weightVarible([3, 3, 32, 64])  # define the size of convolution kernel(filter), input and output
    b2 = baisVarible([64])
    # convolution
    conv2 = tf.nn.relu(conv2d(x, w2) + b2)
    # pooling
    pool2 = maxPool(conv2)
    # Drop out randomly in case of over_fitting
    drop2 = dropout(pool2, keep_pro_5)

    # Third layer
    w3 = weightVarible([3, 3, 64, 64])  # define the size of convolution kernel(filter), input and output
    b3 = baisVarible([64])
    # convolution
    conv3 = tf.nn.relu(conv2d(x, w3) + b3)
    # pooling
    pool3 = maxPool(conv3)
    # Drop out randomly in case of over_fitting
    drop3 = dropout(pool3, keep_pro_5)

    # Fully connected layer
    wf = weightVarible([8 * 8 * 64, 512])  # define the size of convolution kernel(filter), input and output
    bf = baisVarible([512])
    # Drop out randomly in case of over_fitting
    drop3_flat = tf.reshape(drop3, [-1, 8 * 8 * 64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, wf) + bf)
    dropf = dropout(dense, keep_pro_75)

    # Output layer
    wout = weightVarible([512, 2])
    bout = baisVarible([2])
    out = tf.add(tf.matmul(dropf, wout), bout)

    return out


output = cnnlayer()
predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess = tf.Sessiob()
saver.restore(sess, tf.train.latest_checkpoint('.'))


def my_faces(image):
    res = sess.run(predict, feed_dict={x: [image / 255.0], keep_pro_5: 1.0, keep_pro_75: 1.0})
    if res[0] == 1:
        return True
    else:
        False


detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):

        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2]

        face = cv2.resize(face, (size, size))
        print('Is this my face' % my_faces)

        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

sess.close()
