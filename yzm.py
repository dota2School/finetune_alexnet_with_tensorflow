content="content"
from PIL import Image
import tensorflow as tf
import numpy as np
import random
import shenjian as sj
import base64
from io import BytesIO
import re
import json
import urllib.request as client

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
            'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
PADING = [' ']


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def load_data(file_name):
    file = open(file_name)
    content = file.read()
    content = content.replace("\n","")
    imgs = re.findall(r'content.*?\}', content)
    result = []
    for img in imgs:
        img_x = re.findall(r'content:.*?result:', img)
        img_y = re.findall(r'result:.*?\}', img)
        img_x = img_x[0].replace("content:", "").replace("result:", "")
        img_y = json.loads(img_y[0].replace("result:", ""))
        if 'result' in img_y:
            if len(img_y['result']) == 4 and re.search(r'[0-9|a-z|A-Z]+$', img_y['result']):
                result.append((img_x, img_y['result']))
    return result

url = 'http://demo.shenjianshou.cn/tensor/baidu/'


def request():
    content = client.urlopen(url=url).read()
    content = content.decode('utf-8')
    data = json.loads(content)
    data['value'] = str(data['value'])
    size = len(data['value'])
    for i in range(size, 10):
        data['value'] += ' '
    return data['content'], data['value']


class Model(object):

    def __init__(self, text_set=number+PADING, captcha_size=10, width=120, height=26):
        self.text_set = text_set
        self.captcha_size = captcha_size
        self.width = width
        self.height = height
        self.captcha_len = len(text_set)
        self.X = tf.placeholder(tf.float32, [None, self.width*self.height])
        self.Y = tf.placeholder(tf.float32, [None, self.captcha_size*self.captcha_len])
        self.keep_prob = tf.placeholder(tf.float32)
        self.x = tf.reshape(self.X, shape=[-1, self.height, self.width, 1])

        self.w_alpha = 0.01
        self.b_alpha = 0.1
        #定义三层卷积层

        self.w_c1 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 1, 32]))
        self.b_c1 = tf.Variable(self.b_alpha * tf.random_normal([32]))
        self.conv1_a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.x, self.w_c1, strides=[1, 1, 1, 1], padding='SAME'), self.b_c1))
        self.conv1_b = tf.nn.max_pool(self.conv1_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.conv1 = tf.nn.dropout(self.conv1_b, self.keep_prob)

        self.w_c2 = tf.Variable(self.w_alpha*tf.random_normal([3, 3, 32, 64]))
        self.b_c2 = tf.Variable(self.b_alpha*tf.random_normal([64]))
        self.conv2_a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv1, self.w_c2, strides=[1, 1, 1, 1], padding='SAME'), self.b_c2))
        self.conv2_b = tf.nn.max_pool(self.conv2_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.conv2 = tf.nn.dropout(self.conv2_b, self.keep_prob)

        self.w_c3 = tf.Variable(self.w_alpha*tf.random_normal([3, 3, 64, 64]))
        self.b_c3 = tf.Variable(self.b_alpha*tf.random_normal([64]))
        self.conv3_a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv2, self.w_c3, strides=[1, 1, 1, 1], padding='SAME'), self.b_c3))
        self.conv3_b = tf.nn.max_pool(self.conv3_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.conv3 = tf.nn.dropout(self.conv3_b, self.keep_prob)

        #全连接层

        self.w_d = tf.Variable(self.w_alpha*tf.random_normal([15*4*64, 1024]))
        self.b_d = tf.Variable(self.b_alpha*tf.random_normal([1024]))
        self.dense = tf.reshape(self.conv3, [-1, self.w_d.get_shape().as_list()[0]])
        self.dense = tf.nn.relu(tf.add(tf.matmul(self.dense, self.w_d), self.b_d))
        self.dense = tf.nn.dropout(self.dense, self.keep_prob)

        self.w_out = tf.Variable(self.w_alpha*tf.random_normal([1024, self.captcha_size*self.captcha_len]))
        self.b_out = tf.Variable(self.b_alpha*tf.random_normal([self.captcha_size*self.captcha_len]))
        self.out = tf.add(tf.matmul(self.dense, self.w_out), self.b_out)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.out))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.predict = tf.reshape(self.out, [-1, self.captcha_size, self.captcha_len])
        self.max_idx_p = tf.argmax(self.predict, 2)
        self.max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.captcha_size, self.captcha_len]), 2)
        self.correct_pred = tf.equal(self.max_idx_p, self.max_idx_l)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        #self.data = load_data("data")
        #self.test = load_data("test")
        self.data_index = 0
        self.test_index = 0
        self.session = sj.Session(auto_init=True)


    def random_captcha_text(self):
        captcha_text = []
        for i in range(self.captcha_size):
            c = random.choice(self.text_set)
            captcha_text.append(c)
        return captcha_text

    def gen_captcha_text_and_image(self):
        img_x, img_y = request()
        imgdata = base64.b64decode(img_x)
        img_x = Image.open(BytesIO(imgdata))
        if img_x.width != self.width or img_x.height != self.height:
            img_x = img_x.resize((self.width, self.height), Image.ANTIALIAS)
        img_x = img_x.point(lambda x: 255 if x > 125 else 0)
        img_x = np.array(img_x)
        self.data_index += 1
        return img_y, img_x

    def gen_captcha_text_and_image_test(self):
        img_x, img_y = request()
        imgdata = base64.b64decode(img_x)
        img_x = Image.open(BytesIO(imgdata))
        if img_x.width != self.width or img_x.height != self.height:
            img_x = img_x.resize((self.width, self.height), Image.ANTIALIAS)
        img_x = img_x.point(lambda x: 255 if x > 125 else 0)
        img_x = np.array(img_x)
        self.test_index += 1
        return img_y, img_x

    def text2vec(self, text):
        text_len = len(text)
        if text_len != self.captcha_size:
            raise ValueError("验证码长度不匹配")
        vector = np.zeros(self.captcha_len * self.captcha_size)

        def char2pos(c):
            if c == ' ':
                k = 10
                return k
            k = ord(c)-48
            if k > 9:
                k = ord(c)-55
                if k > 35:
                    k = ord(c) - 61
                    if k > 61:
                        raise ValueError('No Map '+c)
            return k
        for i, c in enumerate(text):
            idx = i*self.captcha_len+char2pos(c)
            vector[idx] = 1
        return vector

    def vec2text(self, vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_idx = c % self.captcha_len
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx == 10:
                char_code = ord(' ')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

    def get_next_batch(self, batch_size=128):
        batch_x = np.zeros([batch_size, self.width*self.height])
        batch_y = np.zeros([batch_size, self.captcha_len * self.captcha_size])
        for i in range(batch_size):
            text, image = self.gen_captcha_text_and_image()
            image = convert2gray(image)
            batch_x[i, :] = image.flatten() / 255
            batch_y[i, :] = self.text2vec(text)
        return batch_x, batch_y

    def get_next_batch_test(self, batch_size=128):
        batch_x = np.zeros([batch_size, self.width*self.height])
        batch_y = np.zeros([batch_size, self.captcha_len * self.captcha_size])
        for i in range(batch_size):
            text, image = self.gen_captcha_text_and_image_test()
            image = convert2gray(image)
            batch_x[i, :] = image.flatten() / 255
            batch_y[i, :] = self.text2vec(text)
        return batch_x, batch_y

    def train(self):
        step = 0
        while True:
            batch_x, batch_y = self.get_next_batch_test(64)
            _, loss_ = self.session.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
            step += 1
            print(step, loss_)
            if step % 10 == 0:
                batch_x_test, batch_y_test = self.get_next_batch(100)
                acc = self.session.run(self.accuracy, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.0})
                print(acc)
                if acc > 0.99:
                    return

    def serve(self, parm):
        parm = json.loads(parm)['argument']
        print('parm=  '+parm)
        img_data = base64.b64decode(parm)
        img_x = Image.open(BytesIO(img_data))
        if img_x.width != self.width or img_x.height != self.height:
            img_x = img_x.resize((self.width, self.height), Image.ANTIALIAS)
        img_x = img_x.point(lambda x: 255 if x > 125 else 0)
        img_x = np.array(img_x)
        img_x = convert2gray(img_x)
        img_x = img_x.flatten() / 255
        predict = tf.argmax(tf.reshape(self.out, [-1, self.captcha_size, self.captcha_len]), 2)
        text_list = self.session.run(predict, feed_dict={self.X: [img_x], self.keep_prob: 1})
        text = text_list[0].tolist()
        vector = np.zeros(self.captcha_size*self.captcha_len)
        print(text)
        i = 0
        for n in text:
            vector[i*self.captcha_len + n] = 1
            i += 1
        result = self.vec2text(vector)
        print(result)
        return str(result)
m = Model()
sj.run(m.train, m.serve)

