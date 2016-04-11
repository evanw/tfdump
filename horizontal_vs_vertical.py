# This classifies an image as containing either a horizontal or a vertical line

from PIL import Image, ImageDraw
import random
import math

random.seed(123)

def generate_sample():
  img = Image.new('RGB', (width * 4, height * 4))
  draw = ImageDraw.Draw(img)
  draw.rectangle((0, 0, width * 4, height * 4), fill='#FFFFFF')

  # Pick a random angle between 0 (horizontal) and 1 (vertical)
  angle = random.random()

  # Draw a random line with that angle
  c = math.cos(angle * math.pi / 2) * 1000
  s = math.sin(angle * math.pi / 2) * 1000
  x = (0.25 + 0.5 * random.random()) * width * 4
  y = (0.25 + 0.5 * random.random()) * height * 4
  if random.random() < 0.5:
    s = -s
  draw.line((x - c, y - s, x + c, y + s), fill='#000000')

  # Shrink the image down for anti-aliasing
  img = img.resize((width * 2, height * 2), Image.BILINEAR)
  img = img.resize((width, height), Image.BILINEAR)

  # Use grayscale image data
  pixels = [pixel[0] for pixel in img.getdata()]

  # Have the network say if the image contains either a horizontal or a vertical line
  labels = [float(angle < 0.5), float(angle >= 0.5)]

  # Normalization makes the difference between never converging at all and converging almost immediately
  mean = sum(pixel for pixel in pixels) / len(pixels)
  pixels = [(pixel - mean) / 255 for pixel in pixels]

  return pixels, labels

width, height = 28, 28
training_samples = [generate_sample() for i in xrange(500)]
testing_samples = [generate_sample() for i in xrange(100)]
print 'created samples'

import tensorflow as tf
sess = tf.InteractiveSession()

input_size = width * height
output_size = 2

x = tf.placeholder(tf.float32, shape=[None, input_size])
y_ = tf.placeholder(tf.float32, shape=[None, output_size])

sess.run(tf.initialize_all_variables())

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,width,height,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([(width/4) * (height/4) * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, (width/4)*(height/4)*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, output_size])
b_fc2 = bias_variable([output_size])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for index in range(100):
  batch = random.sample(testing_samples, 50)
  if index%10 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x: [i for i, o in batch],
        y_: [o for i, o in batch],
        keep_prob: 1.0})
    print("step %d, training accuracy %g"%(index, train_accuracy))
  train_step.run(feed_dict={
    x: [i for i, o in batch],
    y_: [o for i, o in batch],
    keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: [i for i, o in testing_samples],
    y_: [o for i, o in testing_samples],
    keep_prob: 1.0}))
