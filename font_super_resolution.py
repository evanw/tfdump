from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import random
import sys

dummy_draw = ImageDraw.Draw(Image.new('L', (1, 1)))

def percent(x, n):
  return '%.0f%%' % (x * 100.0 / n)

def render_glyph(c, font):
  w, h = dummy_draw.textsize(c, font=font)
  w, h = w + w % 2, h + h % 2 # Round the size up to the next even number
  image_2x = Image.new('L', (w, h))
  draw = ImageDraw.Draw(image_2x)
  draw.rectangle((0, 0, w, h), fill='#FFFFFF')
  draw.text((0, 0), c, fill='#000000', font=font)
  image_1x = image_2x.resize((w / 2, h / 2), Image.BILINEAR)
  return image_1x, image_2x

def generate_samples(image_1x, image_2x, radius, samples):
  w1, h1 = image_1x.size
  w2, h2 = image_2x.size
  assert w1 * 2 == w2 and h1 * 2 == h2

  # Visit every 2x2 pixel group
  for ry in xrange(0, h1):
    for rx in xrange(0, w1):
      data_in = []

      # Save the pixel that downsampled to and the surrounding area
      for dy in xrange(-radius, radius + 1):
        y = ry + dy
        for dx in xrange(-radius, radius + 1):
          x = rx + dx
          if 0 <= x < w1 and 0 <= y < h1:
            data_in.append(float(image_1x.getpixel((x, y))))
          else:
            data_in.append(255.0)

      # Save the original pixel group (the network will try to predict this)
      data_out = [
        float(image_2x.getpixel((rx, ry))),
        float(image_2x.getpixel((rx + 1, ry))),
        float(image_2x.getpixel((rx, ry + 1))),
        float(image_2x.getpixel((rx + 1, ry + 1))),
      ]

      # TODO: Rotate and reflect this to get more samples
      samples.append((data_in, data_out, sum(data_out)))

def generate_ascii_samples(radius):
  glyphs = []
  samples = []

  serif = ImageFont.truetype('FreeSerif.ttf', 50)
  glyphs += [render_glyph(chr(c), serif) for c in xrange(0x21, 0x7F)]

  # sans_serif = ImageFont.truetype('FreeSans.ttf', 50)
  # glyphs += [render_glyph(chr(c), sans_serif) for c in xrange(0x21, 0x7F)]

  print 'generating samples'

  for i in xrange(0, len(glyphs)):
    sys.stdout.write('\r' + percent(i, len(glyphs)))
    sys.stdout.flush()
    image_1x, image_2x = glyphs[i]
    generate_samples(image_1x, image_2x, radius, samples)

  print '\rdone'

  white = [x for x in samples if x[2] == 4 * 255]
  black = [x for x in samples if x[2] == 0]
  other = [x for x in samples if 0 < x[2] < 4 * 255]

  print 'white: %s (%d samples)' % (percent(len(white), len(samples)), len(white))
  print 'black: %s (%d samples)' % (percent(len(black), len(samples)), len(black))
  print 'other: %s (%d samples)' % (percent(len(other), len(samples)), len(other))

  print 'applying limiting'

  limit = min(len(white), len(black), len(other) / 10)
  random.shuffle(white)
  random.shuffle(black)
  white = white[:limit]
  black = black[:limit]
  samples = white + black + other

  print 'white: %s (%d samples)' % (percent(len(white), len(samples)), len(white))
  print 'black: %s (%d samples)' % (percent(len(black), len(samples)), len(black))
  print 'other: %s (%d samples)' % (percent(len(other), len(samples)), len(other))

  return samples

def fuzzy_zeros(w, h):
  rows = []
  for x in xrange(0, w):
    cols = []
    for y in xrange(0, h):
      cols.append(random.gauss(0, 0.01))
    rows.append(cols)
  return rows

def weight_variable(shape, value):
  initial = tf.constant(value, shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def main():
  radius = 2 # The kernel has size 2 * radius + 1
  samples = generate_ascii_samples(radius)
  sess = tf.InteractiveSession()

  input_size = len(samples[0][0])
  output_size = len(samples[0][1])
  hidden1 = 32

  x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
  y = tf.placeholder(tf.float32, shape=[None, output_size], name='y')

  values1 = fuzzy_zeros(input_size, hidden1)
  values2 = fuzzy_zeros(hidden1, output_size)

  # The first feature vector should start off as a pass-through for the center pixel
  values1[input_size / 2][0] = 1.0

  # The collector layer should start off as a pass-through from the first feature vector to all four pixels in the 2x2 group
  for i in xrange(0, output_size):
    values2[0][i] = 1.0

  weights1 = weight_variable([input_size, hidden1], values1)
  bias1 = bias_variable([hidden1])
  layer1 = tf.nn.relu(tf.matmul(x, weights1) + bias1)

  weights2 = weight_variable([hidden1, output_size], values2)
  bias2 = bias_variable([output_size])
  layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + bias2)

  error = tf.reduce_mean(tf.reduce_sum(tf.square(layer2 - y), reduction_indices=[1]))
  train_step = tf.train.AdamOptimizer(0.0001).minimize(error)
  sess.run(tf.initialize_all_variables())

  # Training loop
  for i in range(10000):
    batch = random.sample(samples, 50)

    x_values = [image_1x for image_1x, image_2x, total in batch]
    y_values = [image_2x for image_1x, image_2x, total in batch]

    current = error.eval(feed_dict={
      x: x_values,
      y: y_values,
    })

    print 'step %d, error %g' % (i, current)

    train_step.run(feed_dict={
      x: x_values,
      y: y_values,
    })

main()
