from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import random
import sys

serif = ImageFont.truetype('FreeSerif.ttf', 50)
sans_serif = ImageFont.truetype('FreeSans.ttf', 50)
dummy_draw = ImageDraw.Draw(Image.new('L', (1, 1)))

def percent(x, n):
  return '%.0f%%' % (x * 100.0 / n)

def clamp(x):
  return max(0, min(255, int(x * 256)))

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

  # TODO: Get the network learning bilinear interpolation, remove this when that works
  image_2x = image_1x.resize((w2, h2), Image.BILINEAR)

  # Visit every 2x2 pixel group
  for ry in xrange(0, h1):
    for rx in xrange(0, w1):
      data_in = []

      # Save the pixel that this group was downsampled to and the surrounding area
      for dy in xrange(-radius, radius + 1):
        y = ry + dy
        for dx in xrange(-radius, radius + 1):
          x = rx + dx
          if 0 <= x < w1 and 0 <= y < h1:
            data_in.append(image_1x.getpixel((x, y)) / 255.0)
          else:
            data_in.append(1.0)

      # Save the original pixel group (the network will try to predict this)
      d00 = image_2x.getpixel((2 * rx, 2 * ry))
      d10 = image_2x.getpixel((2 * rx + 1, 2 * ry))
      d01 = image_2x.getpixel((2 * rx, 2 * ry + 1))
      d11 = image_2x.getpixel((2 * rx + 1, 2 * ry + 1))
      data_out = [
        d00 / 255.0,
        d10 / 255.0,
        d01 / 255.0,
        d11 / 255.0,
      ]

      # TODO: Rotate and reflect this to get more samples
      samples.append((data_in, data_out, d00 + d10 + d01 + d11))

def test_solution(image_1x, image_2x, image_2x_name, radius, x, layer_2x2):
  samples = []
  pixels_2x = []
  w1, h1 = image_1x.size
  w2, h2 = image_2x.size
  super_image_2x = Image.new('L', (w2, h2))
  generate_samples(image_1x, image_2x, radius, samples)

  for ry in xrange(0, h1):
    row1 = []
    row2 = []
    for rx in xrange(0, w1):
      pixels_2x2 = layer_2x2.eval(feed_dict={
        x: [samples[rx + ry * w1][0]],
      })[0]
      super_image_2x.putpixel((2 * rx, 2 * ry), clamp(pixels_2x2[0]))
      super_image_2x.putpixel((2 * rx + 1, 2 * ry), clamp(pixels_2x2[1]))
      super_image_2x.putpixel((2 * rx, 2 * ry + 1), clamp(pixels_2x2[2]))
      super_image_2x.putpixel((2 * rx + 1, 2 * ry + 1), clamp(pixels_2x2[3]))

  super_image_2x.save(open(image_2x_name, 'w'))

def load_cached_ascii_samples(radius):
  try:
    return np.load('radius_1.npy')
  except IOError:
    pass
  samples = generate_ascii_samples(radius)
  np.save('radius_1.npy', samples)
  return samples

def generate_ascii_samples(radius):
  glyphs = []
  samples = []

  glyphs += [render_glyph(chr(c), serif) for c in xrange(0x21, 0x7F)]
  glyphs += [render_glyph(chr(c), sans_serif) for c in xrange(0x21, 0x7F)]

  print 'generating samples'

  for i in xrange(0, len(glyphs)):
    sys.stdout.write('\r' + percent(i, len(glyphs)))
    sys.stdout.flush()
    image_1x, image_2x = glyphs[i]
    generate_samples(image_1x, image_2x, radius, samples)

  print '\rdone'

  white = [x for x in samples if x[2] > 4 * 254]
  black = [x for x in samples if x[2] < 4 * 1]
  other = [x for x in samples if 4 * 1 <= x[2] <= 4 * 254]

  print 'white: %s (%d samples)' % (percent(len(white), len(samples)), len(white))
  print 'black: %s (%d samples)' % (percent(len(black), len(samples)), len(black))
  print 'other: %s (%d samples)' % (percent(len(other), len(samples)), len(other))

  print 'applying limiting'

  random.shuffle(white)
  random.shuffle(black)
  white = white[:len(other) / 10]
  black = black[:len(other) / 10]
  samples = white + black + other

  print 'white: %s (%d samples)' % (percent(len(white), len(samples)), len(white))
  print 'black: %s (%d samples)' % (percent(len(black), len(samples)), len(black))
  print 'other: %s (%d samples)' % (percent(len(other), len(samples)), len(other))

  return np.array(samples)

def weight_variable(shape, values=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(values if values is not None else initial)

def bias_variable(shape, value=0.1):
  initial = tf.constant(value, shape=shape)
  return tf.Variable(initial)

def main():
  radius = 1 # The kernel has size 2 * radius + 1
  samples = load_cached_ascii_samples(radius)
  sess = tf.InteractiveSession()

  input_size = len(samples[0][0])
  output_size = len(samples[0][1])
  # sizes = [input_size, 32, 24, 16, 8, output_size]
  # sizes = [input_size, 8, 8, output_size]
  sizes = [input_size, output_size]

  x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
  y = tf.placeholder(tf.float32, shape=[None, output_size], name='y')

  # A B C
  # D E F
  # G H I

  weights = [
    [
      [0.25 * 0.25, 0.0,         0.0,         0.0], # A
      [0.75 * 0.25, 0.75 * 0.25, 0.0,         0.0], # B
      [0.0,         0.25 * 0.25, 0.0,         0.0], # C
      [0.75 * 0.25, 0.0,         0.75 * 0.25, 0.0], # D
      [0.75 * 0.75, 0.75 * 0.75, 0.75 * 0.75, 0.75 * 0.75], # E
      [0.0,         0.75 * 0.25, 0.0,         0.75 * 0.25], # F
      [0.0,         0.0,         0.25 * 0.25, 0.0], # G
      [0.0,         0.0,         0.75 * 0.25, 0.75 * 0.25], # H
      [0.0,         0.0,         0.0,         0.25 * 0.25], # I
    ],
  ]

  biases = [
    0.0,
  ]

  layer = x
  for i in xrange(1, len(sizes)):
    size_before, size_after = sizes[i - 1:i + 1]
    weights = weight_variable([size_before, size_after], weights[i - 1])
    bias = bias_variable([size_after], biases[i - 1])
    layer = tf.nn.relu(tf.matmul(layer, weights) + bias)

  error = tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(layer, [-1, output_size]) - y), reduction_indices=[1]))
  train_step = tf.train.AdamOptimizer(0.0001).minimize(error)
  sess.run(tf.initialize_all_variables())

  g_1x, g_2x = render_glyph('g', serif)
  g_2x_linear = g_1x.resize(g_2x.size, Image.BILINEAR)

  g_2x.save(open('image_2x.png', 'w'))
  g_2x_linear.save(open('image_2x_linear.png', 'w'))

  test_batch = random.sample(samples, 100)

  # Training loop
  for i in range(10000):
    if i % 50 == 0:
      test_solution(g_1x, g_2x, 'image_2x_super_%d.png' % i, radius, x, layer)

      current = error.eval(feed_dict={
        x: [image_1x for image_1x, image_2x, total in test_batch],
        y: [image_2x for image_1x, image_2x, total in test_batch],
      })

      print 'step %d, error %g' % (i, current)

    batch = random.sample(samples, 100)
    train_step.run(feed_dict={
      x: [image_1x for image_1x, image_2x, total in batch],
      y: [image_2x for image_1x, image_2x, total in batch],
    })

main()
