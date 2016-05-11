from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import random
import shutil
import sys
import os

serif = ImageFont.truetype('FreeSerif.ttf', 50)
sans_serif = ImageFont.truetype('FreeSans.ttf', 50)
dummy_draw = ImageDraw.Draw(Image.new('L', (1, 1)))

def percent(x, n):
  return '%.0f%%' % (x * 100.0 / n)

def clamp(x):
  return max(0, min(255, int(x * 256)))

def render_text(text, font):
  w, h = dummy_draw.textsize(text, font=font)
  w, h = w + w % 2, h + h % 2 # Round the size up to the next even number
  image_2x = Image.new('L', (w, h))
  draw = ImageDraw.Draw(image_2x)
  draw.rectangle((0, 0, w, h), fill='#FFFFFF')
  draw.text((0, 0), text, fill='#000000', font=font)
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
  name = 'fsr_radius_%d.npy' % radius
  try:
    return np.load(name)
  except IOError:
    pass
  samples = generate_ascii_samples(radius)
  np.save(name, samples)
  return samples

def generate_ascii_samples(radius):
  glyphs = []
  samples = []

  glyphs += [render_text(chr(c), serif) for c in xrange(0x21, 0x7F)]
  glyphs += [render_text(chr(c), sans_serif) for c in xrange(0x21, 0x7F)]

  print 'generating samples'

  for i in xrange(0, len(glyphs)):
    sys.stdout.write('\r' + percent(i, len(glyphs)))
    sys.stdout.flush()
    image_1x, image_2x = glyphs[i]
    generate_samples(image_1x, image_2x, radius, samples)

  print '\rdone'

  return np.array(samples)

def fill(shape, callback):
  return callback() if not shape else [fill(shape[1:], callback) for x in xrange(shape[0])]

def main():
  radius = 2 # The kernel has size 2 * radius + 1
  samples = load_cached_ascii_samples(radius)
  sess = tf.InteractiveSession()

  input_size = len(samples[0][0])
  output_size = len(samples[0][1])
  sizes = [input_size, 8, output_size]

  x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
  y = tf.placeholder(tf.float32, shape=[None, output_size], name='y')

  layer = x
  for i in xrange(1, len(sizes)):
    size_before, size_after = sizes[i - 1:i + 1]
    weights = tf.Variable(fill([size_before, size_after], lambda: random.uniform(0.5, 1.5) / (size_before * size_after)))
    bias = tf.Variable(fill([size_after], lambda: random.uniform(0.5, 1.5) / size_after))
    layer = tf.nn.relu(tf.matmul(layer, weights) + bias)

  error = tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(layer, [-1, output_size]) - y), reduction_indices=[1]))
  train_step = tf.train.AdamOptimizer(0.001).minimize(error)
  sess.run(tf.initialize_all_variables())

  test_1x, test_2x = render_text('test', serif)
  test_2x_linear = test_1x.resize(test_2x.size, Image.BILINEAR)

  shutil.rmtree('./fsr_data/', ignore_errors=True)
  os.mkdir('./fsr_data')
  test_1x.save(open('./fsr_data/image_1x.png', 'w'))
  test_2x.save(open('./fsr_data/image_2x.png', 'w'))
  test_2x_linear.save(open('./fsr_data/image_2x_linear.png', 'w'))

  test_batch = random.sample(samples, 100)

  # Training loop
  for i in range(10000):
    if i % 50 == 0:
      test_solution(test_1x, test_2x, './fsr_data/image_2x_super_%d.png' % i, radius, x, layer)

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
