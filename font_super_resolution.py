from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import random
import shutil
import json
import math
import sys
import os

serif = ImageFont.truetype('FreeSerif.ttf', 50)
sans_serif = ImageFont.truetype('FreeSans.ttf', 50)
dummy_draw = ImageDraw.Draw(Image.new('L', (1, 1)))

def percent(x, n):
  return '%.0f%%' % (x * 100.0 / n)

def clamp(x):
  return max(0, min(255, int(x * 256)))

def render_text(text, font, offset_x=0, offset_y=0):
  w, h = dummy_draw.textsize(text, font=font)
  w += offset_x
  h += offset_y
  w, h = w + w % 2, h + h % 2 # Round the size up to the next even number
  image_1x = Image.new('L', (w / 2, h / 2))
  image_2x = Image.new('L', (w, h))
  draw = ImageDraw.Draw(image_2x)
  draw.rectangle((0, 0, w, h), fill='#FFFFFF')
  draw.text((offset_x, offset_y), text, fill='#000000', font=font)

  # Gamma-correct downsampling
  pixels = []
  for y in xrange(0, h, 2):
    for x in xrange(0, w, 2):
      d00 = image_2x.getpixel((x, y))
      d10 = image_2x.getpixel((x + 1, y))
      d01 = image_2x.getpixel((x, y + 1))
      d11 = image_2x.getpixel((x + 1, y + 1))
      pixels.append(round(255 * math.pow((
        math.pow(d00 / 255.0, 2.2) +
        math.pow(d10 / 255.0, 2.2) +
        math.pow(d01 / 255.0, 2.2) +
        math.pow(d11 / 255.0, 2.2)
      ) / 4, 1 / 2.2)))
  image_1x.putdata(pixels)

  return image_1x, image_2x

def generate_samples(image_1x, image_2x, radius, samples_in, samples_out):
  w1, h1 = image_1x.size
  w2, h2 = image_2x.size
  assert w1 * 2 == w2 and h1 * 2 == h2

  # Visit every 2x2 pixel group
  for ry in xrange(h1):
    for rx in xrange(w1):
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

      samples_in.append(data_in)
      samples_out.append(data_out)

def test_solution(image_1x, image_2x, image_2x_name, radius, x, layer_2x2):
  samples_in = []
  samples_out = []
  pixels_2x = []
  w1, h1 = image_1x.size
  w2, h2 = image_2x.size
  generate_samples(image_1x, image_2x, radius, samples_in, samples_out)
  pixels_2x2 = layer_2x2.eval(feed_dict={
    x: samples_in,
  })

  pixels = []
  for ry in xrange(h1):
    for rx in xrange(w1):
      pixel_2x2 = pixels_2x2[rx + ry * w1]
      pixels.append(clamp(pixel_2x2[0]))
      pixels.append(clamp(pixel_2x2[1]))
    for rx in xrange(w1):
      pixel_2x2 = pixels_2x2[rx + ry * w1]
      pixels.append(clamp(pixel_2x2[2]))
      pixels.append(clamp(pixel_2x2[3]))

  super_image_2x = Image.new('L', (w2, h2))
  super_image_2x.putdata(pixels)
  super_image_2x.save(open(image_2x_name, 'w'))

def load_cached_ascii_samples(radius):
  name = 'fsr_radius_%d.npz' % radius

  print 'loading', name
  try:
    data = np.load(name)
    print 'done loading'
    return data['samples_in'], data['samples_out']
  except IOError:
    pass

  print 'generating', name
  samples_in, samples_out = generate_ascii_samples(radius)
  np.savez(name, samples_in=samples_in, samples_out=samples_out)
  print '\rdone generating'
  return samples_in, samples_out

def generate_ascii_samples(radius):
  glyphs = []
  samples_in = []
  samples_out = []

  # Make sure training data includes all 2x2 downsamples, not just those with even coordinates
  for offset_x in range(2):
    for offset_y in range(2):
      glyphs += [render_text(chr(c), serif, offset_x, offset_y) for c in xrange(0x21, 0x7F)]
      glyphs += [render_text(chr(c), sans_serif, offset_x, offset_y) for c in xrange(0x21, 0x7F)]

  for i in xrange(len(glyphs)):
    sys.stdout.write('\r' + percent(i, len(glyphs)))
    sys.stdout.flush()
    image_1x, image_2x = glyphs[i]
    generate_samples(image_1x, image_2x, radius, samples_in, samples_out)

  samples_in = np.array(samples_in, np.float32)
  samples_out = np.array(samples_out, np.float32)

  return samples_in, samples_out

def fill(shape, callback):
  return callback() if not shape else [fill(shape[1:], callback) for x in xrange(shape[0])]

def main():
  radius = 1 # The kernel has size 2 * radius + 1
  samples_in, samples_out = load_cached_ascii_samples(radius)
  sess = tf.InteractiveSession()

  input_size = len(samples_in[0])
  output_size = len(samples_out[0])
  sizes = [input_size, 8, output_size]

  x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
  y = tf.placeholder(tf.float32, shape=[None, output_size], name='y')

  final_layer = x
  layer_weights = []
  layer_biases = []

  for i in xrange(1, len(sizes)):
    size_before, size_after = sizes[i - 1:i + 1]
    weights = tf.Variable(fill([size_before, size_after], lambda: random.uniform(0.5, 1.5) / (size_before * size_after)))
    bias = tf.Variable(fill([size_after], lambda: random.uniform(0.5, 1.5) / size_after))
    layer_weights.append(weights)
    layer_biases.append(bias)
    final_layer = tf.nn.relu(tf.matmul(final_layer, weights) + bias)

  error = tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(final_layer, [-1, output_size]) - y), reduction_indices=[1]))
  train_step = tf.train.AdamOptimizer(0.01).minimize(error)
  sess.run(tf.initialize_all_variables())

  test_1x, test_2x = render_text('test', serif, 0, 0)
  test_2x_linear = test_1x.resize(test_2x.size, Image.BILINEAR)

  shutil.rmtree('./fsr_data/', ignore_errors=True)
  os.mkdir('./fsr_data')
  test_1x.save(open('./fsr_data/image_1x.png', 'w'))
  test_2x.save(open('./fsr_data/image_2x.png', 'w'))
  test_2x_linear.save(open('./fsr_data/image_2x_linear.png', 'w'))

  test_batch_indices = random.sample(xrange(len(samples_in)), 100)
  test_batch_in = [samples_in[i] for i in test_batch_indices]
  test_batch_out = [samples_out[i] for i in test_batch_indices]

  # Training loop
  for i in xrange(100 * 1000 + 1):
    if i % 1000 == 0:
      open('fsr_model_%d.json' % radius, 'w').write(json.dumps({
        'sizes': sizes,
        'weights': [[[float(a) for a in b] for b in c] for c in sess.run(layer_weights)],
        'biases': [[float(a) for a in b] for b in sess.run(layer_biases)],
      }, indent=2))

      test_solution(test_1x, test_2x, './fsr_data/image_2x_super_%d.png' % i, radius, x, final_layer)

      current = error.eval(feed_dict={
        x: test_batch_in,
        y: test_batch_out,
      })

      print 'step %d, error %g' % (i, current)

    batch_indices = random.sample(xrange(len(samples_in)), 100)
    train_step.run(feed_dict={
      x: [samples_in[i] for i in batch_indices],
      y: [samples_out[i] for i in batch_indices],
    })

main()
