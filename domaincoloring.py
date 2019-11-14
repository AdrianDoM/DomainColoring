#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

eps = 0.005

def domain_coloring(f, limits=[-5, 5, -5, 5], samples=1000, steps=None,
                    raygrid=True, eps=0.0025, dynamic_eps=False):
  """
  Draw a domain coloring plot of the given function.

  Parameters
  ----------
  f : function
    The complex function to plot as a domain coloring.

  limits : 4-tuple, optional
    The limits of the plot given as '[xmin, xmax, ymin, ymax]'.
    Default: '[-5, 5, -5, 5]'

  samples : int or 2-tuple, optional
    The number of samples to take for each coordinate. If it is an int, the same
    number of samples is taken for the x and y coordinates. If it is a 2-tuple,
    the first number is be used for the x coordinate and the second for y.
    Ignored if *steps* is provided.
    Default: '100'

  steps : float or 2-tuple, optional
    The distance between each sample for each coordinate. If provided, *samples*
    is ignored. If is an float, the same step size is used for the x and y
    coordinates. If it is a 2-tuple, the first number is used for the x
    coordinate and the second for y.

  raygrid : bool, optional
    Whether or not to highlight polar rays in white.
    Default: 'True'

  eps : float, optional
    The difference in angle to the rays of the polar grid of a ray to be
    considered part of the grid. Ignored if *raygrid* is False.
    Default: '0.0025'
  
  dynamic_eps : bool, optional
    Whether or not to scale eps according to the distance to the origin. This
    makes rays in the domain have a constant width. Ignored if *raygrid* is
    False.
    Default: 'False'
  """

  if not callable(f):
    raise TypeError(f'f should be a function, but got {type(samples)}')

  if steps is None:
    if isinstance(samples, int):
      samples = [samples, samples]
    elif not isinstance(samples, (tuple, list)):
      raise TypeError(f'samples should be of type int or 2-tuple, but got {type(samples)}')
    
    x = np.linspace(limits[0], limits[1], samples[0])
    y = np.linspace(limits[2], limits[3], samples[1])
  
  else:
    if isinstance(steps, float):
      steps = [steps, steps]
    elif not isinstance(steps, (tuple, list)):
      raise TypeError(f'steps should be of type int or 2-tuple, but got {type(steps)}')

    x = np.arange(limits[0], limits[1], steps[0])
    y = np.arange(limits[2], limits[3], steps[1])

  xx, yy = np.meshgrid(x, y)
  z = xx + 1j * yy
  w = f(z)

  hsv_map = _get_hsv(w, raygrid, eps, dynamic_eps)
  rgb_map = hsv_to_rgb(hsv_map)

  _, ax = plt.subplots()
  im = ax.imshow(rgb_map, interpolation='bilinear', extent=limits)
  ax.invert_yaxis()

  return im


def _get_hsv(z, raygrid, eps, dynamic_eps):
  arg = np.angle(z)
  mod = np.abs(z)
  cropped_mod = _crop_mod(mod)

  h = (arg / (2 * np.pi)) % 1

  s = cropped_mod.copy()
  s[cropped_mod <  0.5] = 1
  s[cropped_mod >= 0.5] = 1 - (s[cropped_mod >= 0.5] - 0.5) / 5

  v = cropped_mod.copy()
  v[cropped_mod <  0.5] = 0.9 - (0.5 - v[cropped_mod < 0.5]) / 2
  v[cropped_mod >= 0.5] = 0.9

  if raygrid:
    if dynamic_eps:
      dynamic_eps = np.arctan2(eps, mod)
    else:
      dynamic_eps = eps

    ray_idx = np.logical_or(h <= dynamic_eps, 1 - dynamic_eps <= h)
    s[ray_idx] = 0.3
    v[ray_idx] = 1

    for a in np.linspace(0, 1, 12, endpoint=False)[1:]:
      ray_idx = np.logical_and(a - dynamic_eps <= h, h <= a + dynamic_eps)
      s[ray_idx] = 0.3
      v[ray_idx] = 1

  return np.stack([h, s, v], axis=-1)


def _crop_mod(mod):
  maxMod = mod.max()
  cropped = mod.copy()
  while maxMod > 1:
    bigger = cropped > 1
    cropped[bigger] = (cropped[bigger] - 1) / 2
    maxMod = (maxMod - 1) / 2
  return cropped


im = domain_coloring(lambda z: (z**2 - 1) * (z - 2 - 1j)**2 / (z**2 + 2 + 2j),
                     steps=0.005)
plt.show()