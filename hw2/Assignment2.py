# -*- coding: utf-8 -*-
"""EEP 596 HW2
"""

import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

class ComputerVisionAssignment():
  def __init__(self) -> None:
    self.ant_img = cv2.imread('ant_outline.png', cv2.IMREAD_COLOR)
    self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def floodfill(self, seed = (0, 0)):

    # Define the fill color (red)
    fill_color =  (0, 0, 255) # (B, G, R)
    # Create a copy of the input image to keep the original image unchanged
    output_image = self.ant_img.copy()
    # Define a stack for floodfill
    x, y = seed  # (x, y) corresponds to image[y, x]
    h, w = output_image.shape[:2]
    if not (0 <= x < w and 0 <= y < h) :
      raise ValueError('Seed must be inside the image.')
    target_color = output_image[y, x].copy()
    if np.array_equal(target_color, fill_color) :
      return output_image
    stack = [(x, y)]
    output_image[y, x] = fill_color

    while stack:
      x, y = stack.pop()
      for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
          if dx == 0 and dy == 0 :
            continue
          tx, ty = x + dx, y + dy
          if (0 <= tx < w and 0 <= ty < h
              and np.array_equal(output_image[ty, tx], target_color)):
            # Mark when pushed to avoid adding a pixel more than once.
            output_image[ty, tx] = fill_color
            stack.append((tx, ty))

    # PNG preserves exact colors; JPEG compression can disrupt color matching.
    cv2.imwrite('ant_floodfill.png', output_image)
    return output_image

  def convolve_1d(self, image, kernel, axis):
    """Apply three taps without flipping: axis=0 vertical, axis=1 horizontal."""
    image = image.astype(np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.shape != (3,) or axis not in (0, 1):
      raise ValueError('Use a length-3 kernel and axis 0 or 1.')
    h, w = image.shape
    padding = ((1, 1), (0, 0)) if axis == 0 else ((0, 0), (1, 1))
    padded = np.pad(image, padding, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    for y in range(h):
      for x in range(w):
        # Apply in written order, as required for the derivative kernel.
        if axis == 0:
          output[y, x] = (kernel[0] * padded[y, x]
                        + kernel[1] * padded[y + 1, x]
                        + kernel[2] * padded[y + 2, x])
        else:
          output[y, x] = (kernel[0] * padded[y, x]
                        + kernel[1] * padded[y, x + 1]
                        + kernel[2] * padded[y, x + 2])
    return output
    
  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """
    kernel = 0.25 * np.array([1, 2, 1]) # 1D Gaussian kernel
    image = self.cat_eye.astype(np.float64)

    self.blurred_images = []
    for i in range(5):
        # Apply convolution
        vertical = self.convolve_1d(image, kernel, axis=0)
        image = self.convolve_1d(vertical, kernel, axis=1)
        
        # Store the blurred image
        self.blurred_images.append(image)
        
        cv2.imwrite(f'gaussian blur {i}.jpg', image)
    return self.blurred_images

  def gaussian_derivative_vertical(self):
    # Run gaussian_blur first to provide the five smoothed inputs.
    smoothing = 0.25 * np.array([1, 2, 1])
    differentiating = 0.5 * np.array([1, 0, -1])
    self.vDerive_images = []

    for image in self.blurred_images:
      horizontal = self.convolve_1d(image, smoothing, axis=1)
      derivative = self.convolve_1d(horizontal, differentiating, axis=0)
      output = np.clip(2 * derivative + 127, 0, 255).astype(np.uint8)
      self.vDerive_images.append(output)

    return self.vDerive_images

  def gaussian_derivative_horizontal(self):
    #Define kernels

    # Store images after computing horizontal derivative
    smoothing = 0.25 * np.array([1, 2, 1])
    differentiating = 0.5 * np.array([1, 0, -1])
    self.hDerive_images = []

    for image in self.blurred_images:
      vertical = self.convolve_1d(image, smoothing, axis=0)
      derivative = self.convolve_1d(vertical, differentiating, axis=1)
      output = np.clip(2 * derivative + 127, 0, 255).astype(np.uint8)
      self.hDerive_images.append(output)

    return self.hDerive_images

  def gradient_magnitute(self):
    # Store the computed gradient magnitute
    self.gdMagnitute_images =[]
    for i, (vimg, himg) in enumerate(zip(self.vDerive_images, self.hDerive_images)):
      # Convert before subtraction to avoid uint8 wraparound.
      gx = (himg.astype(np.float64) - 127) / 2
      gy = (vimg.astype(np.float64) - 127) / 2
      image = np.clip(4 * (np.abs(gx) + np.abs(gy)), 0, 255).astype(np.uint8)
      self.gdMagnitute_images.append(image)
      cv2.imwrite(f'gradient {i}.jpg', image)
    return self.gdMagnitute_images
    
  def scipy_convolve(self):
    # Define the 2D smoothing kernel
    smoothing = 0.25 * np.array([[1, 2, 1]])
    differentiating = 0.5 * np.array([[1], [0], [-1]])

    # Store outputs
    self.scipy_smooth = []

    for i in range(5):
      # Perform convolution
      horizontal = scipy.signal.convolve2d(
        self.blurred_images[i], smoothing, mode = 'same', boundary = 'fill', fillvalue = 0
      )
      derivative = scipy.signal.convolve2d(
        horizontal, differentiating, mode = 'same', boundary = 'fill', fillvalue = 0
      )
      output = np.clip(2 * derivative + 127, 0, 255).astype(np.uint8)
      self.scipy_smooth.append(output)
      cv2.imwrite(f'scipy smooth {i}.jpg', output)

    return self.scipy_smooth

  def box_filter(self, num_repetitions):
    # Define box filter
    box_filter = np.array([1, 1, 1], dtype=np.int64)
    out = box_filter.copy()

    for _ in range(num_repetitions):
      # Perform 1D conlve
      out = np.convolve(out, box_filter, mode = 'full')

    return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # # Task 1 floodfill
    floodfill_img = ass.floodfill((150, 160))
    cv2.imwrite('ant_floodfill.png', floodfill_img)

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # Task 3 Convolution for differentiation along the vertical direction
    vertical_derivative = ass.gaussian_derivative_vertical()

    # Task 4 Differentiation along another direction along the horizontal direction
    horizontal_derivative = ass.gaussian_derivative_horizontal()

    # Task 5 Gradient magnitude.
    Gradient_magnitude = ass.gradient_magnitute()

    # Task 6 Built-in convolution
    scipy_convolve = ass.scipy_convolve()

    # Task 7 Repeated box filtering
    box_filter = ass.box_filter(5)
    # print(box_filter)
