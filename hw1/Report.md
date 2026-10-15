# Homework 1

**Name:** Yuyang Lu  

## 0. Environment

```text
Python: 3.11.17
NumPy: 2.4.6
Matplotlib: 3.11.2
OpenCV: 5.0.0
PyTorch: 2.11.0
```

Package versions were collected separately; `check_package_versions()` currently has its print calls commented out.

## 1. Image loading and analysis

```text
Image object type: <class 'numpy.ndarray'>
Pixel dtype: uint8
Image dimensions (height, width, channels): (321, 433, 3)
Image dtype is uint8: True
Pixel channel is uint8: True
```

![Original image](original_image.png)

## 2. Red channel

Output shape: `(321, 433, 3)`; dtype: `uint8`.

![Red channel](report_outputs/red_image.png)

## 3. Photographic negative

Output shape: `(321, 433, 3)`; dtype: `uint8`.

![Photographic negative](report_outputs/negative_image.png)

## 4. Swapped red and blue channels

Output shape: `(321, 433, 3)`; dtype: `uint8`.

![Swapped red and blue channels](report_outputs/swapped_image.png)

## 5. Foliage detection

Output shape: `(321, 433)`; dtype: `uint8`.

![Foliage detection](report_outputs/foliage_image.png)

## 6. Translation

Shift: 200 pixels right, 100 pixels down.

Output shape: `(321, 433, 3)`; dtype: `uint8`.

![Translation](report_outputs/shifted_image.png)

## 7. Clockwise 90-degree rotation

Output shape: `(433, 321, 3)`; dtype: `uint8`.

![Clockwise 90-degree rotation](report_outputs/rotated_image.png)

## 8. Similarity transform

Parameters: scale = 2.0; theta = 45.0 degrees clockwise; shift = [100.0, 100.0].

Output shape: `(321, 433, 3)`; dtype: `uint8`.

![Similarity transform](report_outputs/transformed_image.png)

## 9. Grayscale conversion

Output shape: `(321, 433)`; dtype: `uint8`.

![Grayscale conversion](report_outputs/gray_image.png)

## 10. Binary image moments

![Glasses outline](glasses_outline.png)

Moments use the stored pixel intensities: 255 for foreground and 0 for background.

Returned values:

```text
m00 = 3006960
m10 = 1432355910
m01 = 575000265
x_bar = 476.346845319
y_bar = 191.223117368
mu20 = 35161031847.7
mu02 = 15134711034.4
mu11 = 14713466709.7
```

Additional moments requested by the handout (computed separately for this report; not returned by the current function):

```text
m20 = 717459250950
m02 = 125088054195
m11 = 288613029000
mu10 = 4.47034835815e-08
mu01 = 1.49011611938e-08
```

## 11. Orientation, eccentricity, and fitted ellipse

```text
Orientation (degrees, clockwise from positive horizontal): 27.8814963047
Eccentricity: 0.91040788693
Ellipse image shape: (518, 774, 3)
Ellipse image dtype: uint8
```

![Glasses with fitted red ellipse](report_outputs/glasses_with_ellipse.png)
