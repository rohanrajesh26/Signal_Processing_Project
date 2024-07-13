import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the Discrete Fourier Transform
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Compute the magnitude spectrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

# Create a mask for high-pass filtering
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols, 2), np.uint8)
r = 90
center = (crow, ccol)
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r ** 2
mask[mask_area] = 0

# Apply the mask and compute the inverse DFT
fshift = dft_shift * mask
fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Adjust the brightness of the image
max_val = np.max(img_back)
img_back = max_val - img_back

# Display the images
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')

plt.subplot(2, 2, 3)
plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask')

plt.subplot(2, 2, 4)
plt.imshow(img_back, cmap='gray')
plt.title('After Inverse FFT')

plt.show()

