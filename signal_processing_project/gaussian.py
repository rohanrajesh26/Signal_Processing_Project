import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_filter(shape, sigma):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h /= h.sum()
    return h

# Load the image
img = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply 2D DFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Create Gaussian filter
rows, cols = img.shape
gaussian = gaussian_filter((rows, cols), sigma=10)

# Apply the Gaussian filter in the frequency domain
fshift_filtered = fshift * gaussian

# Apply inverse DFT to reconstruct the image
f_ishift_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_ishift_filtered)
img_filtered = np.abs(img_filtered)

# Display the original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(img_filtered, cmap='gray')
plt.title('Filtered Image')

plt.show()
