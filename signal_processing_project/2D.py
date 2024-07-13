import numpy as np
import cv2
import matplotlib.pyplot as plt


images=[cv2.imread('image1.jpg'),
        cv2.imread('image2.jpg'),
        cv2.imread('image3.jpg'),
        cv2.imread('image4.jpg'),
        cv2.imread('image5.jpg')]

# converting to gray scale
images_gray=[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# implementing fft 
images_dft=[np.fft.fft2(img_gray) for img_gray in images_gray]
images_dft_shift=[np.fft.fftshift(img) for img in images_dft]

# magnitude spectrum
magnitude_spectrum=[np.log(1+np.abs(img)) for img in images_dft_shift]

# implementing inverse fft 
images_idft=[np.fft.ifft2(image) for image in images_dft]

# reconstructing the original image
reconstructed_images=[(np.real(img)) for img in images_idft]

fig,axs=plt.subplots(5,3,figsize=(10,20))
for i in range(len(images)):
    axs[i][0].imshow(images_gray[i],cmap='gray')
    axs[i][0].set_title('Original Image')
    
    axs[i][1].imshow(magnitude_spectrum[i],cmap='gray')
    axs[i][1].set_title('Magnitude Spectrum(log scale)')
    
    axs[i][2].imshow(reconstructed_images[i],cmap='gray')
    axs[i][2].set_title('Reconstructed image')
    
plt.subplots_adjust(hspace=0.5)
plt.show()
    
    



