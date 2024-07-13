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

# implementing fft on rows
images_dft_rows=[np.fft.fft(img_gray,axis=1) for img_gray in images_gray]
images_dft_rows_shift=[np.fft.fftshift(img) for img in images_dft_rows]
# implementing fft on columns
images_dft_columns=[np.fft.fft(img_gray,axis=0) for img_gray in images_gray]
images_dft_columns_shift=[np.fft.fftshift(img) for img in images_dft_columns]


# implementing inverse fft on rows
images_idft_rows=[np.fft.ifft(image,axis=1) for image in images_dft_rows]
# implementing inverse fft on columns
images_idft_columns=[np.fft.ifft(image,axis=0) for image in images_dft_columns]

# reconstructing the original image
reconstructed_images=[(np.real(image_rows+image_columns)) for image_rows,image_columns in zip(images_idft_rows,images_idft_columns)]

fig,axs=plt.subplots(5,4,figsize=(10,20))
for i in range(len(images)):
    axs[i][0].imshow(images_gray[i],cmap='gray')
    axs[i][0].set_title('Original Image')
    
    axs[i][1].imshow(np.log(1+np.abs(images_dft_rows_shift[i])),cmap='gray')
    axs[i][1].set_title('DFT on rows')
    
    axs[i][2].imshow(np.log(1+np.abs(images_dft_columns_shift[i])),cmap='gray')
    axs[i][2].set_title('DFT on columns')
    
    axs[i][3].imshow(reconstructed_images[i],cmap='gray')
    axs[i][3].set_title('Reconstructed image')
    
plt.subplots_adjust(wspace=0.5)
plt.show()
    
    



