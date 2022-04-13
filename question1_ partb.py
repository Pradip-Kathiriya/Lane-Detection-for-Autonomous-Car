import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# path = r'/home/pradip/Desktop/673/Project/project 2/pradip_project2/prob1data'
def main():
    
    # read each video frame from the given data set
    for a in range(25):

        image = cv.imread("data/"+ str(a)+".png", 0)
        
        img = image.copy()

        # divide the image into horizontal slice
        vert_slice = np.array_split(image, 8, axis = 0)

        # divide each horizontal slices into verticale slice
        # store all the block into list
        blocks = [np.array_split(slice, 8, axis = 1) for slice in vert_slice]

        # perform histogram equilisation in each blocks
        for i in range(len(blocks)):
            for j in range(len(blocks[i])):

                sub_image = blocks[i][j]
                
                # initialise array to store the bins of the image
                # image will be divided into 256 parts
                bins = np.zeros((256))

                # count number of pixel in each bin
                for k in range(sub_image.shape[1]):
                    for l in range(sub_image.shape[0]):
                        n = sub_image[l][k]
                        bins[n] = bins[n] + 1
                
                cumm_sum = 0 # variable to store the sum of number of pixel having intensity less than given pixel
                CFD = []     # varible to store value of cummulative frequency distribution function for each pixel

                # compute CFD for each pixel
                for m in range(len(bins)):
                    cumm_sum = cumm_sum + bins[m]
                    CFD.append(cumm_sum)

                CFD = CFD/cumm_sum

                # perform histogram equailisation by replacing each pixel value in image by CFD*255
                for n in range(sub_image.shape[1]):
                    for o in range(sub_image.shape[0]):
                        sub_image[o][n] = CFD[sub_image[o][n]]*255

        # assemble all the blocks into image
        final_image = np.block(blocks)
        
        final_image = np.concatenate((img, final_image), axis=0)
        
        cv.putText(final_image,'Before Histogram Equalisation',(10,25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0) , 2)
        cv.putText(final_image,'After Histogram Equalisation',(10,410),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0) , 2)
        
        final_image = cv.resize(final_image, (int(final_image.shape[1]*0.8), int(final_image.shape[0]*0.8)))

        # show image
        cv.imshow('Adaptive Histogram Equalisation', final_image)

        cv.waitKey(10)

if __name__ == "__main__":
    main()

