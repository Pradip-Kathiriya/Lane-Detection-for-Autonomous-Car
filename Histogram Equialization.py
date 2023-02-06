import cv2 as cv
import numpy as np

def main():
    
    # read each video frame from the given data set
    for a in range(25):

        image = cv.imread("data/"+ str(a)+".png", cv.COLOR_BGR2GRAY)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # initialise array to store the bins of the image
        # image will be divided into 256 parts
        bins = np.zeros((256))

        # count number of pixel in each bin
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                n = image[j][i]
                bins[n] = bins[n] + 1
            
        cumm_sum = 0 # variable to store the sum of number of pixel having intensity less than given pixel
        CFD = []     # varible to store value of cummulative frequency distribution function for each pixel

        # compute CFD for each pixel
        for i in range(len(bins)):
            cumm_sum = cumm_sum + bins[i]
            CFD.append(cumm_sum)

        CFD = CFD/cumm_sum

        # make deep copy of image
        img = np.copy(image)

        # perform histogram equailisation by replacing each pixel value in image by CFD*255
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                n = img[j][i]
                img[j][i] = CFD[n]*255
                
        # concatenate raw image and image after histogram equalisation
        final_image = np.concatenate((image, img), axis=0)
        
        cv.putText(final_image,'Before Histogram Equalisation',(10,25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0) , 2)
        cv.putText(final_image,'After Histogram Equalisation',(10,410),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0) , 2)

        final_image = cv.resize(final_image, (int(final_image.shape[1]*0.8), int(final_image.shape[0]*0.8)))

        # show image
        cv.imshow('Histogram Equalisation', final_image)
        
        # cv.imwrite("output_question1_partA.jpg", final_image)

        cv.waitKey(10)

if __name__ == "__main__":

    main()  
    

    
