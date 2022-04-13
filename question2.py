import cv2 as cv
import numpy as np

def preProcessing(image):
    
    """
    preProcessing function pre-process video frame to remove noise and unnecessary detail from the video
    :Param image: raw video frame
    :Return: processed image
    
    """
    
    # convert the image into gray scale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    
    # define a mask to detect only those area which is in front of car
    mask  = np.zeros_like(gray)
    vertices = np.array([[0,gray.shape[0]],[410,330],[530,330],[gray.shape[1],gray.shape[0]]])
    cv.fillPoly(mask, [vertices], 255) 
    masked = np.bitwise_and(gray, mask)
    
    # apply threshold to convert image into binary
    isTrue,  threshold = cv.threshold(masked, 200, 255, cv.THRESH_BINARY) 
    
    #detect edges in the image
    canny = cv.Canny(threshold, 40, 170)   
    
    return canny

def detectLane(image, frame):
    
    """
    detectLane function detect the solid and dash lane and plot it on the original video frame
    :Param image: pre-processed video frame
    :Param frame: original video frame
    :Return: original video frame with detected lane plotted on it
    
    """
    
    ############################
    ### solid lane detection ###
    ############################
    
    # detect line in the pre-processed image
    # since the minLineLength is very high, it will detect only those line which is corrosponding to solid lane
    linesP = cv.HoughLinesP(image, 1, np.pi/180, threshold = 20, minLineLength = 150, maxLineGap = 5)  
    
    # find the mean of all start and end points in the detected lines
    xstart_mean = int(np.mean(linesP[:,:,0]))                              
    ystart_mean = int(np.mean(linesP[:,:,1]))  
    xend_mean = int(np.mean(linesP[:,:,2]))             
    yend_mean = int(np.mean(linesP[:,:,3])) 
    
    # find the slope and intercept using mean of start and end points
    slope = float(yend_mean - ystart_mean)/float(xend_mean - xstart_mean)
    intercept = ystart_mean - slope * xstart_mean
    
    # using the slope and intercept detected above,find the end poits of the detected lane
    y1 = 330
    x1 = int((y1-intercept)/slope)
    y2 = frame.shape[0]                         
    x2 = int((y2-intercept)/slope)
    
    # draw the detected--solid-- lane on the video frame
    cv.line(frame, (x1,y1),(x2,y2), (0,255,0),10)     
    
    
    #############################
    ### dashed lane detection ###
    #############################                                                                                    

    dashed = []     # list to store the start and end points of lines corrosponding to dashed lane
    
    # find all the possible line in the masked image
    linesP = cv.HoughLinesP(image, 1, np.pi/180, threshold = 10, minLineLength = 20, maxLineGap = 30) 
    
    # if the solid lene has a positive slope, then dashed lane will have negative slope and vice versa
    # append the line corrospoding to dashed line in the dashed list                                     
    if slope > 0:
        if linesP is not None:                                                             
            for i in range(0, len(linesP)):                                                                               
                l = linesP[i][0]
                m = float(l[3]-l[1])/float(l[2]-l[0])    
                if m < 0:
                    dashed.append([l[0],l[1],l[2],l[3]])
                                    
    if slope < 0: 
        if linesP is not None:                                 
            for i in range(0, len(linesP)):                                                                               
                l = linesP[i][0]
                m = float(l[3]-l[1])/float(l[2]-l[0])    
                if m > 0:
                    dashed.append([l[0],l[1],l[2],l[3]])

    dashed = np.array(dashed)
    
    # find the slope and intercept using mean of start and end points
    xstart_mean = int(np.mean(dashed[:,0]))
    ystart_mean = int(np.mean(dashed[:,1]))
    xend_mean = int(np.mean(dashed[:,2]))
    yend_mean = int(np.mean(dashed[:,3]))
    
    slope = float(yend_mean - ystart_mean)/float(xend_mean - xstart_mean)
    intercept = ystart_mean - slope * xstart_mean
    
    # using the slope and intercept detected above,find the end poits of the detected lane
    y1 = 330
    x1 = int((y1-intercept)/slope)
    y2 = frame.shape[0]
    x2 = int((y2-intercept)/slope)
 
    # draw the detected--dashed-- lane on the video frame
    cv.line(frame, (x1,y1),(x2,y2), (0,0,255),10)
    
def main():
    
    capture = cv.VideoCapture("data/whiteline.mp4", 0)

    while True:
    
        isTrue, frame = capture.read()                                          
        if not isTrue:                                                           
            break
        
        image =  preProcessing(frame)               
        
        detectLane(image, frame)
        
        # generate video
        video.write(frame)    
    
        cv.imshow('Video', frame)
        if cv.waitKey(60) & 0xFF==ord('d'):
            break  
               
if __name__=="__main__":
    
    # generate output video
    video  = cv.VideoWriter('Question3_output.avi', cv.VideoWriter_fourcc(*'XVID'),10,(960,540))
    
    main()  
    
    video.release()      
