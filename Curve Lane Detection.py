import cv2 as cv
import numpy as np

def homography(pts_src, pts_dst, src, dst):
    
    """
    homography function will warp region of interest from source image to destination image
    :Param pts_src: corrospandace points on the source image
    :Param pts_dst: corrospandace points on the destination image
    :Return: destination image after warping region og interest from source image
    
    """
    # compute homography matrix
    h,_ = cv.findHomography(pts_src,pts_dst)
    # perform warping from source to destination
    warped_image = cv.warpPerspective(src, h, (dst.shape[1],dst.shape[0]))
    return warped_image

def binaryToRGB(image):
    
    """
    binaryToRGB function convert input binary image into binary image with three color channel
    this is required when concatenating RGB and binary image
    :Param: binary image
    :Return: binary image with three color channel 
    
    """
    
    RGB_image = np.zeros((image.shape[0], image.shape[1],3), dtype = 'uint8')
    RGB_image[:,:,0] = image
    RGB_image[:,:,1] = image
    RGB_image[:,:,2] = image
    
    return RGB_image

def yellowLaneDetection(warped_image, warped_image_thresh_withLane,blank_image):
    
    """
    yellowLaneDetection function detect yellow line in the give image, draw it on the image and compute the radius of curvature
    :Param warped_image: bird eye view of the lane
    :Param warped_image_thresh_withLane: image to draw the detection points and curve on
    :Param blank_image: image to draw the detection points and curve on
    :Return yellow_lane_radius: radius of yellow lane
    :Return yellow_lane_points: list of the detected points of yellow lane
    :Return curve_coeff: co-effecient of yellow lane equation
    
    
    """
    
    # convert the image into hsv color space and apply mask to detect only yellow lane
    hsv = cv.cvtColor(warped_image, cv.COLOR_BGR2HSV)
    lower_hue = np.array([10,100,160])  
    upper_hue = np.array([35,255,255])
    masked = cv.inRange(hsv, lower_hue, upper_hue)
    
    # find the x and y cooridnate of yellow lane
    yellow_points = np.where(masked != 0)
    # find the equation of curve for yellow lane
    curve_coeff = np.polyfit(yellow_points[0][:],yellow_points[1][:],2)
    # find the extrapolate points of curve using above equation
    Y2_points = np.linspace(0, warped_image.shape[0], 40)
    X2_points = np.polyval(curve_coeff, Y2_points)
    
    # find the radius of yellow lane
    yellow_lane_radius = np.abs(float((1 + (2*curve_coeff[0]*Y2_points[20] + curve_coeff[1])**2)**(3.0/2.0))/float(2*curve_coeff[0]))

    # plot the curve on the blank_image and warped_image_thresh_withLane image
    yellow_lane_points = (np.asarray([X2_points, Y2_points]).T).astype(np.int32)   
    cv.polylines(blank_image, [yellow_lane_points], False, (0,0,255), thickness = 10)
    cv.polylines(warped_image_thresh_withLane, [yellow_lane_points], False, (0,0,255), thickness = 2)
    
    # draw the detected yellow points on the warped_image_thresh_withLane image
    for points in yellow_lane_points:
        points = tuple(points)
        if points[1] == 600:
            continue
        if masked[points[1],points[0]] == 0:
            continue
        cv.circle(warped_image_thresh_withLane,points, 10, (0,255,255),thickness = 2)
        
    return yellow_lane_radius, yellow_lane_points, curve_coeff

def whiteLaneDetection(warped_image, warped_image_thresh_withLane,blank_image):
    
    """
    whiteLaneDetection function detect white line in the give image, draw it on the image and compute the radius of curvature
    :Param warped_image: bird eye view of the lane
    :Param warped_image_thresh_withLane: image to draw the detection points and curve on
    :Param blank_image: image to draw the detection points and curve on
    :Return white_lane_radius: radius of white lane
    :Return white_lane_points: list of the detected points of white lane
    :Return curve_coeff: co-effecient of white lane equation
    
    
    """

    white_lane = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(white_lane, 225, 255, cv.THRESH_BINARY)
    # find the x and y cooridnate of white lane
    white_points = np.where(threshold != 0)
    # find the equation of curve for white lane
    curve_coeff = np.polyfit(white_points[0][:],white_points[1][:],2)
    # find the extrapolate points of curve using above equation
    Y1_points = np.linspace(0, warped_image.shape[0], 40)
    X1_points = np.polyval(curve_coeff, Y1_points)
    
    # find the radius of white lane
    white_lane_radius = np.abs(float((1 + (2*curve_coeff[0]*Y1_points[20] + curve_coeff[1])**2)**(3.0/2.0))/float(2*curve_coeff[0]))
    
    # plot the curve on the blank_image and warped_image_thresh_withLane image
    white_lane_points = (np.asarray([X1_points, Y1_points]).T).astype(np.int32)   
    cv.polylines(blank_image, [white_lane_points], False, (0,0,255), thickness = 10)
    cv.polylines(warped_image_thresh_withLane, [white_lane_points], False, (0,0,255), thickness = 2)

    # draw the detected white points on the warped_image_thresh_withLane image
    for points in white_lane_points:
        points = tuple(points)
        if points[1] == 600:
            continue
        if threshold[points[1],points[0]] == 0:
            continue
        cv.circle(warped_image_thresh_withLane,points, 10, (0,255,255),thickness = 2)
        
    return white_lane_radius, white_lane_points, curve_coeff


def main():
    
    while True:
        
        isTrue, frame = capture.read()
        if not isTrue:
            break
        # frame = cv.flip(frame,1)
        
        original_image = frame.copy()
        
        # compute the threshold of the original image to show it in the final output as frame 2
        gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        _,thresh_image = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
        original_image_thresh = binaryToRGB(thresh_image)

        # warp the region of interest of lane using homography to take the bird eye view
        blank = np.zeros((600,400,3), dtype = 'uint8')
        pts_src = np.array([[150,680],[540,440],[735,440],[1230,680]])
        pts_dst = np.array([[0,blank.shape[0]],[0,0],[blank.shape[1],0],[blank.shape[1],blank.shape[0]]])
        warped_image = homography(pts_src, pts_dst, frame, blank)

        # convert the warp image into binary image with three color channel to show it in the final output as frame 3
        warped_image_gray = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)
        _,warped_image_thresh = cv.threshold(warped_image_gray, 180, 255, cv.THRESH_BINARY)
        warped_image_thresh = binaryToRGB(warped_image_thresh)
        
        # warped_image_thresh_withLane will be use for drawing dected line and points on it
        # the result will be shown in output as frame 4
        warped_image_thresh_withLane = warped_image_thresh.copy()
        
        # generate blank image to draw the lane on it
        # it is necessary to draw the lane on the blank image to superimpose it on the original image
        blank_image = np.zeros_like(warped_image)
        
        ############################
        ### white lane detection ###
        ############################
        
        white_lane_radius, white_lane_points, curve_coeff = whiteLaneDetection(warped_image, warped_image_thresh_withLane,blank_image)
        
        #############################
        ### yellow lane detection ###
        #############################
        
        yellow_lane_radius, yellow_lane_points, curve_coeff = yellowLaneDetection(warped_image, warped_image_thresh_withLane,blank_image)

        #####################################
        ### calculate radius of curvature ###
        #####################################
        
        radius = round((white_lane_radius + yellow_lane_radius)/2.0,2)

        # fill the lane with the red color using fillpoly function
        yellow_lane_points = np.flipud(yellow_lane_points)
        points = np.concatenate((white_lane_points, yellow_lane_points))
        cv.fillPoly(blank_image, [points], color=[0,0,255])
        
        # temporary image of inverse homography
        # this image is a blank image with detected lane plot on it after filling the lane
        temp_image = homography(pts_dst, pts_src, blank_image, original_image)
        
        # add the orignal image with the detected lane image to get the final output
        final_image = cv.add(original_image,temp_image)
        
        # resize the final image to show it in final output
        final_image = cv.resize(final_image,(int(original_image.shape[1]*0.7),int(original_image.shape[0]*0.25)+int(original_image.shape[0]*0.45)))
        
        ########################################
        ### check the direction of curvature ###
        ########################################
        if curve_coeff[0] < 0:
            cv.putText(final_image,'Go Left',(50,50),cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0,255) , 2)
        elif curve_coeff[0] > 0:
            cv.putText(final_image,'Go Right',(50,50),cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0,255) , 2)
        else:
            cv.putText(final_image,'Go Straight',(50,50),cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0,255) , 2)
        
        
        
        ###########################
        ### code to show output ###
        ###########################
        
        # concatenate orignal image(frame 1 in the output) with its threshold(frame 2 in the output) and resize the resultant image
        video1 = np.concatenate((original_image, original_image_thresh), axis = 1)
        video1 = cv.resize(video1, (int(original_image.shape[1]*0.25),int(original_image.shape[0]*0.25)))

        # write frame number 1 and 2 on the above two images
        cv.putText(video1,'(1)',(30,30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0,0) , 2)
        cv.putText(video1,'(2)',(180,30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0,0) , 2)
        
        # concatenate warped image threshold(frame 3 in the output) and image showing detected points and curve(frame 4 in the output) on the warped image
        # resize the resultant image
        video2 = np.concatenate((warped_image_thresh, warped_image_thresh_withLane), axis = 1)
        video2 = cv.resize(video2, (int(original_image.shape[1]*0.25),int(original_image.shape[0]*0.45)))
        
        # write frame number 3 and 4 on the above two image
        cv.putText(video2,'(3)',(30,30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0,0) , 2)
        cv.putText(video2,'(4)',(180,30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0,0) , 2)

        # concatenate above two pair
        video3 = np.concatenate((video1, video2), axis = 0)
        
        # concatenate result of above operation with final output image of the code
        video4 = np.concatenate((final_image, video3), axis = 1)
        
        # generate white image to concatenate at the bottom of the output
        video5 = np.zeros([int(original_image.shape[0]*0.10),int(original_image.shape[1]*0.70)+int(original_image.shape[1]*0.25),3],dtype=np.uint8)
        video5.fill(255)
        
        # add detail of the output on the white image
        cv.putText(video5,'(1): Original image, (2): Detected Yellow and White lane, (3): Warped image, (4): Detected points and curve fitting',(10,25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0) , 1)
        cv.putText(video5,'Average Radius:'+ " " + str(radius) + " " + "m",(10,55),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0) , 1)
        
        # concatenate white image with the output to generate final output image
        video6 = np.concatenate((video4, video5), axis = 0)
        cv.imshow('video', video6)
        
        # generate video
        video.write(video6)
        
        if cv.waitKey(70) & 0xFF == ord('d'):
            break
    
    
if __name__=="__main__":
    
    # read the vide 
    capture = cv.VideoCapture('data/challenge.mp4')

    # generate output video
    video  = cv.VideoWriter('Question4_output.avi', cv.VideoWriter_fourcc(*'XVID'),10,(1216,576))
    
    main() 
    
    video.release()
    
