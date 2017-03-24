#importing some useful packages
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# %matplotlib inline

# the top and the bottom vertical thresholds of region of interest
topY = 330
bottomY = 540
topLeftX = 440
topRightX = 540
bottomLeftX = 50
bottomRightX = 945
topCenterX = (topLeftX + topRightX)/2


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_region_of_interest(img, color=[0, 0, 255], thickness=2):
    cv2.line(img, (bottomLeftX, bottomY), (topLeftX, topY), color, thickness)
    cv2.line(img, (topLeftX, topY), (topRightX, topY), color, thickness)
    cv2.line(img, (topRightX, topY), (bottomRightX, bottomY), color, thickness)
    cv2.line(img, (bottomRightX, bottomY), (bottomLeftX, bottomY), color, thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    leftLineTopSum = 0
    leftLineBottomSum = 0
    leftLinesCount = 0

    rightLineTopSum = 0
    rightLineBottomSum = 0
    rightLinesCount = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            # figuring out linear function for line above
            plft = np.polyfit((x1,x2),(y1,y2),1)

            # distinguishing line by slope koef. of linear function
            if plft[0] < 0:
                if topLeftX <= (topY-plft[1])/plft[0] <= topCenterX:
                    leftLinesCount = leftLinesCount + 1

                    # extrapolating line to the top and the bottom lines of the region of interest
                    leftLineBottomSum = leftLineBottomSum + (bottomY-plft[1])/plft[0]
                    leftLineTopSum = leftLineTopSum + (topY-plft[1])/plft[0]
            elif plft[0] > 0:
                if topCenterX <= (topY-plft[1])/plft[0] <= topRightX:
                    rightLinesCount = rightLinesCount + 1

                    # extrapolating line to the top and the bottom lines of the region of interest
                    rightLineBottomSum = rightLineBottomSum + (bottomY-plft[1])/plft[0]
                    rightLineTopSum = rightLineTopSum + (topY-plft[1])/plft[0]
            else:
                return

    # calculating the average of interesction points (intersection of the extrapolated line and the top or the bottom of region)
    leftTopX = int(leftLineTopSum / leftLinesCount) if leftLinesCount > 0 else 0
    leftTopY = topY
    leftBottomX = int(leftLineBottomSum / leftLinesCount) if leftLinesCount > 0 else 0
    leftBottomY = bottomY

    rightTopX = int(rightLineTopSum / rightLinesCount) if rightLinesCount > 0 else 0
    rightTopY = topY
    rightBottomX = int(rightLineBottomSum / rightLinesCount) if rightLinesCount > 0 else 0
    rightBottomY = bottomY

    cv2.line(img, (leftBottomX, leftBottomY), (leftTopX, leftTopY), color, thickness)
    cv2.line(img, (rightBottomX, rightBottomY), (rightTopX, rightTopY), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, a=0.8, b=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * a + img * b + l
    NOTE: initial_img and img must be the same shape!
    """

    return cv2.addWeighted(initial_img, a, img, b, l)


# import os
# os.listdir("test_images/")
def pipeline(initial_image):

    image = np.copy(initial_image)

    # Read in and grayscale the image
    gray = grayscale(image)


    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)


    # Define our parameters for Canny and apply
    low_threshold = 80
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Get an image masked by polygon
    imshape = image.shape

    vertices = np.array([[(bottomLeftX,bottomY),(topLeftX, topY), (topRightX, topY), (bottomRightX,bottomY)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 25     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 8 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the region of interest
    # draw_region_of_interest(line_image)

    # Draw the lines on the edge image
    result_image = weighted_img(line_image, initial_image)

    return result_image

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = pipeline(image)
    return result

import os

in_dir = "test_images"
out_dir = "test_images_output"
#check id out dir exists
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file in os.listdir(in_dir):
    # print "test_images/" + file
    filePath = os.path.join(in_dir, file)# "test_images/" + file
    outfilePath = os.path.join(out_dir, file)# "test_images_output/" + file

    initial_image = mpimg.imread(filePath)
    output_image = pipeline(initial_image)
    mpimg.imsave(outfilePath, output_image)

# # video_file = 'solidYellowLeft.mp4'
# video_file = 'solidWhiteRight.mp4'
# video_in_dir = 'test_videos'
# video_out_dir = 'test_videos_output'
# if not os.path.exists(video_out_dir):
#     os.makedirs(video_out_dir)

# from moviepy.editor import VideoFileClip
# from IPython.display import HTML

# white_output =  os.path.join(video_out_dir, video_file)
# clip1 = VideoFileClip(os.path.join(video_in_dir, video_file))
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

print("End")