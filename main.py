import cv2
import os

# capturing video
filename = "running1.mp4"
capture = cv2.VideoCapture(filename)

# video's frames per second
fps = float(capture.get(cv2.CAP_PROP_FPS))

# number of detections after line crossed
crossed_line = 0

# total elapsed time in seconds
elapsed_time = 0.00

# run started/ended flags
started = False
ended = False

# read the first frame of the input video
successful, frame1 = capture.read()

# resize height, width of frame
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)//2)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)//2)
frame1 = cv2.resize(frame1,(width,height))

# preview image on start of program
preview = None

# if the finish line has been set by the user (by drawaing line with mouse)
finish_line_set = False

# initial start and end points of finish line
startPoint = (-1, -1)
endPoint = (-1, -1)

# function for drawing finish line
def drawFinishLine(event, x, y, flags, param):
    global startPoint,frame1, preview, endPoint, finish_line_set
    if event == cv2.EVENT_LBUTTONDOWN:
        # button click point is the start point
        startPoint = (x,y)
        # preview image is first frame of input video with this start point
        preview = frame1.copy()
        cv2.line(preview, startPoint, (x,y), (0,255,0), 1)

    elif event == cv2.EVENT_MOUSEMOVE:
        # if we have set the start point, show the preview line while moving mouse
        if preview is not None:
            # preview image is first frame of input video with line from start point to current point of the mouse
            preview = frame1.copy()
            cv2.line(preview, startPoint, (x,y), (0,255,0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        # draw the line if we have a start and endpoint
        if preview is not None:
            # set preview to None so we can show our frame with the line
            preview = None
            # we have set the finish line
            endPoint = (x, y)
            finish_line_set = True
            # show frame with line
            cv2.line(frame1, startPoint, (x,y), (255,0,0), 1)

cv2.namedWindow("Automatic Sprint Timer")
cv2.setMouseCallback("Automatic Sprint Timer", drawFinishLine)

# run this until we set the finish line. 
while (not finish_line_set):
    # show this to user when setting finish line
    if preview is not None:
        print("setting finish line")
        cv2.imshow("Automatic Sprint Timer", preview)
    # finish line not set -- show first frame only
    else:
        cv2.imshow('Automatic Sprint Timer', frame1)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
      break;


# use frame1 and frame2 to get contours
def getContours(frame1, frame2):

    # find difference between two frames
    abs_diff = cv2.absdiff(frame1, frame2)
    
    # draw finish line on frame
    frame1 = cv2.line(frame1, startPoint, endPoint, (0, 0, 255), 2) 

    # convert frame to grayscale
    grayed = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)

    # smoothen frame
    blurred = cv2.GaussianBlur(grayed, (9, 9), 0)

    # binary image
    _, bin_image = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# Detect motion -- starting and ending points in the video
def detectMotion(ctrs):
    global crossed_line, started, ended
    # loop through contours to detect motion
    for contour in ctrs:

        x, y, w, h = cv2.boundingRect(contour)

        # detect start motion
        if cv2.contourArea(contour) > 15:
            # draw box around motion
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            started = True

        # detect end motion
        if x < startPoint[0] and cv2.contourArea(contour) > 300:
            print("crossed the line")
            crossed_line += 1
            if crossed_line > 5:
                ended = True
                break;


# output video
output_filename = "output_" + filename
video_writer = cv2.VideoWriter(os.path.join(output_filename), cv2.VideoWriter_fourcc(*'mp4v'), fps//2, (width, height)) 

# loop through input video
while successful:
    # write frame to output video and show frame on screen
    video_writer.write(frame1) 
    cv2.imshow("Automatic Sprint Timer", frame1)

    # read frame by frame
    # READING 2 FRAMES HERE...
    # we are doing WHILE successful, so it means we are only looking at HALF the frames
    successful, frame1 = capture.read()
    successful2, frame2 = capture.read()

    # if frame was read successfully
    if successful:

        # resize frames
        frame1 = cv2.resize(frame1,(width,height))
        frame2 = cv2.resize(frame2,(width,height))

        # get contours
        contours = getContours(frame1, frame2)

        # draw the stopwatch/timer on the video
        cv2.rectangle(frame1, (0,0), (250,40), (0, 0, 0), -1)
        cv2.putText(frame1, "time: " + str(round(elapsed_time, 2)),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

        # increment the time
        if started and not ended:
            elapsed_time+= 2*(1/fps)
        print("time:", elapsed_time)

        # call detectMotion() fxn to detect start and finish points in the input video
        detectMotion(contours)

    if cv2.waitKey(100) == 13:
        exit()

# Close everything
capture.release()
cv2.destroyAllWindows()
video_writer.release()