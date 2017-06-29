import cv2
import utils

file_to_split = r"C:\videos\a\output3.avi"

cap = cv2.VideoCapture(file_to_split)

if not cap.isOpened():
	raise Exception("Failed to open file.")

num_of_splits = 4

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

split_width_size = int(width / num_of_splits)
split_height_size = int(height / num_of_splits)
print(split_width_size)
print(split_height_size)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = utils.get_fps(file_to_split)
writer = cv2.VideoWriter("output.avi", fourcc, fps, (int(width), int(height)))
frames_counter = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    frames_counter += 1

    if ret==True:
        # relevant_frame = frame[i*split_height_size:(i+1)*split_height_size,j*split_width_size:(j+1)*split_width_size]       

        # if i*num_of_splits + j == 6:
        if 0 < frames_counter < 30 * fps:
    		cv2.rectangle(frame,(1*split_width_size,2*split_height_size),(2*split_width_size,3*split_height_size),(0,0,255),5)
        if 20 * fps < frames_counter < 30 * fps:
    		cv2.rectangle(frame,(0*split_width_size,2*split_height_size),(1*split_width_size,3*split_height_size),(0,0,255),5)
        if 0 * fps < frames_counter < 10 * fps:
    		cv2.rectangle(frame,(2*split_width_size,2*split_height_size),(3*split_width_size,3*split_height_size),(0,0,255),5)


    	# cv2.rectangle(frame,(0,0),(400,400),(0,255,0),5)
    	writer.write(frame)
    	cv2.imshow('sa',frame)
    	k = cv2.waitKey(30) & 0xff
    	if k == 'q':
        	break
    else:
        break
writer.release()