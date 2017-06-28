import numpy as np
import cv2

class FramesCompareExtractor():
	def __init__(self):
		pass

	def get_features_names(self):
		return "frames_compare"

	def extract(self, file_path):
		features = []
		cap = cv2.VideoCapture(file_path)

		ret, prev_frame = cap.read()
		ret, prev_frame = cap.read()
		if not ret:
			return

		prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
		frames_counter = 0
		while cap.isOpened():
			ret, next_frame = cap.read()

			if not ret:
				break
			next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
			if frames_counter % 100 == 0:
				(score, diff) = compare_ssim(prev_frame, next_frame, full=True)
				diff = (diff * 255).astype("uint8")
				# cv2.imshow("Diff", diff)

			prev_frame = next_frame
			frames_counter += 1
			
		cap.release()
		cv2.destroyAllWindows()
		
		return [] 


class MotionDetectionExtractor():
	def __dense_optical_flow(self, VIDEO_PATH, threshold_1=1.5, threshold_2=50):
		cap = cv2.VideoCapture(VIDEO_PATH)
		ret, frame1 = cap.read()
		prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
		grayscale = np.zeros_like(prvs)
		grayscale[...] = 0
		moved_frames = 0
		total_frames = 0

		while cap.isOpened():
			ret, frame2 = cap.read()
			if not ret:
				break
			total_frames += 1

			next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

			num_of_moving_pixels = len(mag[mag > threshold_1])
			if num_of_moving_pixels > threshold_2:
				moved_frames +=1
			
			prvs = next
		cap.release()
		cv2.destroyAllWindows()
		
		print([float(moved_frames) / float(total_frames)])

		return [float(moved_frames) / float(total_frames)]

	def get_features_names(self):
		return "is_motion"

	def extract(self, file_path):
		return self.__dense_optical_flow(file_path)


if __name__ == "__main__":
	extractor = MotionDetectionExtractor()
	for i in xrange(16):
		print(extractor.extract(r"C:\videos\1\4\4_0_" + str(i) + ".avi"))
	# feature_extractor = FramesCompareExtractor()
	# feature_extractor.extract(r"C:\videos\1.avi")
