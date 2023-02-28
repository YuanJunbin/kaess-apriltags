import cv2
import glob
import os

# Input directory to your images here.
all_image_directory = "/home/siheng/Documents/Calibration/ICRA2023/images/"
results_directory = "/home/siheng/Documents/Calibration/ICRA2023/results"

# Setting up ArUco detector
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.markerBorderBits = 2
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParams.cornerRefinementWinSize = 10
arucoParams.adaptiveThreshWinSizeStep = 2

# Uses ArUco detector to count corners. Includes flag for visualization.
def count_corners(image, viz=False):
	total_corners = 0
	(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
		parameters=arucoParams)

	if viz:
		# verify *at least* one ArUco marker was detected
		viz_corners = corners
		if len(viz_corners) > 0:
			# flatten the ArUco IDs list
			ids = ids.flatten()
			# loop over the detected ArUCo corners
			for (markerCorner, markerID) in zip(viz_corners, ids):
				# extract the marker corners (which are always returned in
				# top-left, top-right, bottom-right, and bottom-left order)
				viz_corners = markerCorner.reshape((4, 2))
				(topLeft, topRight, bottomRight, bottomLeft) = viz_corners
				# convert each of the (x, y)-coordinate pairs to integers
				topRight = (int(topRight[0]), int(topRight[1]))
				bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
				bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
				topLeft = (int(topLeft[0]), int(topLeft[1]))

				# draw the bounding box of the ArUCo detection
				cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
				cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
				cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
				cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
				# compute and draw the center (x, y)-coordinates of the ArUco
				# marker
				cX = int((topLeft[0] + bottomRight[0]) / 2.0)
				cY = int((topLeft[1] + bottomRight[1]) / 2.0)
				cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
				# draw the ArUco marker ID on the image
				cv2.putText(image, str(markerID),
					(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (0, 255, 0), 2)
				print("[INFO] ArUco marker ID: {}".format(markerID))
				# show the output image
			cv2.imshow("Image", image)
			cv2.waitKey(0)

	if len(corners) > 0:
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			total_corners += len(corners)

	return total_corners

l_image_directory = glob.glob(os.path.join(all_image_directory,"*"))
print(l_image_directory)

for image_directory in l_image_directory:
	# Deal with multiple camera folders:
	l_camera_directory = glob.glob(os.path.join(image_directory,"*"))

	for camera_directory in l_camera_directory:
		all_image_fp = glob.glob(os.path.join(camera_directory,"*"))
		# Sorting by image #
		all_image_fp.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

		l_image_corner = []
		for image_fp in all_image_fp:
			image = cv2.imread(image_fp)
			if not image is None:
				n_corners = count_corners(image, viz=False)
				l_image_corner.append(n_corners)
				print("Image = {}\tCorners detected={}".format(image_fp, n_corners))
			else:
				print("{} is not an image.".format(image_fp))
		# Take the list of image corners and write it.
		bag_name = image_directory.split("/")[-1]
		f = open(os.path.join(results_directory, bag_name + "-aruco.txt"), "w")
		# print(l_image_corner)
		f.write("total:{}\n".format(sum(l_image_corner)))
		f.write("frame_n,corners\n")
		for i, n_corners in enumerate(l_image_corner):
			f.write("{},{}".format(i,str(n_corners)+"\n"))
		f.close()