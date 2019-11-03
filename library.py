import cv2
import numpy as np

def boundary_adjacent_count(pre_image, boundary_vector):
	'''Counts the number of adjacent pixel changes in a particular boundary'''

	[m,m] = pre_image.shape

	for count in range(int(m/2)):

		#top:
		for i in range(count,m-count-1):
			if pre_image[count][i] != pre_image[count][i+1]:
				boundary_vector[count][pre_image[count][i]][1] += 1

		#right:
		for j in range(count,m-count-1):
			if pre_image[j][m-count-1] != pre_image[j+1][m-count-1]:
				boundary_vector[count][pre_image[j][m-count-1]][1] += 1

		#bottom:
		for i in range(m-count-1,count,-1):
			if pre_image[m-count-1][i] != pre_image[m-count-1][i-1]:
				boundary_vector[count][pre_image[m-count-1][i]][1] += 1

		#left:
		for j in range(m-count-1,count,-1):
			if pre_image[j][count] != pre_image[j-1][count]:
				boundary_vector[count][pre_image[j][count]][1] += 1

	return boundary_vector


def boundary_frequency(pre_image):
	'''Calculates the frequency of each pixel in each boundary'''

	[m,m] = pre_image.shape
	#print(m)

	boundary_vector = np.zeros((m//2 ,256 ,2), dtype=int) #[no. of boundaries, frequency, difference]
	#print(boundary_vector.shape)

	for count in range(0,int(m/2),1):

		#top and bottom
		for i in range(count,m-count):

			boundary_vector[count][pre_image[count][i]][0] += 1
			#if pre_image[count][i] != pre_image[count][i-1]:
				#boundary_vector[pre-image[count][i]][1] += 1

			boundary_vector[count][pre_image[m-count-1][i]][0] += 1
			#if pre_image[m-count][i] != pre_image[m-count][i-1]:
				#boundary_vector[pre-image[count][i]][1] += 1


		#left and right
		for j in range(count+1,m-count-1):
			boundary_vector[count][pre_image[j][count]][0] += 1
			#if pre_image[count][i] != pre_image[count][i-1]:
				#boundary_vector[pre-image[count][i]][1] += 1

			boundary_vector[count][pre_image[j][m-count-1]][0] += 1
			#if pre_image[m-count][i] != pre_image[m-count][i-1]:
				#boundary_vector[pre-image[count][i]][1] += 1

	return boundary_vector


def preprocess_dim(image):
	'''Resize the image for suitable comparison'''

	grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #matrix
	[m,n] = grayimage.shape

	if m>n:
		if m%2 == 0:
			grayimage = cv2.resize(grayimage,(m,m))
		else:
			grayimage = cv2.resize(grayimage,(m+1,m+1))
	else:
		if n%2 == 0:
			grayimage = cv2.resize(grayimage,(n,n))
		else:
			grayimage = cv2.resize(grayimage,(n+1,n+1))

	grayimage = cv2.resize(grayimage,(200,200))

	return grayimage

def get_feature_vector(image):

	'''Get final feature vector for an image'''

	preprocessed_image = preprocess_dim(image)
	boundary_vector = boundary_frequency(preprocessed_image)
	final_vector = boundary_adjacent_count(preprocessed_image,boundary_vector)

	return final_vector

def print_vector(feature_vector):

	'''Print the feature vector'''

	for i in range(100):
		print("boundary {}".format(i))
		for j in range(256):
			print(j,feature_vector[i][j][0],feature_vector[i][j][1])

def print_non_zero(feature_vector):

	'''Print the non zero elements of the feature vector'''

	for i in range(100):
		print("boundary {}".format(i))
		for j in range(256):
			if feature_vector[i][j][0] and feature_vector[i][j][1] != 0:
				print(j,feature_vector[i][j][0],feature_vector[i][j][1])


def compare(image1, image2):
	
	'''Compare the feature vectors of 2 images'''

	fv1 = get_feature_vector(image1)
	fv2 = get_feature_vector(image2) 
	
	distance = 0
	total_count = 0
	for i in range(100):
		boundary_sum = 0
		freq = 0 
		for j in range(256):
			a = fv1[i][j][0] + fv2[i][j][0]
			b = fv1[i][j][1] + fv2[i][j][1]
			c = fv1[i][j][0] - fv2[i][j][0]
			d = fv1[i][j][1] - fv2[i][j][1]
			if a + b != 0:
				boundary_sum += abs(c) + abs(d)
				freq += fv1[i][j][0]
		
		total_count += freq
		distance += boundary_sum*freq 

	distance = float(distance/total_count) 

	return distance 