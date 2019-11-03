from library import get_feature_vector, print_vector, print_non_zero, compare, preprocess_dim
import cv2
import numpy as np 


if __name__ == "__main__":
	
	query_image = cv2.imread('query.jpg')
	db_image1 = cv2.imread('test1.jpg')
	db_image2 = cv2.imread('test2.jpg')
	db_image3 = cv2.imread('test3.jpg')    
	distance1 = compare(db_image1, query_image)
	distance2 = compare(db_image2, query_image)
	distance3 = compare(db_image3, query_image) 
	print("Distance 1 is: {}".format(distance1))
	print("Distance 2 is: {}".format(distance2))
	print("Distance 3 is: {}".format(distance3))
	
	
	
	