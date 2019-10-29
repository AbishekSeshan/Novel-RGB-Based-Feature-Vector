from library import get_feature_vector, print_vector, print_non_zero
import cv2
import numpy as np 


def compare(image1, image2):
    
    '''Compare the feature vectors of 2 images'''

    fv1 = get_feature_vector(image1)
    fv2 = get_feature_vector(image2) 
    
    distance = 0

    for i in range(3):
        boundary_sum = 0
        freq = 0 
        for j in range(256):
            freq += fv1[i][j][0] #frequency of ith boundary, jth pixel
            boundary_sum += abs(fv1[i][j][0] -fv2[i][j][0]) + abs(fv1[i][j][1] -fv2[i][j][1])
        
        distance += boundary_sum/freq

    return distance 


if __name__ == "__main__":
    
    query_image = cv2.imread('query.jpg')
    db_image1 = cv2.imread('test1.jpg')
    db_image2 = cv2.imread('test2.jpg')    
    distance1 = compare(db_image1, query_image)
    distance2 = compare(db_image2, query_image) 
    print("Distance 1 is: {}".format(distance1))
    print("Distance 2 is: {}".format(distance2)) 
    
    