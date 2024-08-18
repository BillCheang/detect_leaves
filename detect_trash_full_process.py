from detect_leaves import process_image as detect_leaves
from detect_trash import detect_trash_objects
import pixel_to_mm_converter
import size_threshold_filter
import publish_to_ros

import cv2
import numpy as np

'''
 annotations_template= {
            "class_id": 0,
            "x_center": 0,  
            "y_center": 0, 
            "width": 0,     
            "height": 0 ,   
    }
'''
def process_trash_detection():
    img = cv2.imread()
    
    # Step 1: Detect trash objects in the image
    annotations = detect_trash_objects(img)
    
    # Step 2: Detect leaves and combine with trash annotations
    annotations = detect_leaves(image=img,
                                anntation_from_detect_trash=annotations)
    
    # Step 3: Convert pixel measurements to millimeters
    annotations = pixel_to_mm_converter(image=img,
                                        annotations=annotations)
    
    # Step 4: Filter annotations by size threshold
    annotations = size_threshold_filter(annotations=annotations)
    
    # Step 5: Publish the processed annotations to ROS
    publish_to_ros(annotations)

