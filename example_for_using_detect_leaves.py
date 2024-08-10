import cv2
import numpy as np
import os
from detect_leaves import process_image 
# 設定圖像資料夾路徑
image_folder = './leaves_test_dataset/'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.JPG')]
image_files.sort()  # 將圖片排序

# 初始顯示圖片
initial_image = "IMG_4327.JPG"
image_index = image_files.index(initial_image) if initial_image in image_files else 0

image = cv2.imread(os.path.join(image_folder, image_files[image_index]))


image_width = 4032
image_height = 3024

example_from_detect_trash_annotations = [
    {
        "class_id": 0,
        "x_center": int(0.5 * image_width),  # 480.0
        "y_center": int(0.7 * image_height), # 480.0
        "width": int(0. * image_width),      # 320.0
        "height": int(0. * image_height)     # 192.0
    },
]

# 處理並顯示初始圖片
filtered_image,annotations=process_image(image=image, 
              anntation_from_detect_trash=example_from_detect_trash_annotations,
              image_name=image_files[image_index])

#cv2.imshow('Filtered1 Image', filtered_image)
while True:
    key = cv2.waitKey(0)  # 使用較小的延遲時間，以便更快響應按鍵
    if key & 0xFF == ord('q'):
        break
# 關閉所有窗口
cv2.destroyAllWindows()
