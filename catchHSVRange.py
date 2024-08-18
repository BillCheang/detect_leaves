import cv2
import numpy as np
import os
from detect_leaves import process_image as P_I
# 設定圖像資料夾路徑
image_folder = './leaves_test_dataset/'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.JPG')]
image_files.sort()  # 將圖片排序

# 初始顯示圖片
initial_image = "IMG_4355.JPG" #4361
image_index = image_files.index(initial_image) if initial_image in image_files else 0

# 回調函數，用於處理滑鼠事件
def mouse_callback(event, x, y, flags, param):
    global hsv_values, image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 獲取在點擊位置的HSV值
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        hsv_values = (h, s, v)
        print(f"HSV Values at ({x}, {y}): H={h}, S={s}, V={v}")

# 初始化變數
hsv_values = (0, 0, 0)

def load_image(index):
    global image, hsv_image
    image_path = os.path.join(image_folder, image_files[index])
    image = cv2.imread(image_path)
    #image = cv2.resize(image, (640, 480))
    image,_=P_I(image, [],image_files[image_index])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 加載初始圖片
load_image(image_index)

# 創建一個窗口和滑塊
cv2.namedWindow('Image')
cv2.createTrackbar('H Lower', 'Image', 0, 179, lambda x: None)
cv2.createTrackbar('H Upper', 'Image', 179, 179, lambda x: None)
cv2.createTrackbar('S Lower', 'Image', 0, 255, lambda x: None)
cv2.createTrackbar('S Upper', 'Image', 255, 255, lambda x: None)
cv2.createTrackbar('V Lower', 'Image', 0, 255, lambda x: None)
cv2.createTrackbar('V Upper', 'Image', 255, 255, lambda x: None)

cv2.setMouseCallback('Image', mouse_callback)

while True:
    # 讀取滑塊的值
    h_lower = cv2.getTrackbarPos('H Lower', 'Image')
    h_upper = cv2.getTrackbarPos('H Upper', 'Image')
    s_lower = cv2.getTrackbarPos('S Lower', 'Image')
    s_upper = cv2.getTrackbarPos('S Upper', 'Image')
    v_lower = cv2.getTrackbarPos('V Lower', 'Image')
    v_upper = cv2.getTrackbarPos('V Upper', 'Image')
    
    # 設定 HSV 範圍
    lower_bound = np.array([h_lower, s_lower, v_lower])
    upper_bound = np.array([h_upper, s_upper, v_upper])
    
    # 計算遮罩
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
  
    # 應用遮罩
    dst = cv2.bitwise_and(image, image, mask=mask)
    h,w,_=dst.shape
    
    new_width = int(w * 20 / 100)
    new_height = int(h *20  / 100)
    dim = (new_width, new_height)
    dst = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
    # 顯示圖片檔名
    filename = image_files[image_index]
    #cv2.putText(dst, filename, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 顯示影像
    cv2.imshow('Image', dst)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('n'):
        # 跳到下一張圖片
        image_index = (image_index + 1) % len(image_files)
        load_image(image_index)
    elif key == ord('p'):
        # 跳到上一張圖片
        image_index = (image_index - 1) % len(image_files)
        load_image(image_index)

# 釋放資源
cv2.destroyAllWindows()