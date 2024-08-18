import cv2
import numpy as np

def superpixel_leaf_segmentation(image):
    # 將圖像轉換為Lab顏色空間
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # 使用SLIC算法進行超像素分割
    slic = cv2.ximgproc.createSuperpixelSLIC(lab_image, region_size=30, ruler=10.0)
    slic.iterate(10)  # 迭代次數

    # 獲取超像素分割後的標籤
    superpixel_mask = slic.getLabelContourMask()

    # 將超像素邊界設為紅色
    image_with_contours = image.copy()
    image_with_contours[superpixel_mask == 255] = [0, 0, 255]  # 設置紅色

    # 創建一個空白圖像來存儲分割結果
    labels = slic.getLabels()
    segmented_image = np.zeros_like(image)

    # 遍歷所有標籤，根據標籤將像素賦值到分割圖像中
    for label in np.unique(labels):
        mask = labels == label
        segmented_image[mask] = image[mask]

    return segmented_image, image_with_contours

# 測試超像素分割
if __name__ == "__main__":
    img = cv2.imread("./leaves_dataset/IMG_4325.jpg")
    img = cv2.resize(img, (640, 480))
    # 定義 ROI 的寬度和高度
    roi_width = 450
    roi_height = 480 // 2

    # 定義中心點
    center_x, center_y = 640 // 2, 480 * 3 // 4
    # 計算 ROI 的邊界
    x = center_x - roi_width // 2
    y = center_y - roi_height // 2
    w = roi_width
    h = roi_height

    # 提取 ROI
    roi = img[y:y + h, x:x + w].copy()
    segmented_image, image_with_contours = superpixel_leaf_segmentation(roi)

    # 顯示分割後的圖像和邊界圖像
    cv2.imshow("Segmented Leaves", segmented_image)
    cv2.imshow("Image with Superpixel Contours", image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
