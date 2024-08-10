import cv2
import numpy as np

def process_image(image, anntation_from_detect_trash, image_name=""):
    # 常數設置
    SCALE_PERCENT = 10  # 縮放比例
    AREA_THRESHOLD_PERCENT = 0.03  # 面積閾值百分比
    COLOR_RANGES = [
        {"name": "leave1_green", "lower": (29, 40, 92), "upper": (67, 200, 255)},
        {"name": "leave_yellow", "lower": (18, 62, 82), "upper": (40, 180, 255)},
        {"name": "leave_brown", "lower": (6, 35, 9), "upper": (38, 143, 255)},
        #{"name": "test", "lower": (0, 0, 0), "upper": (255, 255, 255)},
    ]
    EXCLUDE_COLOR_RANGES = [
       {"name": "ground", "lower": (8, 0, 116), "upper": (40, 40, 255)},
       {"name": "nosie", "lower": (0, 19, 101), "upper": (18, 65, 113)},
    ]

    h, w, _ = image.shape

    # Step 1: 生成 trash 的掩膜
    trash_mask = generate_trash_mask(anntation_from_detect_trash, h, w)

    # Step 2: 排除 trash 的區域
    image_without_trash = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(trash_mask))

    # Step 3: 提取 ROI
    roi, x_roi, y_roi, w_roi, h_roi, center_x, center_y = extract_roi(image_without_trash, w, h)

    # Step 4: 進行顏色過濾
    hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).copy()
    final_mask = apply_color_filters(hsv_image, COLOR_RANGES, EXCLUDE_COLOR_RANGES)

    # Step 5: 檢測輪廓並生成註釋
    contours = detect_contours(final_mask, w_roi, h_roi, AREA_THRESHOLD_PERCENT)
    filtered_image, annotations = annotate_contours(contours, roi, w_roi, h_roi, final_mask)

    # Step 6: 合併 trash 的註釋
    annotations.extend(anntation_from_detect_trash)

    # Step 7: 縮放並顯示圖像
    display_images(image_without_trash, roi, filtered_image, w, h, SCALE_PERCENT, x_roi, y_roi, w_roi, h_roi, center_x, center_y, image_name)

    return filtered_image, annotations

def generate_trash_mask(anntation_from_detect_trash, h, w):
    """生成用於標記 trash 的掩膜"""
    trash_mask = np.zeros((h, w), dtype=np.uint8)
    for annotation in anntation_from_detect_trash:
        x_center = int(annotation['x_center'])
        y_center = int(annotation['y_center'])
        width = int(annotation['width'])
        height = int(annotation['height'])

        x1 = max(x_center - width // 2, 0)
        y1 = max(y_center - height // 2, 0)
        x2 = min(x_center + width // 2, w)
        y2 = min(y_center + height // 2, h)

        cv2.rectangle(trash_mask, (x1, y1), (x2, y2), 255, -1)
    return trash_mask

def extract_roi(image_without_trash, w, h):
    """提取圖像的 ROI"""
    w_roi = int(w // 2)
    h_roi = int(h // 2)
    center_x, center_y = w // 2, h * 3 // 4
    x_roi = center_x - w_roi // 2
    y_roi = center_y - h_roi // 2
    
    roi = image_without_trash[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
    return roi, x_roi, y_roi, w_roi, h_roi, center_x, center_y

def apply_color_filters(hsv_image, color_ranges, exclude_color_ranges):
    """應用顏色過濾並生成最終掩膜"""
    filter_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    exclude_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    # 生成過濾掩膜
    for color in color_ranges:
        lower_bound = np.array(color["lower"])
        upper_bound = np.array(color["upper"])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        filter_mask = cv2.bitwise_or(filter_mask, mask)

    # 生成排除掩膜
    for color in exclude_color_ranges:
        lower_bound = np.array(color["lower"])
        upper_bound = np.array(color["upper"])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        exclude_mask = cv2.bitwise_or(exclude_mask, mask)

    # 濾出特定顏色
    final_mask = cv2.bitwise_and(filter_mask, cv2.bitwise_not(exclude_mask))

    # 去除噪點
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    return final_mask

def detect_contours(final_mask, w_roi, h_roi, area_threshold_percent):
    """檢測圖像中的輪廓"""
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = w_roi * h_roi
    area_threshold = total_area * area_threshold_percent // 100

    # 過濾掉面積小於閾值的輪廓
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_threshold]
    return contours

def annotate_contours(contours, roi, w_roi, h_roi, final_mask):
    """繪製輪廓並生成註釋"""
    filtered_image = cv2.bitwise_and(roi, roi, mask=final_mask)
    annotations = []
    for cnt in contours:
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        
        # 在 filtered_image 上繪製 bounding box
        cv2.rectangle(filtered_image, (x_cnt, y_cnt), (x_cnt + w_cnt, y_cnt + h_cnt), (0, 255, 0), 10)
        
        # 在 ROI 上繪製 bounding box
        cv2.rectangle(roi, (x_cnt, y_cnt), (x_cnt + w_cnt, y_cnt + h_cnt), (0, 255, 0), 10)

        x_center = (x_cnt + w_cnt / 2) / w_roi
        y_center = (y_cnt + h_cnt / 2) / h_roi
        width = w_cnt / w_roi
        height = h_cnt / h_roi

        class_id = 0
        annotation = {
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height
        }
        annotations.append(annotation)
    return filtered_image, annotations


def display_images(image_without_trash, roi, filtered_image, w, h, scale_percent, x_roi, y_roi, w_roi, h_roi, center_x, center_y, image_name):
    """縮放並顯示處理後的圖像"""
    new_width = int(w * scale_percent / 100)
    new_height = int(h * scale_percent / 100)
    dim = (new_width, new_height)

    roi_resized = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)
    filtered_image_resized = cv2.resize(filtered_image, dim, interpolation=cv2.INTER_AREA)
    image_resized = cv2.resize(image_without_trash, dim, interpolation=cv2.INTER_AREA)
    
    cv2.rectangle(image_resized, 
                  (int(x_roi * scale_percent / 100), int(y_roi * scale_percent / 100)), 
                  (int((x_roi + w_roi) * scale_percent / 100), int((y_roi + h_roi) * scale_percent / 100)), 
                  (0, 255, 0), 2)
    cv2.circle(image_resized, 
               (int(center_x * scale_percent / 100), int(center_y * scale_percent / 100)), 
               5, (0, 0, 255), -1)
    cv2.putText(image_resized, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('ROI', roi_resized)
    cv2.imshow('Original Image', image_resized)
    cv2.imshow('Filtered Image', filtered_image_resized)
