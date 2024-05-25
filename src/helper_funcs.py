import pytesseract
import numpy as np
import imutils
import cv2

def apply_filter(image):
    filtered_image = filter_image(image)
    location, cropped_image,parameters = find_contours(filtered_image,image)
    final_image = mask_image(image, location)
    return final_image,parameters

def scan_plate(plate):
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
    custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\ -  --psm 7'
    plate_number = (pytesseract.image_to_string(plate,lang='eng',config=custom_config))
    plate_number = plate_number.strip()
    return plate_number


def filter_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
    cv2.destroyWindow('gray')
    bfiltered_image = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow('bilateral', bfiltered_image)
    cv2.waitKey(0)
    cv2.destroyWindow('bilateral')
    canny_image = cv2.Canny(bfiltered_image, 50, 200)
    cv2.imshow('canny', canny_image)
    cv2.waitKey(0)
    cv2.destroyWindow('canny')
    return canny_image

def find_contours(image, default_image):
    keypoints = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    countours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    location = None
    for countour in countours:
        #perimeter=cv2.arcLength(countour,True)
        #approx = cv2.approxPolyDP(countour, perimeter, True)
        approx = cv2.approxPolyDP(countour, 0.02 * cv2.arcLength(countour, True), True)
        if len(approx) == 4:
            location = approx
            #x,y,w,h = cv2.boundingRect(countour)
            #crp_img = default_image[y:y+h,x:x+w]
            cropped_image = crop_image(default_image,countour)
            parameters = return_parameters(countour)
            break
        else:
            cropped_image = image
            parameters = None
    
    return location,cropped_image,parameters


def crop_image(image, countour):
    x,y,w,h = cv2.boundingRect(countour)
    cropped_image = image[y:y+h,x:x+w]
    cv2.imshow('cropped', cropped_image)
    cv2.waitKey(0)
    cv2.destroyWindow('cropped')
    return cropped_image

def return_parameters(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return x,y,w,h

def put_rectangle_and_text(image, parameters, text):
    try:
        cv2.rectangle(image,(parameters[0], parameters[1]), (parameters[0] + parameters[2], parameters[1] + parameters[3]),(0, 255, 0) , 2)
        cv2.putText(image, text, (parameters[0], parameters[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow('rectangled_image', image)
        cv2.waitKey(0)
        cv2.destroyWindow('rectangled_image')
    except TypeError as e:
        pass

def mask_image(image,location):
    if location is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        new_image = cv2.drawContours(mask, [location], -1, 255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)
        return new_image
    else:
        return image