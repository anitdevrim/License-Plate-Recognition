import cv2
import pandas as pd
from helper_funcs import *

def main():

    plate_images = []
    images_path = [
    '../images/plaka1.jpg',
    '../images/plaka2.jpg',
    '../images/plaka3.jpg',
    '../images/plaka4.jpg',
    '../images/plaka5.jpg',
    '../images/plaka7.jpg',
    '../images/plaka8.jpg',
    '../images/plaka9.jpg',
    '../images/plaka10.jpg',
    '../images/plaka11.jpg',
    '../images/plaka12.jpg',
    '../images/plaka13.jpg',
    '../images/plaka14.jpg',
    '../images/plaka15.jpg',
    '../images/plaka16.jpg',
    '../images/plaka17.jpg',
    '../images/plaka6.jpg',
    '../images/plaka18.jpg',
    ]


    for path in images_path:
        plate_images.append(cv2.imread(path))

    plate_texts = []

    for i,image in enumerate(plate_images):
        final_image, parameters = apply_filter(image)
        plate_text = scan_plate(final_image)
        put_rectangle_and_text(image,parameters,plate_text)
        plate_texts.append(plate_text)
        print(f'{i} - {plate_text}')
    
    sliced_paths = [path[10:] for path in images_path]
    df = pd.DataFrame({
    'Column1': sliced_paths,
    'Column2': plate_texts
    })
    df.to_csv('Dataset_VehicleNo.csv')


if __name__ == '__main__':
    main()
