import cv2
from os import listdir

"""scale image to percent width and height image"""

imgs_out_path = "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analyis_Machine_Intelligence/Abschlussprojekt/forms_test_small/"
imgs_path = "C:/Users/Pasca/Documents/Goethe-Uni/Master_1.Semester/Pattern_Analyis_Machine_Intelligence/Abschlussprojekt/forms_test/"
imgs = listdir(imgs_path)

percent = 25

for img_path in imgs:
    img = cv2.imread(imgs_path + img_path, cv2.IMREAD_UNCHANGED)

    print('Original Dimensions : ', img.shape)
    width = int(img.shape[1] *  percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(imgs_out_path + img_path, resized)
