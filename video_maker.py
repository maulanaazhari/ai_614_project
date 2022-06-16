
import cv2
import subprocess
import os
from tqdm import tqdm
import numpy as np

name_list = ['random_tree', 'random_tree_informed', 'rrt', 'rrt_star', 'rrt_star2']
# name_list = ['random_tree']
for name in name_list:
    # name = 'random_tree_informed'
    image_folder = [f'{name}_env00', f'{name}_env01', f'{name}_env02']
    video_name = f'{name}.avi'

    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images1 = ["{}/{}.png".format(image_folder[0], i) for i in range(5000)]
    images2 = ["{}/{}.png".format(image_folder[1], i) for i in range(5000)]
    images3 = ["{}/{}.png".format(image_folder[2], i) for i in range(5000)]

    # print(images1[0])
    frame = cv2.imread(images1[0])
    height, width, layers = frame.shape
    empty_img = np.ones((height, width, layers))*255

    video = cv2.VideoWriter(video_name, 0, 30, (width*2,height*2))

    for i in tqdm(range(5000)):
        im1 = cv2.imread(images1[i])
        im2 = cv2.imread(images2[i])
        im3 = cv2.imread(images3[i])
        st1 = np.concatenate((im1, im2), axis=1)
        st2 = np.concatenate((im3, empty_img), axis=1)
        st = np.concatenate((st1, st2), axis=0)
        # st = cv2.resize(st, (width, height), interpolation=cv2.INTER_AREA)
        video.write(st.astype(np.uint8))

    cv2.destroyAllWindows()
    video.release()
    # subprocess.call(['ffmpeg', '-i', f'{name}.avi', '-vcodec', 'libx265', '-crf', '28', f'{name}_cmp.avi'])
    print(f'{name} finished!')