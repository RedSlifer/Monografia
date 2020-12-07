import cv2
import os


def clear_files(path):
    files = [file for file in os.listdir(path) if file.endswith('.png')]

    for file in files:
        os.remove(os.path.join(path, file))


def get_frames(video, path):
    video = cv2.VideoCapture(video)
    success, image = video.read()
    counter = 0

    while success:
        if counter < 10:
            cv2.imwrite(fr'{path}\frame00{counter}.png', image)  # Save frame as image file
            success, image = video.read()
            counter += 1
        elif 10 <= counter < 100:
            cv2.imwrite(fr'{path}\frame0{counter}.png', image)  # Save frame as image file
            success, image = video.read()
            counter += 1
        else:
            cv2.imwrite(fr'{path}\frame{counter}.png', image)  # Save frame as image file
            success, image = video.read()
            counter += 1

    print('finish')


def save_frame(frame, path, counter):
    if counter < 10:
        cv2.imwrite(fr'{path}\image00{counter}.png', frame)
    elif 10 <= counter < 100:
        cv2.imwrite(fr'{path}\image0{counter}.png', frame)
    else:
        cv2.imwrite(fr'{path}\image{counter}.png', frame)


def path_name(path, counter):
    if counter < 10:
        return fr'{path}\image00{counter}.png'
    elif 10 <= counter < 100:
        return fr'{path}\image0{counter}.png'
    else:
        return fr'{path}\image{counter}.png'


def apply_filter(path_source, path_target, counter):
    if counter < 10:
        mask_frame = cv2.imread(fr'{path_source}\image00{counter}.png')
        blur = cv2.medianBlur(mask_frame, 7)
        filtered_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(fr'{path_target}\image00{counter}.png', filtered_frame)

    elif 10 <= counter < 100:
        mask_frame = cv2.imread(fr'{path_source}\image0{counter}.png')
        blur = cv2.medianBlur(mask_frame, 7)
        filtered_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(fr'{path_target}\image0{counter}.png', filtered_frame)

    else:
        mask_frame = cv2.imread(fr'{path_source}\image{counter}.png')
        blur = cv2.medianBlur(mask_frame, 7)
        filtered_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(fr'{path_target}\image{counter}.png', filtered_frame)
