import cv2
import argparse
import numpy
import Utils
import time

Utils.clear_files(r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\image_frames')
Utils.clear_files(r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\mask_frames')
Utils.get_frames('sample_1.AVI', r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\image_frames')

video_or_images_path = input()

parser = argparse.ArgumentParser(description='Program that implements a background subtraction using the KNN method')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image',
                    default=f'{video_or_images_path}')
parser.add_argument('--algo', type=str, help='Background subtraction method KNN', default='KNN')
args = parser.parse_args()

background_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
background_subtractor.setkNNSamples(7)

capture_frames = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))

counter = 0

if not capture_frames.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

while True:
    _, frame = capture_frames.read()
    if frame is None:
        break

    foreground_mask_frame = background_subtractor.apply(frame)
    Utils.save_frame(foreground_mask_frame, r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\mask_frames', counter)
    Utils.apply_filter(r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\mask_frames', r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\filtered_mask', counter)

    auxiliary_mask = cv2.imread(
        Utils.path_name(r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\mask_frames', counter))

    white_pixels = numpy.where(
        (auxiliary_mask[:, :, 0] == 255) & (auxiliary_mask[:, :, 1] == 255) & (auxiliary_mask[:, :, 2] == 255))

    auxiliary_mask[white_pixels] = (0, 0, 0)
    auxiliary_mask = cv2.cvtColor(auxiliary_mask, cv2.COLOR_BGR2GRAY)
    Utils.save_frame(auxiliary_mask, r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\modified_mask_frames',
                     counter)

    reconstructed_image = cv2.inpaint(frame, auxiliary_mask, 3, cv2.INPAINT_NS)
    Utils.save_frame(reconstructed_image, r'C:\Users\cassio\PycharmProjects\FinalProject\Frames\reconstructed-frames',
                     counter)

    _, threshold_frame = cv2.threshold(foreground_mask_frame, 127, 255, cv2.THRESH_OTSU)

    counter += 1
    '''
    cv2.imshow('frame', frame)
    cv2.imshow('foreground mask frame', foreground_mask_frame)
    cv2.imshow('modified foreground mask', auxiliary_mask)
    cv2.imshow('threshold frame', threshold_frame)
    cv2.imshow('reconstructed image', reconstructed_image)
    '''
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
