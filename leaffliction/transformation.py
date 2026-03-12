#!/usr/bin/env python3

import argparse
import cv2
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image", nargs="?", help="Path to a single image"
    )

    parser.add_argument("-src", help="Source directory")
    parser.add_argument("-dst", help="Destination directory")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-blur", action="store_true", help="Gaussian blur")
    group.add_argument("-mask", action="store_true", help="Mask")
    group.add_argument("-roi", action="store_true", help="ROI objects")
    group.add_argument("-analyze", action="store_true", help="Analyze object")
    group.add_argument("-pseudolandmarks", action="store_true", help="Pseudolandmarks")

    args = parser.parse_args()

    transform_flags = [
        args.blur,
        args.mask,
        args.roi,
        args.analyze,
        args.pseudolandmarks,
    ]
    transform_count = sum(transform_flags)

    if args.image and not args.src and not args.dst:
        if transform_count != 0:
            parser.error("Transformation flags are not allowed in single image mode.")
        return args

    if args.src and args.dst and not args.image:
        if transform_count != 1:
            parser.error("Directory mode requires exactly ONE transformation flag.")
        return args

    parser.error(
        "Invalid usage.\n"
        "Use either:\n"
        "  ./Transformation.py <image_path>\n"
        "OR\n"
        "  ./Transformation.py -src <source_dir> -dst <dest_dir> -<mask>"
    )


def process_directory(src, dst, mask):
    pass


def process_image(path):
    img, _, _ = pcv.readimage(filename=path)

    results = {
        "Original": img,
        "Gaussian blur": transform_blur(img),
        "Mask": transform_mask(img),
        "ROI objects": transform_roi(img),
        "Analyze object": transform_analyze(img),
        "Pseudolandmarks": transform_pseudolandmarks(img),
    }

    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, (title, result) in zip(axes.flatten(), results.items()):
        if result.ndim == 2:
            ax.imshow(result, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    fig1.tight_layout()
    plot_histogram(img)
    plt.show()

def transform_blur(img):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    blurred = pcv.gaussian_blur(img=gray, ksize=(11, 11), sigma_x=0)
    return blurred

def transform_mask(img):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    blurred = pcv.gaussian_blur(img=gray, ksize=(11, 11), sigma_x=0)
    binary = pcv.threshold.binary(gray_img=blurred, threshold=120, object_type='light')
    masked = pcv.apply_mask(img=img, mask=binary, mask_color='white')
    return masked

def transform_roi(img):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    blurred = pcv.gaussian_blur(img=gray, ksize=(11, 11), sigma_x=0)
    binary = pcv.threshold.binary(gray_img=blurred, threshold=120, object_type='dark')
    roi = pcv.roi.rectangle(img=img, x=5, y=5, h=245, w=245)
    kept_mask = pcv.roi.filter(mask=binary, roi=roi, roi_type='partial')
    roi_img = img.copy()
    cv2.rectangle(roi_img, (5, 5), (250, 250), (255, 0, 0), 2)
    roi_img[kept_mask > 0] = [0, 255, 0]
    return roi_img

def transform_analyze(img):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    blurred = pcv.gaussian_blur(img=gray, ksize=(11, 11), sigma_x=0)
    binary = pcv.threshold.binary(gray_img=blurred, threshold=120, object_type='dark')
    roi = pcv.roi.rectangle(img=img, x=5, y=5, h=245, w=245)
    kept_mask = pcv.roi.filter(mask=binary, roi=roi, roi_type='partial')
    shape_img = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    return shape_img

def transform_pseudolandmarks(img):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    blurred = pcv.gaussian_blur(img=gray, ksize=(11, 11), sigma_x=0)
    binary = pcv.threshold.binary(gray_img=blurred, threshold=120, object_type='dark')
    roi = pcv.roi.rectangle(img=img, x=5, y=5, h=245, w=245)
    kept_mask = pcv.roi.filter(mask=binary, roi=roi, roi_type='partial')
    top, bottom, center = pcv.homology.x_axis_pseudolandmarks(img=img, mask=kept_mask)
    landmark_img = img.copy()
    for pt in top:
        cv2.circle(landmark_img, (int(pt[0][0]), int(pt[0][1])), 4, (0, 0, 255), -1)
    for pt in bottom:
        cv2.circle(landmark_img, (int(pt[0][0]), int(pt[0][1])), 4, (255, 0, 0), -1)
    for pt in center:
        cv2.circle(landmark_img, (int(pt[0][0]), int(pt[0][1])), 4, (0, 255, 255), -1)
    return landmark_img


def plot_histogram(img):
    total_pixels = img.shape[0] * img.shape[1]

    plt.figure(figsize=(8,6))

    # RGB
    colors = {'blue':0, 'green':1, 'red':2}
    for name, i in colors.items():
        hist = cv2.calcHist([img],[i],None,[256],[0,256]).flatten()
        hist = (hist / total_pixels) * 100
        plt.plot(hist, label=name)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_channels = {'hue':0, 'saturation':1, 'value':2}
    for name, i in hsv_channels.items():
        hist = cv2.calcHist([hsv],[i],None,[256],[0,256]).flatten()
        hist = (hist / total_pixels) * 100
        plt.plot(hist, label=name)

    # LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_channels = {'lightness':0, 'green-magenta':1, 'blue-yellow':2}
    for name, i in lab_channels.items():
        hist = cv2.calcHist([lab],[i],None,[256],[0,256]).flatten()
        hist = (hist / total_pixels) * 100
        plt.plot(hist, label=name)

    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels (%)")
    plt.title("Color histogram")
    plt.xlim([0,255])
    plt.legend()
    plt.grid(True)


if __name__ == "__main__":
    args = parse_arguments()

    if args.image:
        process_image(args.image)
    elif args.src and args.dst and args.mask:
        process_directory(args.src, args.dst, args.mask)
    else:
        print("Invalid usage.")
