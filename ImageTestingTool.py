import numpy as np
import os
from IPython.display import display, Image
from PIL import Image

# filenames for the training and testing folders
train_folder = "Images"

# standard dimensions to which all images will be rescaled
dimensions = (50, 50)

# maximum angle by which the image can be rotated during data augmentation
max_angle = 15


# function to rotate an image by a given angle and fill in the black corners created
# with a specified color
def rotate_img(image, angle, color, filter=Image.NEAREST):
    if image.mode == "P" or filter == Image.NEAREST:
        matte = Image.new("1", image.size, 1)  # mask
    else:
        matte = Image.new("L", image.size, 255)  # true matte
    bg = Image.new(image.mode, image.size, color)
    bg.paste(
        image.rotate(angle, filter),
        matte.rotate(angle, filter)
    )
    return bg


# function to turn grey-colored backgrounds to white. r, b and g specify the
# exact shade of grey color to eliminate. Source: stackoverflow.
def make_greyscale_white_bg(im, r, b, g):
    im = im.convert('RGBA')  # Convert to RGBA
    data = np.array(im)  # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

    # Replace grey with white... (leaves alpha values alone...)
    grey_areas = (red == r) & (blue == b) & (green == g)
    data[..., :-1][grey_areas.T] = (255, 255, 255)  # Transpose back needed

    im2 = Image.fromarray(data)
    im2 = im2.convert('L')  # convert to greyscale image
    return im2


# # Make a specified number of copies if the given image by rotating the original image by
# # some random angle, and save the images according to the naming scheme followed by the original images
def random_rotate(img, copies, curr_filename, path):
    c_color = img.getpixel((0, 0))  # get the pixel values of top-left corner of image
    for i in range(copies):
        # rotate image by a random angle from [-max_angle, max_angle], using the c_color to fill in the corners
        new_im = rotate_img(img, np.random.randint((0 - max_angle), max_angle), c_color)
        # save new image to file
        new_im.save(os.path.join(path, "bcc" + str(curr_filename).zfill(6) + ".bmp"))
        curr_filename = curr_filename + 1


# augment the dataset by adding random rotations. The count of the original images is needed
# for naming the new images in a sequential order
def augment_by_rotations(folder, prev_cnt):
    classes = [os.path.join(folder, d) for d in sorted(os.listdir(folder))]  # get list of all sub-folders in folder

    for path_to_folder in classes:
        if os.path.isdir(path_to_folder):
            images = [os.path.join(path_to_folder, i) for i in sorted(os.listdir(path_to_folder)) if i != '.DS_Store']
            filename = prev_cnt
            for image in images:
                im = Image.open(image)
                # make 4 copies of each image, with random rotations added in
                random_rotate(im, 4, filename, path_to_folder)
                filename = filename + 4
            print("Finished augmenting " + path_to_folder)


# function to invert colors (black -> white and white-> black). Since most of the image consists
# of white areas, specified by (255, 255, 255) in RGB, inverting the colors means more zeros, making
# future operations less computationally expensive
def invert_colors(im):
    im = im.convert('RGBA')  # Convert to RGBA
    data = np.array(im)  # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

    # Replace black with red temporarily... (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    data[..., :-1][black_areas.T] = (255, 0, 0)  # Transpose back needed

    # Replace white areas with black
    white_areas = (red == 255) & (blue == 255) & (green == 255)
    data[..., :-1][white_areas.T] = (0, 0, 0)  # Transpose back needed

    # Replace red areas (originally white) with black
    red_areas = (red == 255) & (blue == 0) & (green == 0)
    data[..., :-1][red_areas.T] = (255, 255, 255)  # Transpose back needed

    im2 = Image.fromarray(data)
    im2 = im2.convert('L')  # convert to greyscale image
    return im2


# function to process images (resizing, removal of grey backgrounds if any, color inversion, greyscale conversion)
def process_images(folder):
    classes = [os.path.join(folder, d) for d in sorted(os.listdir(folder))]  # get list of all sub-folders in folder
    img_cnt = 0

    for class_x in classes:
        if os.path.isdir(class_x):
            # get paths to all the images in this folder
            images = [os.path.join(class_x, i) for i in sorted(os.listdir(class_x)) if i != '.DS_Store']
            for image in images:
                img_cnt = img_cnt + 1
                if (img_cnt % 1000 == 0):  # show progress
                    print("Processed %s images" % str(img_cnt))

                im = Image.open(image)
                im = im.resize(dimensions)  # resize image according to dimensions set
                im = make_greyscale_white_bg(im, 127, 127, 127)  # turn grey background (if any) to white, and
                # convert into greyscale image with 1 channel
                im = invert_colors(im)
                im.save(image)  # overwrite previous image file with new image

    print("Finished processing images, images found = ")
    print(img_cnt)


process_images(train_folder)
