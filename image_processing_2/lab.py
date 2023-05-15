"""
6.1010 Spring '23 Lab 2: Image Processing 2
"""

#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


# VARIOUS FILTERS


def get_pixel(image, row, col, boundary_behavior="zero"):
    """
    Returns pixel value from image at specified row and col

    Accounts for edge cases, dependent on boundary_behavior parameter:
    - zero: all out of bounds pixels are 0
    - extend: all out of bounds pixels = the nearest pixel value
    - wrap: all out of bounds pixels take on a value from the
    image wrapping itself (as if image is infinitely tiled)
    """
    in_bounds = in_range(image, row, col)
    if in_bounds == [True, True]:  # in bounds
        return image["pixels"][get_flat_index(image, row, col)]
    elif boundary_behavior == "zero":
        return 0
    elif boundary_behavior == "wrap":
        if in_bounds[0] is False:  # row is not in bounds
            # wraps row in a 'tile-like' mapping
            row = row % image["height"]
        if in_bounds[1] is False:  # col is not in bounds
            # wraps col in a 'tile-like' mapping
            col = col % image["width"]
        return image["pixels"][get_flat_index(image, row, col)]
    elif boundary_behavior == "extend":
        if row < 0:
            row = 0
        elif row >= image["height"]:
            row = image["height"] - 1
        # else => in bounds, don't change
        if col < 0:
            col = 0
        elif col >= image["width"]:
            col = image["width"] - 1
        # else => in bounds, don't change
        return image["pixels"][get_flat_index(image, row, col)]
    else:
        # error in out_of_bounds string
        return None


def set_pixel(image, row, col, color):
    """
    Sets pixel to value; does not account for edge cases
    """
    image["pixels"][get_flat_index(image, row, col)] = color


def get_flat_index(image, row, col):
    """
    Gets pixel value at row, col; does not account for edge cases
    """
    return row * image["width"] + col


def get_image_loc(image, flat_index):
    """
    Takes in image and flat_index for pixels list
    Returns corresponding row and col
    """
    row = flat_index // image["width"]
    col = flat_index % image["width"]
    return (row, col)


def in_range(image, row, col):
    """
    Returns a two-element list indicating
    whether row, col is in range for the image
    """
    # list representation of whether the row
    # and col are in range of the image bounds
    result = [False, False]
    if 0 <= row < image["height"]:  # in bounds
        result[0] = True
    if 0 <= col < image["width"]:  # in bounds
        result[1] = True
    return result


def apply_per_pixel(image, func, boundary_behavior):
    """
    Iterates through each of the pixels in image, and applies
    the function func, which takes in the pixel's color value

    If this function is used for correlate, then the boundary_behavior
    argument is needed for passing along to get_pixel()

    Does not modify original image; returns new image as described above
    """
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0] * (image["height"] * image["width"]),
    }

    for row in range(image["height"]):
        for col in range(image["width"]):
            # handles row by row (left to right, then down)
            color = get_pixel(image, row, col, boundary_behavior)
            new_color = func(color)
            set_pixel(result, row, col, new_color)
    return result


def inverted(image):
    """
    Inverts image pixel values
    """
    return apply_per_pixel(image, lambda color: 255 - color, "zero")


# HELPER FUNCTIONS


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will be one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    > Kernels are represented by a list of lists,
    where each list correspoonds to a row of the kernel
    > Kernels always have an odd number of rows and columns,
    and an equal amount of rows and columns
    """
    # call apply_per_pixel with function to handle kernel correlation

    # index keeps track of how many times
    # apply_per_pixel calls apply_kernel
    # -> this is how we know what pixel is
    # being analyzed despite only the color
    # being passed (which we want to keep,
    # so that apply_per_pixel can be used
    # with functions that are only color
    # dependent). Although below pixel_color
    # is not actually used, apply_per_pixel
    # already implements the loop to alter
    # each pixel and so I think it's good to use
    # the function anyway

    index = [0]

    # --------------------------------------
    def apply_kernel(_):
        """
        Applies kernel to pixel
        """
        # if index[0] % 10000 == 0:
        #     length = len(image['pixels'])
        #     print(f'{index[0]}/{length}')
        # len(kernel) guaranteed to be odd
        kern_offset = len(kernel) // 2

        # use index, update at end
        coords = get_image_loc(image, index[0])
        start_row = coords[0] - kern_offset
        start_col = coords[1] - kern_offset

        # identifies row, col
        # for top-left corner of kernel
        # selection sub-image; shown below
        # for pixel (1,1) and 3x3 kernel

        # [{0}, 0 , 0, 0, 0
        #   0 ,(0), 0, 0, 0
        #   0 , 0 , 0, 0, 0
        #   0 , 0 , 0, 0, 0
        #   0 , 0 , 0, 0, 0
        #   0 , 0 , 0, 0, 0]

        new_pix_color = 0

        # looks at sub-image window with proper pixels
        # from image; then calculates application of kernel
        for row in range(len(kernel)):
            for col in range(len(kernel[0])):
                new_pix_color += kernel[row][col] * get_pixel(
                    image, start_row + row, start_col + col, boundary_behavior
                )

        index[0] += 1
        return new_pix_color

    # --------------------------------------

    return apply_per_pixel(image, apply_kernel, boundary_behavior)


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    pixels = image["pixels"]

    for i, val in enumerate(pixels):
        pixels[i] = round(val)
        if val < 0:
            pixels[i] = 0
        elif val > 255:
            pixels[i] = 255


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.

    box_blur = box_blur_kernel(kernel_size)
    blurred_image = correlate(image, box_blur, "extend")

    round_and_clip_image(blurred_image)

    return blurred_image


# helper for blurred
def box_blur_kernel(kernel_size):
    """
    Creates n x n kernel with identical values that sum to 1
    kernel_size is used as n
    """
    kernel = []
    element = 1 / (kernel_size**2)

    for _ in range(kernel_size):
        kernel.append([element] * kernel_size)

    return kernel


def sharpened(image, n):
    """
    Takes in an image and returns a new image that has been sharpened
    The operation works by creating a blurred version of the image,
    and then subtracting it from a 2x scaled version of the original image

    Alternatively, I thought that the same function could be implemented by
    correlating the image with a single kernel of size nxn, where each element
    is 2/n**2, but this did not work as expected
    """

    blur_kern = box_blur_kernel(n)
    blur_im = correlate(image, blur_kern, "extend")

    index = [0]

    def sharp_val(color):
        result = 2 * color - blur_im["pixels"][index[0]]
        index[0] += 1
        return result

    result_im = apply_per_pixel(image, sharp_val, "extend")
    round_and_clip_image(result_im)

    return result_im

    # idea for alternate implementation, but does not work right now
    # sharp_kernel = box_blur_kernel(n)
    # for row in range(n):
    #     for col in range(n):
    #         sharp_kernel[row][col] *= 2

    # sharp_image = correlate(image, sharp_kernel, 'extend')
    # round_and_clip_image(sharp_image)
    # return sharp_image


def edges(image):
    """
    Returns a new image (does not modify original) where the edges are emphasized

    Utilizes a sobel operator:
    - Calculates two special correlations with image,
    resulting in two intermediate images
    - Sets each pixel value to the euclidean distance
    between the pair of pixels from the intermediate images
    """
    k_row = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    k_col = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    image_r = correlate(image, k_row, "extend")
    image_c = correlate(image, k_col, "extend")

    index = [0]

    def sobel_operator(_):
        result = math.sqrt(
            image_r["pixels"][index[0]] ** 2 + image_c["pixels"][index[0]] ** 2
        )
        index[0] += 1
        return result

    edged_image = apply_per_pixel(image, sobel_operator, "extend")
    round_and_clip_image(edged_image)

    return edged_image


# -----------------------------------------------------
# Lab 3 functions
# -----------------------------------------------------


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def color_filter(color_image):
        """
        Function that takes in color image and
        applies filter; does not modify original image
        """
        comp_pixels = rgb_split(color_image)

        temp_image = {
            "height": color_image["height"],
            "width": color_image["width"],
            "pixels": comp_pixels[0],
        }
        r_image = filt(temp_image)

        temp_image["pixels"] = comp_pixels[1]
        g_image = filt(temp_image)

        temp_image["pixels"] = comp_pixels[2]
        b_image = filt(temp_image)

        return rgb_combine(r_image, g_image, b_image)

    # check if filter is a greyscale filter here?

    return color_filter


def rgb_split(color_image):
    """
    Takes in a  color image
    Returns a tuple of three pixel lists, each corresponding
    to a color channel from the color image; not aliased
    """

    r_pix = [red_val for (red_val, temp1, temp2) in color_image["pixels"]]
    g_pix = [green_val for (temp1, green_val, temp2) in color_image["pixels"]]
    b_pix = [blue_val for (temp1, temp2, blue_val) in color_image["pixels"]]

    return (r_pix, g_pix, b_pix)


def rgb_combine(r_im, g_im, b_im):
    """
    Takes in three 'greyscale' images, each
    corresponding to a color channel, and
    returns a new image with the all three
    channels (in the format of a 3 element
    tuple per pixel); all three images must have
    same height, width
    """
    assert (
        r_im["height"] == g_im["height"]
        and g_im["height"] == b_im["height"]
        and r_im["width"] == g_im["width"]
        and g_im["width"] == b_im["width"]
    ), "error, different sizes for rgb channel images"

    color_image = {
        "height": r_im["height"],
        "width": r_im["width"],
        "pixels": [
            (r_im["pixels"][i], g_im["pixels"][i], b_im["pixels"][i])
            for i in range(len(r_im["pixels"]))
        ],
    }

    return color_image


def make_blur_filter(kernel_size):
    def blur(image):
        return blurred(image, kernel_size)

    return blur


def make_sharpen_filter(kernel_size):
    def sharpen(image):
        return sharpened(image, kernel_size)

    return sharpen


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.

    Assumes each filter is a color filter
    """

    def combined_filters(color_image):
        """
        Assumes input is color_image; converts all filters to color filters
        """
        output_image = color_image
        for filt in filters:
            output_image = filt(output_image)
        return output_image

    return combined_filters


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    # image has color
    new_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": image["pixels"][:],
    }

    for _ in range(ncols):
        print(f"{_+1}/{ncols}")
        grey_image = greyscale_image_from_color_image(new_image)
        print("1")
        energy = compute_energy(grey_image)
        print("2")
        c_energy_map = cumulative_energy_map(energy)
        print("3")
        seam = minimum_energy_seam(c_energy_map)
        print("4")
        new_image = image_without_seam(new_image, seam)
        print("5")

    return new_image


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    rgb_vals = rgb_split(image)

    grey_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0] * image["height"] * image["width"],
    }

    for i in range(len(grey_image["pixels"])):
        new_val = round(
            0.299 * rgb_vals[0][i] + 0.587 * rgb_vals[1][i] + 0.114 * rgb_vals[2][i]
        )
        grey_image["pixels"][i] = new_val

    return grey_image


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    energy_map = energy["pixels"][:]  # gets initial values

    # adds min val from previous row to each pixel value
    # continuosuly modifies pixels, top to bottom row
    for row in range(1, energy["height"]):
        for col in range(energy["width"]):
            prev_row_ind = get_flat_index(energy, row - 1, col)
            if col == 0:
                min_val = min(energy_map[prev_row_ind : prev_row_ind + 2])
            elif col == energy["width"] - 1:
                min_val = min(energy_map[prev_row_ind - 1 : prev_row_ind + 1])
            else:
                min_val = min(energy_map[prev_row_ind - 1 : prev_row_ind + 2])
            energy_map[get_flat_index(energy, row, col)] += min_val

    return {"height": energy["height"], "width": energy["width"], "pixels": energy_map}


#runs faster, a bit harder to read
# def minimum_energy_seam(cem):
#     """
#     Given a cumulative energy map, returns a list of the indices into the
#     'pixels' list that correspond to pixels contained in the minimum-energy
#     seam (computed as described in the lab 2 writeup).
#     """
#     seam = []
#     curr_col = 0

#     #iterates from bottom to top row
#     for row in range(cem['height'] - 1, -1, -1):
#         # finds column in bottom row with lowest value; start of seam
#         if row == cem['height'] - 1:
#             bottom_pixel_rank = [(col, get_pixel(cem, row, col))\
#                  for col in range(cem['width'])]
#             bottom_pixel_rank.sort(key=lambda x:x[1])
#             min_col = bottom_pixel_rank[0][0]

#             seam.append(get_flat_index(cem, row, min_col))
#             curr_col = min_col # start seam at min_col in bottom row
#         else:
#             in_line_ind = get_flat_index(cem, row, curr_col)

#             # the pixel value for the -1 offset is not defined
#             # for when the row and current col are both 0
#             if row == 0 and curr_col == 0:
#                 left_offset_val = None
#             else:
#                 left_offset_val = cem['pixels'][in_line_ind - 1]

#             pixel_rank = [(-1, left_offset_val), (0, cem['pixels'][in_line_ind]), \
#                 (1, cem['pixels'][in_line_ind + 1])]

#             pixel_rank.sort(key=lambda x:x[1])
#             seam_index = -1

#             if curr_col == 0:
#                 if pixel_rank[0][0] == -1:
#                     seam_index = in_line_ind + pixel_rank[1][0]
#             elif curr_col == cem['width'] - 1:
#                 if pixel_rank[0][0] == 1:
#                     seam_index = in_line_ind + pixel_rank[1][0]

#             if seam_index == -1: #unmodiifed by special cases
#                 seam_index = in_line_ind + pixel_rank[0][0]

#             #update curr_col for next row with correct offset value:
#             curr_col += (seam_index - in_line_ind)
#             seam.append(seam_index)
#     return seam


# runs a bit slower, much easier to read
def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    seam = []
    curr_col = 0
    
    # iterates from bottom to top row
    for row in range(cem["height"] - 1, -1, -1):
        pixel_rank = [(col, get_pixel(cem, row, col)) for col in range(cem["width"])]
        pixel_rank.sort(key=lambda x: x[1])

        if row == cem['height'] - 1:
            curr_col = pixel_rank[0][0]
        else:
            for entry in pixel_rank:
                if entry[0] in range(curr_col - 1, curr_col + 2):
                    # update curr_col for next row
                    curr_col = entry[0]
                    break

        seam.append(get_flat_index(cem, row, curr_col))
    return seam


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    seam.sort(reverse=True)
    new_pixels = image["pixels"][:]

    for index in seam:
        del new_pixels[index]

    return {
        "height": image["height"],
        "width": image["width"] - 1,
        "pixels": new_pixels,
    }

def custom_feature(image, theta):
    """
    Rotates image theta degrees
    Returns new image
    Does not equally for all angles, there is some information loss,
    and for some angles an exception is thrown (angles > 72 or so)
    72 degreees returned best result for small frog
    """

    def re_scale(image, theta):
        """
        Returns new image rescaled for rotation; pixel values of 0
        """
        new_height = round(image['height']*math.cos(theta) + image['width']*math.sin(theta))
        new_width = round(image['height']*math.sin(theta) + image['width']*math.cos(theta))

        return {'height': new_height, 'width': new_width, 'pixels': [(0, 0, 0)]*new_height*new_width}
    
    def get_coordinates(image):
        """
        Returns a parallel list of x, y coordinates, corresponding to each pixel in image
        Origin is at center of image

        row, col -> coordinates
        """
        coords = []
        offset_x = image['width']//2
        offset_y = image['height']//2

        for i in range(len(image['pixels'])):
            row, col = get_image_loc(image, i)
            # x, y
            coords.append((-1*((image['width'] - col - 1) - offset_x), (image['height'] - row - 1) - offset_y))
        return coords
    
    def get_index(image, context_image, coord):
        """
        Returns the flat index given an x,y coordinate,
        translated from a given image to the context of a new, context image
        Context image is guaranteed to be bigger than or equal to image, height and width
        Original image origin is at center

        coordinates -> row, col in context image -> flat_index w/ function call
        """
        offset_x = image['width']//2
        offset_y = image['height']//2

        context_offset_x = (context_image['width'] - image['width'])//2
        context_offset_y = (context_image['height'] - image['height'])//2

        col = coord[0] - offset_x + image['width'] - 1 + context_offset_x
        row = -1*coord[1] - offset_y + image['height'] - 1 + context_offset_y
        index = get_flat_index(image, row, col)
        return index
    
    def matrix_mult_2D(matrix, vector, estimate=False):
        """
        Multiples a 2D matrix by a 2D vector, returns resulting 2D vector
        Matrix should be a flat list, vector a tuple, corresponding to row major order
        """
        assert len(matrix) == 4 and len(vector) == 2, "error, passed in matrix or vector is not 2D"

        x = matrix[0]*vector[0] + matrix[1]*vector[1]
        y = matrix[2]*vector[0] + matrix[3]*vector[1]

        if estimate is True:
            x = round(x)
            y = round(y)
        return (x, y)
    
    def add_row_col(image, add_row=False, add_col=False):
        """
        Mutates image, adding either a col or row, or both
        of black pixels
        """
        black = (0, 0, 0)

        if add_row is True:
            image['height'] += 1
            image['pixels'] += [black]*image['width']
        if add_col is True:
            new_pixels = []
            for i in range(image['height']):
                new_pixels += image['pixels'][i*image['width']:(i+1)*image['width']]
                new_pixels += [black]
            image['width'] += 1
            image['pixels'] = new_pixels
    
    # theta = theta*math.pi/180 # converts degrees to radians
    # add_row, add_col = False, False

    # if image['height'] % 2 == 0:
    #     add_row = True
    # if image['width'] % 2 == 0:
    #     add_col = True
    
    new_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'][:]
    }
    # add_row_col(padded_image, add_row, add_col)

    # resized_image = re_scale(padded_image, theta)
    # coords = get_coordinates(padded_image)
    coords = get_coordinates(new_image)

    # 720 x 1280
    for i, val in enumerate(new_image['pixels']):
        matrix = [math.cos(theta), -1*math.sin(theta), math.sin(theta), math.cos(theta)]
        #matrix = [1272/1280, 0.01, 0.01, 707/720]
        new_coord = matrix_mult_2D(matrix, coords[i], estimate=True)
        #dimensions = (new_image['height'], new_image['width'])
        #print(f'dimensions: {dimensions}')
        length = len(new_image['pixels'])
        print(f'{i} out of {length}')
        new_image['pixels'][get_index(new_image, image, new_coord)] = val
    return new_image

# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.

    # original_image = load_color_image('test_results/pattern_carved.png')
    # new_image = seam_carving(original_image, 3)
    # save_color_image(new_image, 'test_results/test12345.png')

    xavier_im = load_color_image("test_images/Xavier.png")
    new_im = custom_feature(xavier_im, 100)
    save_color_image(new_im, "test_results/Xavier_interesting3.png")

    # in_image = load_color_image("test_images/smallfrog.png")
    # out_image = custom_feature(in_image, 72)
    # save_color_image(out_image, "test_results/smallfrog_rotated.png")

    # in_image = load_color_image("test_results/pattern_carved.png")
    # out_image = custom_feature(in_image, 60)
    # save_color_image(out_image, "test_results/rotated_pattern.png")