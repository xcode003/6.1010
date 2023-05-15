"""
6.1010 Spring '23 Lab 1: Image Processing
"""

#!/usr/bin/env python3

import math

from PIL import Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, row, col, boundary_behavior):
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
    '''
    Returns a two-element list indicating
    whether row, col is in range for the image
    '''
    # list representation of whether the row
    # and col are in range of the image bounds
    result = [False, False]
    if row >= 0 and row < image["height"]:  # in bounds
        result[0] = True
    if col >= 0 and col < image["width"]:  # in bounds
        result[1] = True
    return result


def apply_per_pixel(image, func, boundary_behavior):
    '''
    Iterates through each of the pixels in image, and applies
    the function func, which takes in the pixel's color value

    If this function is used for correlate, then the boundary_behavior
    argument is needed for passing along to get_pixel()

    Does not modify original image; returns new image as described above
    '''
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
    '''
    Inverts image pixel values
    '''
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
    # each pixel

    index = [0]

    # --------------------------------------
    def apply_kernel(pixel_color, ref_im=image):
        """
        Applies kernel to pixel
        """
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
                    ref_im, start_row + row, start_col + col, boundary_behavior
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


# FILTERS
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


def box_blur_kernel(kernel_size):
    """
    Creates n x n kernel with identical values that sum to 1
    kernel_size is used as n
    """
    kernel = []
    element = 1 / (kernel_size**2)

    for i in range(kernel_size):
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
    '''
    Returns a new image (does not modify original) where the edges are emphasized

    Utilizes a sobel operator:
    - Calculates two special correlations with image,
    resulting in two intermediate images
    - Sets each pixel value to the euclidean distance
    between the pair of pixels from the intermediate images
    '''
    k_row = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    k_col = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    image_r = correlate(image, k_row, "extend")
    image_c = correlate(image, k_col, "extend")

    index = [0]

    def sobel_operator(pixel_color):
        result = math.sqrt(
            image_r["pixels"][index[0]] ** 2 + image_c["pixels"][index[0]] ** 2
        )
        index[0] += 1
        return result

    # image does not need to be passed, but it is a
    # good frame for the apply_per_pixel function
    edged_image = apply_per_pixel(image, sobel_operator, "extend")
    round_and_clip_image(edged_image)

    return edged_image


# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES
def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
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
    by the "mode" parameter.
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

    # image = {
    #     "height": 4,
    #     "width": 4,
    #     "pixels": [1, 3, 5, 7, 2, 4, 6, 8, 4, 6, 8, 0, 9, 1, 3, 5],
    # }

    # #prints image pixels in row major format
    # for i in range(image['height']):
    #     print(image['pixels'][i*image['width']:(i+1)*image['width']])
    # print()

    # #scale by 3 in each dimension
    # result = {
    #     'height': image['height']*3,
    #     'width': image['width']*3,
    #     'pixels': [0]*image['height']*image['width']*3*3
    # }

    # for row in range(result['height']):
    #     for col in range(result['width']):
    #         result['pixels'][get_flat_index(result, row, col)]\
    #             = get_pixel(image, row - 4, col - 4, 'extend')

    # #prints image pixels in row major format
    # for i in range(result['height']):
    #     print(result['pixels'][i*result['width']:(i+1)*result['width']])

    # kernel = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]

    test_image = load_greyscale_image("test_images/construct.png")
    save_greyscale_image(edges(test_image), "test_results/edges_construct.png")
