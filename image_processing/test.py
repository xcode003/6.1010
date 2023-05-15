#!/usr/bin/env python3

import os
import pickle
import hashlib

import lab
import pytest

TEST_DIRECTORY = os.path.dirname(__file__)


def object_hash(x):
    return hashlib.sha512(pickle.dumps(x)).hexdigest()


def compare_images(im1, im2):
    assert set(im1.keys()) == {'height', 'width', 'pixels'}, 'Incorrect keys in dictionary'
    assert im1['height'] == im2['height'], 'Heights must match'
    assert im1['width'] == im2['width'], 'Widths must match'
    assert len(im1['pixels']) == im1['height']*im1['width'], 'Incorrect number of pixels'
    assert all(isinstance(i, int) for i in im1['pixels']), 'Pixels must all be integers'
    assert all(0<=i<=255 for i in im1['pixels']), 'Pixels must all be in the range from [0, 255]'
    pix_incorrect = (None, None)
    for ix, (i, j) in enumerate(zip(im1['pixels'], im2['pixels'])):
        if i != j:
            pix_incorrect = (ix, abs(i-j))
    assert pix_incorrect == (None, None), 'Pixels must match.  Incorrect value at location %s (differs from expected by %s)' % pix_incorrect



def test_load():
    result = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    expected = {
        'height': 11,
        'width': 11,
        'pixels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    compare_images(result, expected)


def test_inverted_1():
    im = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    result = lab.inverted(im)
    expected = {
        'height': 11,
        'width': 11,
        'pixels': [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    }
    compare_images(result, expected)

def test_inverted_2():
    im = {
        'height': 1,
        'width': 4,
        'pixels': [26, 88, 130, 219]
    }
    result = lab.inverted(im)
    expected = {
        'height': 1,
        'width': 4,
        'pixels': [229, 167, 125, 36]
    }
    compare_images(result, expected)

# does not assert correctly
def test_flat_index():
    im = {
        'height': 5,
        'width': 5,
        'pixels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    }
    index = lab.get_flat_index(im, 3, 4)
    assert index == 19, 'incorrect index'

@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_inverted_images(fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_invert.png' % fname)
    im = lab.load_greyscale_image(inpfile)
    oim = object_hash(im)
    result = lab.inverted(im)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(im) == oim, 'Be careful not to modify the original image!'
    compare_images(result, expected)


@pytest.mark.parametrize("kernsize", [1, 3, 7])
@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_blurred_images(kernsize, fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_blur_%02d.png' % (fname, kernsize))
    input_img = lab.load_greyscale_image(inpfile)
    input_hash = object_hash(input_img)
    result = lab.blurred(input_img, kernsize)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(input_img) == input_hash, "Be careful not to modify the original image!"
    compare_images(result, expected)

def test_blurred_black_image():
    image = {
        'height': 6,
        'width': 5,
        'pixels': [255]*30
    }

    blurred_n_5 = lab.blurred(image, 5)
    blurred_n_7 = lab.blurred(image, 7)

    compare_images(image, blurred_n_5)
    compare_images(image, blurred_n_7)

def test_blurred_centered_pixel():
    centered_pixel = lab.load_greyscale_image("test_images/centered_pixel.png")

    # output for kernel size 5 vs 7 is the same, but the two separate
    # definitions allows for changing the test case to something else, later
    expected_blurred_n_5 = {
        'height': centered_pixel["height"],
        'width': centered_pixel["width"],
        'pixels': [0]*centered_pixel["height"]*centered_pixel["width"]
    }
    # height and width of picture and kernel should be odd in this test case
    start_row = (expected_blurred_n_5['height'] - 5)//2
    start_col = (expected_blurred_n_5['width'] - 5)//2
    for row in range(5):
        for col in range(5):
            expected_blurred_n_5['pixels'][lab.get_flat_index(expected_blurred_n_5, start_row + row, start_col + col)] = 10


    expected_blurred_n_7 = {
        'height': centered_pixel["height"],
        'width': centered_pixel["width"],
        'pixels': [0]*centered_pixel["height"]*centered_pixel["width"]
    }
    # height and width of picture and kernel should be odd in this test case
    start_row = (expected_blurred_n_7['height'] - 7)//2
    start_col = (expected_blurred_n_7['width'] - 7)//2
    for row in range(7):
        for col in range(7):
            expected_blurred_n_7['pixels'][lab.get_flat_index(expected_blurred_n_7, start_row + row, start_col + col)] = 5

    # calling blur function on image
    blurred_n_5 = lab.blurred(centered_pixel, 5)
    blurred_n_7 = lab.blurred(centered_pixel, 7)

    print(blurred_n_5, expected_blurred_n_5)
    print(blurred_n_7, expected_blurred_n_7)

    # compare function output to expected output
    compare_images(expected_blurred_n_5, blurred_n_5)
    compare_images(expected_blurred_n_7, blurred_n_7)

@pytest.mark.parametrize("kernsize", [1, 3, 9])
@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_sharpened_images(kernsize, fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_sharp_%02d.png' % (fname, kernsize))
    input_img = lab.load_greyscale_image(inpfile)
    input_hash = object_hash(input_img)
    result = lab.sharpened(input_img, kernsize)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(input_img) == input_hash, "Be careful not to modify the original image!"
    compare_images(result, expected)


@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_edges_images(fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_edges.png' % fname)
    input_img = lab.load_greyscale_image(inpfile)
    input_hash = object_hash(input_img)
    result = lab.edges(input_img)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(input_img) == input_hash, "Be careful not to modify the original image!"
    compare_images(result, expected)

def test_edges_centered_pixel():
    centered_pixel = lab.load_greyscale_image('test_images/centered_pixel.png')
    start_row = (centered_pixel['height'] - 3)//2
    start_col = (centered_pixel['width'] - 3)//2

    expect_out = {
        'height': centered_pixel['height'],
        'width': centered_pixel['width'],
        'pixels': [0]*centered_pixel['height']*centered_pixel['width']
    }

    for row in range(3):
        for col in range(3):
            if row != 1 or col != 1: # gives pixels value 255, other than central pixel
                expect_out['pixels'][lab.get_flat_index(expect_out, start_row + row, start_col + col)] = 255
    
    edged_c_p = lab.edges(centered_pixel)

    compare_images(edged_c_p, expect_out)


if __name__ == "__main__":
    import sys
    res = pytest.main(["-k", " or ".join(sys.argv[1:]), "-v", __file__])
