"""
6.1010 Spring '23 Lab 0: Audio Processing
"""

import wave
import struct

# No additional imports allowed!


def backwards(sound):
    """
    Takes in a sound in the following format:
    {'rate': sample rate, 'samples': list of samples}

    and returns a new sound with the same sample rate,
    but with the samples reversed
    """
    reverse = sound["samples"][::-1]

    new_sound = {"rate": sound["rate"], "samples": reverse}
    return new_sound


def mix(sound1, sound2, p):
    """
    Mixes two sounds with the same processing rate by
    taking a percentage (p) of each sample in sound1 and
    adding to it the complementary percentage (1-p) of each
    sample in sound2

    returns a new sound

    if sampling rate is different, returns None
    """
    # mix 2 good sounds
    if (
        "rate" not in sound1.keys()
        or "rate" not in sound2.keys()
        or sound1["rate"] != sound2["rate"]
    ):
        print("no rate, or else rates are not equal")
        return None

    new_rate = sound1["rate"]  # get rate
    new_sound = None

    # mono
    if "samples" in sound1.keys():
        sound1 = sound1["samples"]
        sound2 = sound2["samples"]

        if len(sound1) <= len(sound2):
            size = len(sound1)
        else:
            size = len(sound2)

        new_samples = []
        for x in range(size):
            s1 = sound1[x] * p
            s2 = sound2[x] * (1 - p)
            new_samples.append(s1 + s2)  # add sounds

        new_sound = {"rate": new_rate, "samples": new_samples}
    else:  # stereo
        left_mix = mix({"rate": new_rate, "samples": sound1["left"]},\
            {"rate": new_rate, "samples": sound2["left"]}, p)["samples"]

        right_mix = mix({"rate": new_rate, "samples": sound1["right"]},\
            {"rate": new_rate, "samples": sound2["right"]}, p)["samples"]

        new_sound = {"rate": new_rate, "left": left_mix, "right": right_mix}

    return new_sound  # return new sound


def convolve(sound, kernel):
    """
    Applies a filter to a sound, resulting in a new sound that is longer than
    the original mono sound by the length of the kernel - 1.
    Does not modify inputs.

    Args:
        sound: A mono sound dictionary with two key/value pairs:
            * "rate": an int representing the sampling rate, samples per second
            * "samples": a list of floats containing the sampled values
        kernel: A list of numbers

    Returns:
        A new mono sound dictionary.
    """
    samples = []  # a list of scaled sample lists

    for i, scale in enumerate(kernel):
        if scale != 0:
            scaled_sample = [0] * i  # offset scaled sound by filter index
            scaled_sample += [scale * x for x in sound["samples"]]
            samples.append(scaled_sample)

    # combine samples into one list
    # initializes final_sample to a list of zeros of the proper size
    final_sample = [0 for i in range(len(sound["samples"]) + len(kernel) - 1)]

    for sample in samples:
        for i, data in enumerate(sample):
            # sample len will never be longer than target len
            final_sample[i] += data

    return {"rate": sound["rate"], "samples": final_sample}


def echo(sound, num_echoes, delay, scale):
    """
    Compute a new signal consisting of several scaled-down and delayed versions
    of the input sound. Does not modify input sound.

    Args:
        sound: a dictionary representing the original mono sound
        num_echoes: int, the number of additional copies of the sound to add
        delay: float, the amount of seconds each echo should be delayed
        scale: float, the amount by which each echo's samples should be scaled

    Returns:
        A new mono sound dictionary resulting from applying the echo effect.
    """
    delay_n = round(delay * sound["rate"])

    # changed -1 to +1 in order to have proper sized convolution filter (echo_filter)
    echo_filter = [0] * (delay_n * num_echoes + 1)

    # changed to start from index 0
    for i in range(num_echoes + 1):
        # spaces filter values to properly delay echo
        offset = i * delay_n
        # filter values increasingly scale-down
        # input sound with successive echoes
        echo_filter[offset] = scale**i

    return convolve(sound, echo_filter)


def pan(sound, reverse=False):
    """
    Given a stereo sound, returns a new
    stereo sound with a pan effect:

    - right channel samples are
    successively scaled from 0 to 1
    - left channel samples are scaled
    by the complement, from 1 to 0
    """
    assert len(sound) == 3, "sound should be stereo sound"

    left = sound["left"][:]
    right = sound["right"][:]
    size = len(right)

    for i in range(size):
        scale = i / (size - 1)
        right[i] *= scale
        left[i] *= (1 - scale)

    if reverse:
        left, right = right, left

    return {"rate": sound["rate"], "left": left, "right": right}


def remove_vocals(sound):
    """
    Given a stereo sound, often removes
    vocals and returns a new mono sound

    - channel samples are successively
    subtracted to define a new set of samples
    """
    assert len(sound) == 3, "sound should be stereo sound"

    left = sound["left"][:]
    right = sound["right"][:]
    new_samples = []

    assert len(left) == len(right), "left and right samples should be same length"

    # I do not use enumerate below because of readability:
    # I would be subtracting right[i] from an enumerated value of left
    for i in range(len(left)):
        new_samples.append(left[i] - right[i])

    return {"rate": sound["rate"], "samples": new_samples}


# below are helper functions for converting back-and-forth between WAV files
# and our internal dictionary representation for sounds


def bass_boost_kernel(boost, scale=0):
    """
    Constructs a kernel that acts as a bass-boost filter.

    We start by making a low-pass filter, whose frequency response is given by
    (1/2 + 1/2cos(Omega)) ^ N

    Then we scale that piece up and add a copy of the original signal back in.

    Args:
        boost: an int that controls the frequencies that are boosted (0 will
            boost all frequencies roughly equally, and larger values allow more
            focus on the lowest frequencies in the input sound).
        scale: a float, default value of 0 means no boosting at all, and larger
            values boost the low-frequency content more);

    Returns:
        A list of floats representing a bass boost kernel.
    """
    # make this a fake "sound" so that we can use the convolve function
    base = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    kernel = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    for i in range(boost):
        kernel = convolve(kernel, base["samples"])
    kernel = kernel["samples"]

    # at this point, the kernel will be acting as a low-pass filter, so we
    # scale up the values by the given scale, and add in a value in the middle
    # to get a (delayed) copy of the original
    kernel = [i * scale for i in kernel]
    kernel[len(kernel) // 2] += 1

    return kernel


def load_wav(filename, stereo=False):
    """
    Load a file and return a sound dictionary.

    Args:
        filename: string ending in '.wav' representing the sound file
        stereo: bool, by default sound is loaded as mono, if True sound will
            have left and right stereo channels.

    Returns:
        A dictionary representing that sound.
    """
    sound_file = wave.open(filename, "r")
    chan, bd, sr, count, _, _ = sound_file.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    out = {"rate": sr}

    left = []
    right = []
    for i in range(count):
        frame = sound_file.readframes(1)
        if chan == 2:
            left.append(struct.unpack("<h", frame[:2])[0])
            right.append(struct.unpack("<h", frame[2:])[0])
        else:
            datum = struct.unpack("<h", frame)[0]
            left.append(datum)
            right.append(datum)

    if stereo:
        out["left"] = [i / (2**15) for i in left]
        out["right"] = [i / (2**15) for i in right]
    else:
        samples = [(ls + rs) / 2 for ls, rs in zip(left, right)]
        out["samples"] = [i / (2**15) for i in samples]

    return out


def write_wav(sound, filename):
    """
    Save sound to filename location in a WAV format.

    Args:
        sound: a mono or stereo sound dictionary
        filename: a string ending in .WAV representing the file location to
            save the sound in
    """
    outfile = wave.open(filename, "w")

    if "samples" in sound:
        # mono file
        outfile.setparams((1, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = [int(max(-1, min(1, v)) * (2**15 - 1)) for v in sound["samples"]]
    else:
        # stereo
        outfile.setparams((2, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = []
        for l_val, r_val in zip(sound["left"], sound["right"]):
            l_val = int(max(-1, min(1, l_val)) * (2**15 - 1))
            r_val = int(max(-1, min(1, r_val)) * (2**15 - 1))
            out.append(l_val)
            out.append(r_val)

    outfile.writeframes(b"".join(struct.pack("<h", frame) for frame in out))
    outfile.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place to put your
    # code for generating and saving sounds, or any other code you write for
    # testing, etc.

    # here is an example of loading a file (note that this is specified as
    # sounds/hello.wav, rather than just as hello.wav, to account for the
    # sound files being in a different directory than this file)

    # hello = load_wav("sounds/hello.wav")
    # write_wav(backwards(hello), "hello_reversed.wav")

    # converts .wav file to sound
    # mystery = load_wav("sounds/mystery.wav")
    # write_wav(backwards(mystery), "mystery_reversed.wav")

    # synth = load_wav("sounds/synth.wav")
    # water = load_wav("sounds/water.wav")
    # write_wav(mix(synth, water, 0.2), "my_mix.wav")

    # bass_kernel = bass_boost_kernel(1000, 15)
    # chill_music = load_wav("sounds/ice_and_chilli.wav")
    # write_wav(convolve(chill_music, bass_kernel), "super_bass_ice_chilli.wav")

    # chord = load_wav("sounds/chord.wav")
    # write_wav(echo(chord, 5, 0.8, 0.4), "TEST_echo_chord.wav")

    chord = load_wav("sounds/chord.wav", stereo=True)
    write_wav(pan(chord, reverse=True), "TEST_pan_chord.wav")
