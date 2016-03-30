import wave
import pywt
import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import linregress

def read_wav_file(filename):
    """
    :param filename: a 16-bit uncompressed .wav file
    :returns: parsed values in a numpy array, and the sampling frequency of the file (ex. 44100Hz)
    """
    f = wave.open(filename, 'r')
    f.readframes(f.getframerate())  # skip the first second
    # read up to 10 minutes
    ints = [ord(x) for x in f.readframes(600*f.getframerate())]
    # combine adjacent bytes, and convert to signed numbers
    vals = np.array([ints[i] + 256*ints[i+1] for i in xrange(0, len(ints)-1, 2)])
    vals[vals >= 32768] -= 65536

    return vals, f.getframerate()

def find_ticks(data, expected_period):
    """Extract tick times from the given preprocessed data."""
    # process intervals shorter than the period, so we're sure never to catch two ticks at once
    sample_interval = int(0.9*expected_period)
    ticks = []
    for i in xrange(0, len(data) - sample_interval, sample_interval):
        cur = data[i: i + sample_interval]
        idx = np.argmax(cur)
        maxval = cur[idx]
        mean = np.mean(cur)

        if maxval/mean > 3:  # reject intervals with no ticks or too much noise
            ticks.append(i + idx)
    return np.array(ticks)

def find_period(ticks, expected_period):
    """Given the times of a bunch of ticks, estimate the average period between them. This is done
    in a rather inefficient way, by brute force searching through possible periods."""
    best_slope = 1
    best_p = 1
    for p in np.linspace(expected_period - 5, expected_period + 5, 2000):
        slope, intercept, rval, pval, stderr = linregress(ticks/p, ticks % p)
        if stderr + abs(slope) < abs(best_slope):  # search for a fit with zero slope and low error
            best_slope = stderr + abs(slope)
            best_p = p
    return best_p

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure the accuracy of a watch.')
    parser.add_argument('audio_file', help='16-bit uncompressed wav file to analyze')
    parser.add_argument('beat_freq', type=int, help='watch frequency in bph')

    args = parser.parse_args()

    tick_rate = float(args.beat_freq)/3600  # convert to bps
    data, framerate = read_wav_file(args.audio_file)
    expected_period = framerate/tick_rate

    ca, cd = pywt.dwt(data, 'db4')  # extract wavelet coefficients
    cd = gaussian_filter1d(np.abs(cd), 20)  # smoothing seems to help

    ticks = find_ticks(cd, expected_period)

    period = find_period(ticks, expected_period)

    print round((1-period/expected_period)*3600*24, 1), 's/day'
