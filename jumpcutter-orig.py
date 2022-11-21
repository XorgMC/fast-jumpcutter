from contextlib import closing
from PIL import Image
import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree
import os
import argparse
from pytube import YouTube


def download_file(url):
    name = YouTube(url).streams.first().download()
    newname = name.replace(' ', '_')
    os.rename(name, newname)
    return newname


def get_max_volume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv, -minv)


def copy_frame(input_frame, output_frame):
    src = TEMP_FOLDER + "/frame{:06d}".format(input_frame + 1) + ".jpg"
    dst = TEMP_FOLDER + "/newFrame{:06d}".format(output_frame + 1) + ".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if output_frame % 20 == 19:
        print(str(output_frame + 1) + " time-altered frames saved.")
    return True


def input_to_output_filename(filename):
    dot_index = filename.rfind(".")
    return filename[:dot_index] + "_ALTERED" + filename[dot_index:]


def create_path(s):
    # assert (not os.path.exists(s)), "The filepath "+s+" already exists. Don't want to overwrite it. Aborting."

    try:
        if not os.path.exists(s):
            os.mkdir(s)
    except OSError:
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, " \
                      "and try again.) "


def delete_path(s):  # Dangerous! Watch out!
    try:
        rmtree(s, ignore_errors=False)
    except OSError:
        print("Deletion of the directory %s failed" % s)
        print(OSError)


parser = argparse.ArgumentParser(
    description='Modifies a video file to play at different speeds when there is sound vs. silence.')
parser.add_argument('--input_file', type=str, help='the video file you want modified')
parser.add_argument('--url', type=str, help='A youtube url to download and process')
parser.add_argument('--output_file', type=str, default="",
                    help="the output file. (optional. if not included, it'll just modify the input file name)")
parser.add_argument('--silent_threshold', type=float, default=0.03,
                    help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges "
                         "from 0 (silence) to 1 (max volume)")
parser.add_argument('--sounded_speed', type=float, default=1.00,
                    help="the speed that sounded (spoken) frames should be played at. Typically 1.")
parser.add_argument('--silent_speed', type=float, default=5.00,
                    help="the speed that silent frames should be played at. 999999 for jumpcutting.")
parser.add_argument('--frame_margin', type=float, default=1,
                    help="some silent frames adjacent to sounded frames are included to provide context. How many "
                         "frames on either the side of speech should be included? That's this variable.")
parser.add_argument('--sample_rate', type=int, default=44100, help="sample rate of the input and output videos")
parser.add_argument('--frame_rate', type=float, default=30,
                    help="frame rate of the input and output videos. optional... I try to find it out myself, "
                         "but it doesn't always work.")
parser.add_argument('--frame_quality', type=int, default=3,
                    help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the "
                         "default.")

args = parser.parse_args()

frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [args.silent_speed, args.sounded_speed]
if args.url is not None:
    INPUT_FILE = download_file(args.url)
else:
    INPUT_FILE = args.input_file
URL = args.url
FRAME_QUALITY = args.frame_quality

assert INPUT_FILE is not None, "why u put no input file, that dum"

if len(args.output_file) >= 1:
    OUTPUT_FILE = args.output_file
else:
    OUTPUT_FILE = input_to_output_filename(INPUT_FILE)

TEMP_FOLDER = "TEMP"
AUDIO_FADE_ENVELOPE_SIZE = 400  # smooth out audio by quickly fading in/out (arbitrary magic number whatever)

create_path(TEMP_FOLDER)

subprocess.call("ffmpeg -i " + INPUT_FILE + " -qscale:v " + str(FRAME_QUALITY) +
                " " + TEMP_FOLDER + "/frame%06d.jpg -hide_banner", shell=True)

subprocess.call("ffmpeg -i " + INPUT_FILE + " -ab 160k -ac 2 -ar " + str(SAMPLE_RATE) + " -vn " +
                TEMP_FOLDER + "/audio.wav", shell=True)

f = open(TEMP_FOLDER + "/params.txt", "w")
subprocess.call("ffmpeg -i " + TEMP_FOLDER + "/input.mp4 2>&1", shell=True, stdout=f)

sampleRate, audioData = wavfile.read(TEMP_FOLDER + "/audio.wav")
audioSampleCount = audioData.shape[0]
maxAudioVolume = get_max_volume(audioData)

f = open(TEMP_FOLDER + "/params.txt", 'r+')
pre_params = f.read()
f.close()
params = pre_params.split('\n')
for line in params:
    m = re.search('Stream #.*Video.* ([0-9]*) fps', line)
    if m is not None:
        frameRate = float(m.group(1))

samplesPerFrame = sampleRate / frameRate

audioFrameCount = int(math.ceil(audioSampleCount / samplesPerFrame))

hasLoudAudio = np.zeros(audioFrameCount)

for i in range(audioFrameCount):
    start = int(i * samplesPerFrame)
    end = min(int((i + 1) * samplesPerFrame), audioSampleCount)
    audiochunks = audioData[start:end]
    maxchunksVolume = float(get_max_volume(audiochunks)) / maxAudioVolume
    if maxchunksVolume >= SILENT_THRESHOLD:
        hasLoudAudio[i] = 1

chunks = [[0, 0, 0]]
shouldIncludeFrame = np.zeros(audioFrameCount)
for i in range(audioFrameCount):
    start = int(max(0, i - FRAME_SPREADAGE))
    end = int(min(audioFrameCount, i + 1 + FRAME_SPREADAGE))
    shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
    if i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i - 1]:  # Did we flip?
        chunks.append([chunks[-1][1], i, shouldIncludeFrame[i - 1]])

chunks.append([chunks[-1][1], audioFrameCount, shouldIncludeFrame[i - 1]])
chunks = chunks[1:]

outputAudioData = np.zeros((0, audioData.shape[1]))
output_pointer = 0

last_existing_frame = None
for chunk in chunks:
    audio_chunk = audioData[int(chunk[0] * samplesPerFrame):int(chunk[1] * samplesPerFrame)]

    s_file = TEMP_FOLDER + "/tempStart.wav"
    e_file = TEMP_FOLDER + "/tempEnd.wav"
    wavfile.write(s_file, SAMPLE_RATE, audio_chunk)
    with WavReader(s_file) as reader:
        with WavWriter(e_file, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
            tsm.run(reader, writer)
    _, altered_audio_data = wavfile.read(e_file)
    leng = altered_audio_data.shape[0]
    end_pointer = output_pointer + leng
    outputAudioData = np.concatenate((outputAudioData, altered_audio_data / maxAudioVolume))

    # outputAudioData[output_pointer:end_pointer] = altered_audio_data/maxAudioVolume

    # smooth out transitiion's audio by quickly fading in/out

    if leng < AUDIO_FADE_ENVELOPE_SIZE:
        outputAudioData[output_pointer:end_pointer] = 0  # audio is less than 0.01 sec, let's just remove it.
    else:
        premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE) / AUDIO_FADE_ENVELOPE_SIZE
        mask = np.repeat(premask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
        outputAudioData[output_pointer:output_pointer + AUDIO_FADE_ENVELOPE_SIZE] *= mask
        outputAudioData[end_pointer - AUDIO_FADE_ENVELOPE_SIZE:end_pointer] *= 1 - mask

    start_output_frame = int(math.ceil(output_pointer / samplesPerFrame))
    end_output_frame = int(math.ceil(end_pointer / samplesPerFrame))
    for output_frame in range(start_output_frame, end_output_frame):
        input_frame = int(chunk[0] + NEW_SPEED[int(chunk[2])] * (output_frame - start_output_frame))
        did_it_work = copy_frame(input_frame, output_frame)
        if did_it_work:
            last_existing_frame = input_frame
        else:
            copy_frame(last_existing_frame, output_frame)

    output_pointer = end_pointer

wavfile.write(TEMP_FOLDER + "/audioNew.wav", SAMPLE_RATE, outputAudioData)

'''
output_frame = math.ceil(output_pointer/samplesPerFrame)
for endGap in range(output_frame,audioFrameCount):
    copy_frame(int(audioSampleCount/samplesPerFrame)-1,endGap)
'''

subprocess.call("ffmpeg -framerate " + str(frameRate) + " -i " + TEMP_FOLDER + "/newFrame%06d.jpg -i " +
                TEMP_FOLDER + "/audioNew.wav -strict -2 " + OUTPUT_FILE, shell=True)

delete_path(TEMP_FOLDER)
