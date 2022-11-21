from contextlib import closing
from PIL import Image
import subprocess, time, threading
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree, move
import os
import argparse
from pytube import YouTube

outputPointer = 0
chunksDone = 0
lastExistingFrame = None

copiedFiles = {}

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def downloadFile(url):
    name = YouTube(url).streams.first().download()
    newname = name.replace(' ','_')
    os.rename(name,newname)
    return newname

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def copyFrameSafe(inputFrame,outputFrame):
    src = TEMP_FOLDER+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = TEMP_FOLDER+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if outputFrame%20 == 19:
        print(str(outputFrame+1)+" time-altered frames saved.")
    return True

def copyFrameClean(inputFrame,outputFrame):
    src = TEMP_FOLDER+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = TEMP_FOLDER+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if src in copiedFiles.keys():
        copyfile(copiedFiles[src], dst)
    else:
        if not os.path.isfile(src):
            if(int(re.findall(r'\d+', max(copiedFiles.keys()))[-1]) > int(re.findall(r'\d+', src)[-1])):
                print("\n**** ERROR: File", src, "not found and not already copied! ****\n")
                exit(1)
            return False
        copiedFiles[src] = dst
        move(src, dst)
    return True

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def createPath(s):
    try:  
        if not os.path.exists(s):
           os.mkdir(s)
    except OSError:  
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"

def deletePath(s): # Dangerous! Watch out!
    try:  
        rmtree(s,ignore_errors=False)
    except OSError:  
        print ("Deletion of the directory %s failed" % s)
        print(OSError)

parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-i', '--input-file', dest='input_file', type=str,  help='the video file you want modified')
group.add_argument('-u', '--url', dest='url', type=str, help='A youtube url to download and process')
parser.add_argument('-o', '--output_file', dest='output_file', type=str, default="", help="Output Filename (default: input_ALTERED.ext)")
parser.add_argument('-st', '--silent-threshold', dest='silent_threshold', type=float, default=0.03, help="Minimum volume to be considered \"sounded\". (Min: 0, Max: 1, default: 0.03)")
parser.add_argument('-vs', '--voice-speed', dest='sounded_speed', type=float, default=1.00, help="Speed sounded frames should be played at. (default: 1.00)")
parser.add_argument('-ss', '--silent-speed', dest='silent_speed', type=float, default=5.00, help="Speed silent frames should be played at. (default 5.00, 999999 for jumpcutting.)")
parser.add_argument('-fm', '--frame-margin', dest='frame_margin', type=float, default=1, help="Include x adjacent frames around sounded segments. (default: 1)")
parser.add_argument('-ar', '--sample-rate', dest='sample_rate', type=int, default=-1, help="Sample rate of input and output. (default: -1 =autodetect)")
parser.add_argument('-r', '--frame-rate', dest='frame_rate', type=float, default=-1, help="Frame rate of input and output. (default: -1 =autodetect)")
parser.add_argument('-fq', '--frame-quality', dest='frame_quality', type=int, default=3, help="Frame image quality. (best: 1, worst: 31, default: 3)")
parser.add_argument('-sc', '--safe-copy', action='store_true', dest='scopy', help="Use old, safe copy method. Uses much more disk space.")
parser.add_argument('-d', '--display-output', action='store_true', dest='displ', help="Open output file after jumpcutting")
parser.add_argument('-pr', '--pre-fps', dest='prefps', help="Change FPS before jumpcutting")

args = parser.parse_args()

if(args.scopy):
    print("Using safe frame copy method")
    copyFrame = copyFrameSafe
else:
    copyFrame = copyFrameClean

frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [args.silent_speed, args.sounded_speed]
if args.url != None:
    INPUT_FILE = downloadFile(args.url)
else:
    INPUT_FILE = args.input_file
URL = args.url
FRAME_QUALITY = args.frame_quality

assert INPUT_FILE != None , "why u put no input file, that dum"
    
if len(args.output_file) >= 1:
    OUTPUT_FILE = args.output_file
else:
    OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)

TEMP_FOLDER = "TEMP"
AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)

createPath(TEMP_FOLDER)

if(args.prefps != None):
    print("Pre-converting file to", args.prefps, "FPS...")
    subprocess.call("ffmpeg -hide_banner -loglevel error -init_hw_device qsv=hw -filter_hw_device hw -y -i \""+INPUT_FILE+"\" -filter:v fps="+str(int(args.prefps))+" -vf hwupload=extra_hw_frames=64,format=qsv -c:v h264_qsv -b:v 5M -c:a copy \""+TEMP_FOLDER+"/temp.mp4\"", shell=True)
    INPUT_FILE=TEMP_FOLDER+"temp.mp4"
    frameRate=args.prefps

if(frameRate == -1):
    frameRate = convert_to_float(subprocess.check_output(['ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate "'+INPUT_FILE+'"'], shell=True).decode('utf-8'))
    print("Determined framerate: ", frameRate, "FPS")

if(SAMPLE_RATE == -1):
    SAMPLE_RATE = int(subprocess.check_output(['ffprobe -v error -select_streams a -of default=noprint_wrappers=1:nokey=1 -show_entries stream=sample_rate "'+INPUT_FILE+'"'], shell=True).decode('utf-8'))
    print("Determined sample rate: ", SAMPLE_RATE, "Hz")

# Extract audio async...    
p = subprocess.Popen(["ffmpeg -y -i "+INPUT_FILE+" -ab 160k -ac 2 -ar "+str(SAMPLE_RATE)+" -vn -hide_banner -loglevel error "+TEMP_FOLDER+"/audio.wav"], shell=True)
# Execute multithreaded frame extraction...
command = "./conv-parallel.sh -i "+INPUT_FILE+" -o "+TEMP_FOLDER+" -of frame%06d.jpg -qscale:v "+str(FRAME_QUALITY)
subprocess.call(command, shell=True)

while p.poll() is None:
    print('Waiting for audio conversion...')
    time.sleep(1)

sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav")
audioSampleCount = audioData.shape[0]
maxAudioVolume = getMaxVolume(audioData)

samplesPerFrame = sampleRate/frameRate

audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))

hasLoudAudio = np.zeros((audioFrameCount))

for i in range(audioFrameCount):
    start = int(i*samplesPerFrame)
    end = min(int((i+1)*samplesPerFrame),audioSampleCount)
    audiochunks = audioData[start:end]
    maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume
    if maxchunksVolume >= SILENT_THRESHOLD:
        hasLoudAudio[i] = 1

chunks = [[0,0,0]]
shouldIncludeFrame = np.zeros((audioFrameCount))
for i in range(audioFrameCount):
    start = int(max(0,i-FRAME_SPREADAGE))
    end = int(min(audioFrameCount,i+1+FRAME_SPREADAGE))
    shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
    if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]): # Did we flip?
        chunks.append([chunks[-1][1],i,shouldIncludeFrame[i-1]])

chunks.append([chunks[-1][1],audioFrameCount,shouldIncludeFrame[i-1]])
chunks = chunks[1:]

audioThreadNum = 4

outputAudioData = np.zeros((0,audioData.shape[1]))
outputBuf = [np.zeros((0,audioData.shape[1]))] * audioThreadNum

lock = threading.Lock()

def renderChunk(pChunk, threadNum):
    global outputBuf
    outputPointer = 0
    lastExistingFrame = None
    for chunk in pChunk:
        audioChunk = audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)]
    
        sFile = TEMP_FOLDER+"/tempStart" + str(threadNum) +".wav"
        eFile = TEMP_FOLDER+"/tempEnd" + str(threadNum) + ".wav"
        wavfile.write(sFile,SAMPLE_RATE,audioChunk)
        with WavReader(sFile) as reader:
            with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
                tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                tsm.run(reader, writer)
        _, alteredAudioData = wavfile.read(eFile)
        leng = alteredAudioData.shape[0]
        endPointer = outputPointer+leng
        outputBuf[threadNum] = np.concatenate((outputBuf[threadNum],alteredAudioData/maxAudioVolume))

        # smooth out transitiion's audio by quickly fading in/out

        if leng < AUDIO_FADE_ENVELOPE_SIZE:
            outputBuf[threadNum][outputPointer:endPointer] = 0 # audio is less than 0.01 sec, let's just remove it.
        else:
            premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE
            mask = np.repeat(premask[:, np.newaxis],2,axis=1) # make the fade-envelope mask stereo
            outputBuf[threadNum][outputPointer:outputPointer+AUDIO_FADE_ENVELOPE_SIZE] *= mask
            outputBuf[threadNum][endPointer-AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask

        startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
        endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
        for outputFrame in range(startOutputFrame, endOutputFrame):
            inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame))
            didItWork = copyFrame(inputFrame,outputFrame)
            if didItWork:
                lastExistingFrame = inputFrame
            else:
                copyFrame(lastExistingFrame,outputFrame)
        outputPointer = endPointer

        lock.acquire()
        global chunksDone
        chunksDone += 1
        lock.release()

chunksSplit = list(split(chunks, audioThreadNum))

audioThreads = []

i=0
for cchunk in chunksSplit:
    audioThreads.append(threading.Thread(target=renderChunk, args=[cchunk, i]))
    i+=1

print("Rendering audio using", audioThreadNum, "threads...")

for thr in audioThreads:
    thr.start()

lastPercent = 0
lastEta = 0
eta = 0
while any([x.is_alive() for x in audioThreads]):
    percent = (chunksDone / len(chunks)) * 100
    eta = (( 0.1 / (percent-lastPercent)) * (100 - percent)) if percent > lastPercent else eta
    lastEta = (lastEta + eta) / 2
    print("\r" + str( round( percent ) ) + "% completed, eta " + str(round(lastEta)) + " seconds", end='', flush=True)
    lastPercent = percent
    time.sleep(0.1)

print("\nWriting audio...")
outputAudioData = np.concatenate(outputBuf)
wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData)

print("Joining frames and audio to mp4 file...")

command = "ffmpeg -hide_banner -loglevel error -y -hwaccel qsv -c:v mjpeg_qsv -framerate "+str(frameRate)+" -i "+TEMP_FOLDER+"/newFrame%06d.jpg -i "+TEMP_FOLDER+"/audioNew.wav -c:a aac -c:v h264_qsv -b:v 5M "+OUTPUT_FILE
subprocess.call(command, shell=True)

if not os.path.ismount(TEMP_FOLDER):
    deletePath(TEMP_FOLDER)
else:
    subprocess.call("rm " + TEMP_FOLDER + "/*", shell=True)

if args.displ:
    import subprocess, os, platform
    if platform.system() == 'Darwin':       # macOS
        subprocess.call(('open', OUTPUT_FILE))
    elif platform.system() == 'Windows':    # Windows
        os.startfile(OUTPUT_FILE)
    else:                                   # linux variants
        subprocess.call(('xdg-open', OUTPUT_FILE))