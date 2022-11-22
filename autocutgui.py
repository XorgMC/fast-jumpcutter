import os
import argparse

parser = argparse.ArgumentParser(description="Speed up silent parts in videos (e.g. lectures).")
parser.add_argument("--input-file", help="the path to a video file (.mp4)", required=True)
parser.add_argument("--output-file",
                    help="the path to the output video (default: will append \"_faster\" to the existing filename)")
parser.add_argument("--silent-speed", default=10, type=int,
                    help="how much silent parts should be sped up (default: 10)")
parser.add_argument("--silent-threshold", default=600, type=int,
                    help="the threshold of what should be considered silent (default: 600)")
parser.add_argument("--frame-margin", default=1, type=int, dest="framemargin", help="Include n frames around segments")
parser.add_argument("--show-ffmpeg-output", action="store_true",
                    help="if given, shows ffmpeg output (which is hidden by default)")
args = parser.parse_args()

# Loading some of these dependencies takes a while, which should not be done if -h or --help is passed
import subprocess
import click
import time
import datetime
import tempfile
import threading
import numpy as np
# import noisereduce as nr
from scipy.io import wavfile
import cv2
from audiotsm import phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter
from vidgear.gears import WriteGear
import PySimpleGUI as sg


# Utility functions
def get_max_volume(s):
    maxv = np.max(s)
    minv = np.min(s)
    return max(maxv, -minv)


def progress_reader(procs, q):
    while True:
        if procs.poll() is not None:
            break  # Break if FFmpeg sun-process is closed

        progress_text = procs.stdout.readline()  # Read line from the pipe

        # Break the loop if progress_text is None (when pipe is closed).
        if progress_text is None:
            break

        progress_text = progress_text.decode("utf-8")  # Convert bytes array to strings

        # Look for "frame=xx"
        if progress_text.startswith("frame="):
            frame = int(progress_text.partition('=')[-1])  # Get the frame number
            q[0] = frame  # Store the last sample


progress_val = 0
progress2_val = 0
actions_val = 0
debug_output = ""
done_flag = False
exit_flag = False
files = []
update_progress = threading.Event()
exit_event = threading.Event()


def cut_video(input_file):
    global debug_output
    global actions_val
    global progress_val
    global progress2_val
    global done_flag
    global exit_flag
    global files
    # TODO check if dependencies are installed and especially if ffmpeg is available
    # TODO if you're bored, check if this can be made any faster

    # Set stdout, stderr for ffmpeg calls and end for any prints before calling ffmpeg
    stdout = None if args.show_ffmpeg_output else subprocess.DEVNULL
    stderr = None if args.show_ffmpeg_output else subprocess.STDOUT
    end = "\n" if args.show_ffmpeg_output else ""

    # Create a VideoCapture and get some video properties
    # TODO check if the input file exists
    capture = cv2.VideoCapture(input_file)
    # capture.open(args.input_file, cv2.CAP_FFMPEG, ())
    # capture = cv2.VideoCapture(args.input_file, cv2.CAP_FFMPEG, (cv2.CAP_PROP_HW_ACCELERATION,
    #                                                             cv2.VIDEO_ACCELERATION_ANY))
    # video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    # Get temporary filenames
    original_audio_file = next(tempfile._get_candidate_names()) + ".wav"
    video_file = next(tempfile._get_candidate_names()) + ".mp4"
    audio_file = next(tempfile._get_candidate_names()) + ".wav"

    # Extract audio
    print("Extracting audio... ", end=end, flush=True)
    debug_output += "Extrahiere Audio... "

    # subprocess.call("ffmpeg -i \"{}\" -ab 160k -ac 2 -ar 44100 -vn {}".format(input_file, original_audio_file),
    #                shell=True,
    #                stdout=stdout,
    #                stderr=stderr)
    ap = subprocess.Popen(["ffmpeg -hide_banner -loglevel error -i \"{}\" -ab 160k -ac 2 -ar 44100 "
                           "-vn {}".format(input_file, original_audio_file)],
                          shell=True,
                          stdout=stdout,
                          stderr=stderr)

    while ap.poll() is None:
        if exit_flag:
            ap.kill()
            os.remove(original_audio_file)
            debug_output += "\nAbbruch!"
            return

    sample_rate, audio_data = wavfile.read(original_audio_file)
    audio_channels = int(audio_data.shape[1])
    print("done!")
    debug_output += "fertig!"
    actions_val = 5

    # Create output writer
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video_output = cv2.VideoWriter(video_file, fourcc, fps, (video_width, video_height))

    output_params = {"-vcodec": "h264_nvenc"}
    video_output = WriteGear(output_filename=video_file, logging=False, **output_params)

    audio_pointer = 0
    modified_audio = np.zeros_like(audio_data, dtype=np.int16)

    # Points to the start and end of a silent segment
    silent_start = 0
    silent_end = -1
    silent_margin = int(round(args.framemargin / fps * sample_rate))

    AUDIO_FADE_ENVELOPE_SIZE = 400
    premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE) // AUDIO_FADE_ENVELOPE_SIZE
    mask = np.repeat(premask[:, np.newaxis], 2, axis=1)

    # Holds video frames accumulated during silent parts
    buffer = []
    t_start = time.time()
    tpi = 0
    frame_margin_ctr = 0
    nth_frame = 0
    debug_output += "\n"
    with click.progressbar(length=video_length, label="Speeding up silent parts of the video...") as bar:
        for _ in bar:
            if exit_flag:
                capture.release()
                video_output.close()
                os.remove(original_audio_file)
                os.remove(video_file)
                debug_output += "\nAbbruch!"
                return
            debug_output = "\n".join(debug_output.split("\n")[:-1])
            debug_output += "\nBeschleunige stille Teile des Videos..."
            debug_output += bar.format_pct() + " (~"
            debug_output += bar.format_eta() + " verbleibend) "
            progress_val = bar.pct * 100

            ret, frame = capture.read()
            if nth_frame == 1:
                continue
            else:
                nth_frame += 1
            # Break if no frames are left
            if ret == False:
                break

            current_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)

            # Get the audio snippet for this frame
            audio_sample_start = round(current_frame / fps * sample_rate)
            audio_sample_end = int(audio_sample_start + sample_rate / fps)  # exclusive
            audio_sample = audio_data[audio_sample_start:audio_sample_end]

            # This shouldn't happen, but... yeah
            if len(audio_sample) == 0:
                break

            # is this a silent frame?
            if get_max_volume(audio_sample) < args.silent_threshold:
                if frame_margin_ctr >= args.framemargin:  # skipped enough frames?
                    buffer.append(frame)
                    silent_end = audio_sample_end
                    continue
                else:  # keep framemargin frames sound
                    frame_margin_ctr += 1

            if silent_end != -1:
                frame_margin_ctr = 0
                # Phasevocode silent audio
                silent_sample = audio_data[silent_start:silent_end]

                # audiotsm uses flipped dimensions, so transpose the numpy array
                reader = ArrayReader(silent_sample.transpose())
                writer = ArrayWriter(audio_channels)
                tsm = phasevocoder(reader.channels, speed=args.silent_speed)
                tsm.run(reader, writer)

                # Transpose back to regular dimensions
                phasevocoded_sample = writer.data.transpose()

                # Add silent audio sample
                audio_pointer_end = audio_pointer + len(phasevocoded_sample)
                modified_audio[audio_pointer:audio_pointer_end] = phasevocoded_sample

                if len(audio_sample) < AUDIO_FADE_ENVELOPE_SIZE:
                    modified_audio[audio_pointer:audio_pointer_end] = 0
                else:
                    modified_audio[audio_pointer:audio_pointer + AUDIO_FADE_ENVELOPE_SIZE] *= mask
                    modified_audio[audio_pointer_end - AUDIO_FADE_ENVELOPE_SIZE:audio_pointer_end] *= 1 - mask

                # Calculate the position of audio_pointer based on how much
                # sound is needed for the number of video frames added
                audio_pointer += round(len(buffer[::args.silent_speed]) / fps * sample_rate)

                # Write video frames and clear buffer
                for frame in buffer[::args.silent_speed]:
                    video_output.write(frame)

                buffer = []

            # Write video frame
            video_output.write(frame)
            silent_start = audio_sample_end + 1
            silent_end = -1  # indefinite

            # Add audio sample
            audio_pointer_end = audio_pointer + len(audio_sample)
            modified_audio[audio_pointer:audio_pointer_end] = audio_sample
            audio_pointer = audio_pointer_end

            tn = time.time()
            tpi = (tpi + (tn - t_start)) / float(2)
            t_start = tn

        bar.finish()

    if exit_flag:
        capture.release()
        video_output.close()
        os.remove(original_audio_file)
        os.remove(video_file)
        debug_output += "\nAbbruch!"
        return

    debug_output += "\nSchreibe modifiziertes Audio... "
    # Slice off empty end (as frames have been skipped)
    modified_audio = modified_audio[:audio_pointer]
    wavfile.write(audio_file, sample_rate, modified_audio)
    debug_output += "fertig!"
    actions_val = 10

    # Release resources
    # For some reason this is necessary, otherwise the resulting mp4 is corrupted
    capture.release()
    video_output.close()

    if args.output_file is None:
        name, ext = os.path.splitext(input_file)
        out_file = "{}_faster{}".format(name, ext)
    else:
        out_file = args.output_file

    # Merge video and audio with ffmpeg
    print("Merging video and audio... ", end=end, flush=True)
    debug_output += "\nVereine Video und Audio zu einer Datei... "
    if (os.path.exists(out_file)):
        # TODO (y/n) prompt?
        os.remove(out_file)

    mp = subprocess.Popen(
        "ffmpeg -hide_banner -loglevel error -i {} -i {} -progress pipe:1 -c:v copy -c:a aac \"{}\"".format(video_file,
                                                                                                            audio_file,
                                                                                                            out_file),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=stderr)

    q = [0]  # We don't really need to use a Queue - use a list of of size 1
    progress_reader_thread = threading.Thread(target=progress_reader, args=(mp, q))  # Initialize progress reader thread
    progress_reader_thread.start()  # Start the thread

    while mp.poll() is None:
        n_frame = q[0]  # Read last element from progress_reader - current encoded frame
        progress2_val = int(round((n_frame / video_length) * 100))  # Convert to percentage.
        debug_output = "\n".join(debug_output.split("\n")[:-1])
        debug_output += "\nVereine Video und Audio zu einer Datei... "
        debug_output += str(progress2_val) + "%"
        time.sleep(0.5)
        if exit_flag:
            mp.kill()
            progress_reader_thread.join(5)
            os.remove(original_audio_file)
            os.remove(audio_file)
            os.remove(video_file)
            debug_output += "\nAbbruch!"
            return
    mp.stdout.close()  # Close stdin pipe.
    progress_reader_thread.join()  # Join thread
    mp.wait()  # Wait for FFmpeg sub-process to finish

    os.remove(original_audio_file)
    os.remove(audio_file)
    os.remove(video_file)
    print("done!")
    actions_val = 40
    if mp.returncode == 0:
        debug_output += "fertig!\n\n"
    else:
        debug_output += "Fehler :(\n\n"
    actions_val = 50

    print("The output file is available at {}".format(out_file))

    # Compare old and new duration
    old_duration = str(datetime.timedelta(seconds=int(video_length / fps)))
    new_duration = str(datetime.timedelta(seconds=int(len(modified_audio) / sample_rate)))
    debug_output += "Old duration: {}, new duration: {}\n\n".format(old_duration, new_duration)
    done_flag = True
    files.remove(input_file)

def updateWindow(win):
    global progress_val
    global progress_total_val
    while not done_flag:
        pv = progress_val + progress2_val + actions_val
        win["pb1"].update(pv)
        win["debugOutput"].update(debug_output)
        time.sleep(0.4)
    pv = progress_val + progress2_val + actions_val
    win["pb1"].update(pv)
    win["Abbruch"].update("Schließen")


sg.theme('DarkAmber')

layout = [[sg.Text('AutoCutterV3 GUI SUPER HD PREMIUM!!!')],
          [sg.Text('Wähle Video:'), sg.Input(key="InputFile", change_submits=True, readonly=True, text_color="#000"),
           sg.Button("Durchsuchen", key="Selekt")],
          [sg.Button('Start'), sg.Button('Schließen', key="Abbruch")],
          [sg.HorizontalSeparator()],
          [sg.Text('Gesamtfortschritt: (0 von 0 Videos verarbeitet)', key="totalProgress")],
          [sg.ProgressBar(100, key="pb2", size_px=(1, 15), expand_x=True)],
          [sg.Text('Einzelfortschritt:')],
          [sg.Multiline("", key="debugOutput", size=(1, 15), expand_x=True, disabled=True)],
          [sg.ProgressBar(250, key="pb1", size_px=(1, 15), expand_x=True)]
          ]

# sg.FileBrowse("Durchsuchen", key="InputFile", file_types=(("Videos", "*.mp4 *.mkv *.avi *.mpg"),
# #                                                          ("Alle Dateien", "*.*"),))],

# Create the Window
window = sg.Window('Window Title', layout)
convThread = None
updateThread = None
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == "Start":
        vc = 0
        for f in files:
            vc += 1
            done_flag = False
            window["totalProgress"].update("Gesamtfortschritt (" + str(vc) + " von " + str(len(files)) + "Videos "
                                                                                                         "verarbeitet)")
            window["pb2"].update(round((vc / len(files)) * 100))
            debug_output += "--- Konvertiere \"" + f + "\" ---"
            convThread = threading.Thread(target=cut_video, args=[f])
            convThread.start()
            updateThread = threading.Thread(target=updateWindow, args=[window])
            updateThread.start()
            while convThread.is_alive() and updateThread.is_alive():
                time.sleep(0.5)

    if event == "Selekt":
        files = sg.popup_get_file('Unique File select', no_window=True, multiple_files=True,
                                  file_types=(("Videos", "*.mp4 *.mkv "
                                                         "*.avi *.mpg"),
                                              ("Alle Dateien", "*.*"),))
        fs = ""
        if files is None:
            files = []
        for f in files:
            fs += f + ";"
        window["InputFile"].update(fs)
    if event == "Abbruch":
        window["Abbruch"].update("Abbrechen")
        if convThread is None:
            break
        if not convThread.is_alive():
            break
        exit_flag = True
        convThread.join(10)
        done_flag = True
        if updateThread is not None:
            updateThread.join(10)
        updateWindow(window)
        exit_flag = False
        done_flag = False
    if event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
        exit_flag = True
        if convThread is not None:
            convThread.join(10)
        done_flag = True
        if updateThread is not None:
            updateThread.join(10)
        updateWindow(window)
        exit_flag = False
        done_flag = False
        break

window.close()
