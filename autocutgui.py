import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Speed up silent parts in videos (e.g. lectures).")
# parser.add_argument("--input-file", help="the path to a video file (.mp4)", required=True)
parser.add_argument('file', type=argparse.FileType('r'), nargs='*')
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
import soundfile
import cv2
from audiotsm import phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter
from vidgear.gears import WriteGear
import PySimpleGUI as sg

progress_val = 0
progress2_val = 0
actions_val = 0
debug_output = ""
done_flag = False
exit_flag = False
convThread = None
updateThread = None
files = []
update_progress = threading.Event()
exit_event = threading.Event()
silent_speed = args.silent_speed
silent_thres = args.silent_threshold
ffmpeg_path = ""


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


def cut_video(input_file, vc, tc):
    global debug_output, actions_val, progress_val, progress2_val, done_flag, exit_flag, files, silent_thres, \
        silent_speed
    # TODO check if dependencies are installed and especially if ffmpeg is available
    # TODO if you're bored, check if this can be made any faster

    debug_output += "--- Konvertiere \"" + input_file + "\" ---\n"

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
    ap = subprocess.Popen([ffmpeg_path + " -hide_banner -loglevel error -i \"{}\" -ab 160k -ac 2 -ar 44100 "
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

    # sample_rate, audio_data = wavfile.read(original_audio_file)
    audio_data, sample_rate = soundfile.read(original_audio_file, dtype='int16')
    audio_channels = int(audio_data.shape[1])
    print("done!")
    debug_output += "fertig!"
    actions_val = 5

    # Create output writer
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video_output = cv2.VideoWriter(video_file, fourcc, fps, (video_width, video_height))

    output_params = {"-input_framerate": fps, "-r": fps, "-vcodec": "h264_nvenc"}
    video_output = WriteGear(output_filename=video_file, logging=False, **output_params, custom_ffmpeg=ffmpeg_path)

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
            if get_max_volume(audio_sample) < silent_thres:
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
                tsm = phasevocoder(reader.channels, speed=silent_speed)
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
                audio_pointer += round(len(buffer[::silent_speed]) / fps * sample_rate)

                # Write video frames and clear buffer
                for frame in buffer[::silent_speed]:
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
    # wavfile.write(audio_file, sample_rate, modified_audio)
    soundfile.write(audio_file, modified_audio, sample_rate)
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
        name, ext = os.path.splitext(input_file)
        out_file = args.output_file.replace("%name%", name).replace("%ext%", ext)

    # Merge video and audio with ffmpeg
    print("Merging video and audio... ", end=end, flush=True)
    debug_output += "\nVereine Video und Audio zu einer Datei... "
    if os.path.exists(out_file):
        # TODO (y/n) prompt?
        os.remove(out_file)

    mp = subprocess.Popen(ffmpeg_path +
                          " -hide_banner -loglevel error -i {} -i {} -progress pipe:1 -c:v copy -c:a aac \"{}\"".format(
                              video_file,
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

    updateGlobalProgress(vc, tc)
    files.remove(input_file)
    if len(files) == 0:
        progress_val = 100
        progress2_val = 100
        actions_val = 50
        done_flag = True
    else:
        progress_val = 0
        progress2_val = 0
        actions_val = 0
        cut_video(files[0], vc + 1, tc)


def updateGlobalProgress(vc, tc):
    window["totalProgress"].update("Gesamtfortschritt (" + str(vc) + " von " + str(tc) + " Videos verarbeitet)")
    window["pb2"].update(round((vc / tc) * 100))


def updateWindow(win):
    global progress_val
    while not done_flag:
        pv = progress_val + progress2_val + actions_val
        win["pb1"].update(pv)
        win["debugOutput"].update(debug_output)
        time.sleep(0.4)
    pv = progress_val + progress2_val + actions_val
    win["pb1"].update(pv)
    win["Abbruch"].update("Schließen")
    window["Start"].update(disabled=False)
    window["Selekt"].update(disabled=False)


def cancel_conversion():
    global convThread
    global exit_flag
    global done_flag
    global updateThread
    if convThread is None or not convThread.is_alive():
        return
    exit_flag = True
    convThread.join(10)
    done_flag = True
    if updateThread is not None:
        updateThread.join(10)
    updateWindow(window)
    exit_flag = False
    done_flag = False
    convThread = None
    updateThread = None


sg.LOOK_AND_FEEL_TABLE['MyCreatedTheme'] = {'BACKGROUND': '#15EAFF',
                                            'TEXT': '#333',
                                            'INPUT': '#FE01FF',
                                            'TEXT_INPUT': '#111',
                                            'SCROLL': '#FE01FF',
                                            'BUTTON': ('#1B001B', '#AB53FF'),
                                            'PROGRESS': ('#AB53FF', '#4B004B'),
                                            'BORDER': 1, 'SLIDER_DEPTH': 1,
                                            'PROGRESS_DEPTH': 1, }

hdr = "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"
sg.theme('MyCreatedTheme')
# sg.Text('AutoCutterV3 GUI SUPER HD PREMIUM!!!')
layout = [[sg.Image(data=hdr)],
          [sg.Frame("Eingabe:", [[sg.Text('Wähle Video(s):'),
                                  sg.Input(key="InputFile", change_submits=True, readonly=True, text_color="#000"),
                                  sg.Button("Durchsuchen", key="Selekt")],
                                 [sg.Button('Start'), sg.Button('Schließen', key="Abbruch")]]
                    )],
          [sg.Frame("Knöpfe zum Drehen:", [
              [sg.Text("Stille-Schwelle:🔈", pad=(0, 3)),
               sg.Slider(range=(0, 10000), default_value=silent_thres, resolution=100, orientation='horizontal',
                         expand_x=True, disable_number_display=True, pad=(0, 0), enable_events=True, key="thr"),
               sg.Text("🔉 600  ", key="thr-lbl", pad=(0, 3), font=("monospace", 11))],
              [sg.Text("Stille-Geschw.:🐢", pad=(0, 3)),
               sg.Slider(range=(0, 500), default_value=silent_speed, resolution=1, orientation='horizontal',
                         expand_x=True, disable_number_display=True, pad=(0, 0), enable_events=True, key="spd"),
               sg.Text("🐇 10   ", key="spd-lbl", pad=(0, 3), font=("monospace", 11))]
          ], expand_x=True)],
          [sg.HorizontalSeparator()],
          [sg.Frame("Protokoll:", [
              [sg.Text('Gesamtfortschritt: (0 von 0 Videos verarbeitet)', key="totalProgress")],
              [sg.ProgressBar(100, key="pb2", size_px=(1, 15), expand_x=True)],
              [sg.Text('Einzelfortschritt:')],
              [sg.Multiline("", key="debugOutput", size=(1, 15), expand_x=True, disabled=True)],
              [sg.ProgressBar(250, key="pb1", size_px=(1, 15), expand_x=True)]
          ], expand_x=True)]

          ]

# sg.FileBrowse("Durchsuchen", key="InputFile", file_types=(("Videos", "*.mp4 *.mkv *.avi *.mpg"),
# #                                                          ("Alle Dateien", "*.*"),))],

# Create the Window
window = sg.Window('AutoCutter V3', layout, font=("Comic Sans MS", 12), finalize=True)

if os.path.exists("ffmpeg"):
    ffmpeg_path = "./ffmpeg"
else:
    import shutil

    bpath = shutil.which("ffmpeg")
    if bpath is not None:
        ffmpeg_path = "bpath"
    else:
        sg.popup_error("Fehler: ffmpeg wurde nicht gefunden. Bitte lade ffmpeg herunter und platziere es im gleichen " +
                       "Ordner wie diese Datei. FFMPEG bekommst du hier: https://ffmpeg.org/download.html",
                       "Kein FFMPEG :(", non_blocking=False)
        sys.exit(1)


def on_close():
    if convThread is not None and convThread.is_alive():
        if sg.popup("Vorgang abbrechen und beenden?", title="Bestätigung", custom_text=("Ja", "Nein")) == "Ja":
            sg.Popup("Moment, Umwandlung wird abgebrochen...", non_blocking=True, custom_text="")
            cancel_conversion()
            window.Close()
            sys.exit(0)
    else:
        window.Close()
        sys.exit(0)
    return


window.TKroot.protocol("WM_DELETE_WINDOW", on_close)

if len(args.file) > 0:
    print(args.file)
    fs = ""
    for f in args.file:
        files.append(f.name)
        fs += f.name + ";"
    window["InputFile"].update(fs)
    window["Start"].click()

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == "Start":
        if convThread is not None:
            if convThread.is_alive():
                continue
        if len(files) > 0:
            window["Abbruch"].update("Abbrechen")
            window["Selekt"].update(disabled=True)
            window["Start"].update(disabled=True)
            done_flag = False
            print(files)
            convThread = threading.Thread(target=cut_video, args=[files[0], 1, len(files)])
            convThread.start()
            updateThread = threading.Thread(target=updateWindow, args=[window])
            updateThread.start()
    if event == "thr":
        window["thr-lbl"].update("🔉 {}".format(round(values['thr'])).ljust(7))
        silent_thres = round(values['thr'])
    if event == "spd":
        window["spd-lbl"].update("🐇 {}".format(round(values['spd'])).ljust(7))
        silent_speed = round(values['spd'])
    if event == "Selekt":
        filesel = sg.popup_get_file('Unique File select', no_window=True, multiple_files=True,
                                    file_types=(("Videos", "*.mp4 *.mkv "
                                                           "*.avi *.mpg"),
                                                ("Alle Dateien", "*.*"),))
        fs = ""
        if filesel is None:
            files = []
        files.clear()
        for f in filesel:
            files.append(f)
            fs += f + ";"
        window["InputFile"].update(fs)
    if event == "Abbruch":
        if convThread is None or not convThread.is_alive():
            window.close()
            sys.exit(0)
        cancel_conversion()
        window["Abbruch"].update("Schließen")
        debug_output += "\nAbbruch!"
        window["debugOutput"].update(debug_output)
        window["pb1"].update(0)
        window["pb2"].update(0)
        window["Start"].update(disabled=False)
        window["Selekt"].update(disabled=False)
    if event == sg.WIN_CLOSED:
        break

window.close()
