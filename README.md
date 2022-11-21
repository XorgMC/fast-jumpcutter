# fast-jumpcutter
(1 minute video)
python3 jumpcutter-orig.py --input_file VL1a.mp4 --frame_rate 5 --sample_rate  15,43s user 1,53s system 268% cpu 6,316 total
python3 jumpcutter.py --input_file VL1a.mp4 --silent_speed 99999  0.1  0  11,23s user 2,96s system 265% cpu 5,350 total



# jumpcutter
Automatically edits videos. Explanation here: https://www.youtube.com/watch?v=DQ8orIurGxw

Go here for a more polished version of this software that my friends and I have been working on fr the last year or so: https://jumpcutter.com/

Since my GitHub is more like a dumping ground or personal journal, I'm not going to be actively updating this GitHub repo. But if you do want a version of jumpcutter that is actively being worked on, please do check on the version at https://jumpcutter.com/! There's way more developers fixing bugs and adding new features to that tool, and there's a developer's Discord server to discuss anything JC-related, so go check it out!

## Some heads-up:

It uses Python 3.

It works on Ubuntu 16.04 and Windows 10. (It might work on other OSs too, we just haven't tested it yet.)

This program relies heavily on ffmpeg. It will start subprocesses that call ffmpeg, so be aware of that!

As the program runs, it saves every frame of the video as an image file in a
temporary folder. If your video is long, this could take a LOT of space.
I have processed 17-minute videos completely fine, but be wary if you're gonna go longer.

I want to use pyinstaller to turn this into an executable, so non-techy people
can use it EVEN IF they don't have Python and all those libraries. Jabrils 
recommended this to me. However, my pyinstaller build did not work. :( HELP
