#!/bin/bash
STARTTIME=$(date +%s)

# Parallel conversion of video ->  JPEG frames using ffmpeg
# (C) Fabian Schillig 2022

eecho() { printf "\033[0;31m$@\033[0m\n" 1>&2; }

function displaytime {
  local T=$1
  local D=$((T/60/60/24))
  local H=$((T/60/60%24))
  local M=$((T/60%60))
  local S=$((T%60))
  [[ $D > 0 ]] && printf '%dd ' $D
  [[ $H > 0 ]] && printf '%dh ' $H
  [[ $M > 0 ]] && printf '%dm ' $M
  printf '%ds' $S
}

waitall() { # PID...
  ## Wait for children to exit and indicate whether all exited with 0 status.
  local errors=0
  local lastTime=$(date +%s)
  local lastPer=0
  local lastRem=0
  local lastFrm=0
  [ $LOG_FORMAT -eq 1 ] && echo "completed, fps, elapsed, remaining"
  while :; do
    curFrm=$(ls TEMP/ | wc -l)
    progress=$(awk -va="$curFrm" -vb="$FRMS" 'BEGIN{print (a/b)*100}')
    rprog=$(awk -va="$progress" 'BEGIN { printf "%.0f", a}') # ( (time-lastTime) / (progr-lastprogr) * (100-progr))
    timePer=$(awk -va="$progress" -vb="$lastPer" -vc="$(date +%s)" -vd="$lastTime" 'BEGIN { printf "%.0f", ( (c-d)/(a-b) ) * (100-a) }' 2>/dev/null)
    fps=$(awk -va="$curFrm" -vb="$lastFrm" -vc="$(date +%s)" -vd="$lastTime" 'BEGIN { printf "%.0f", ( (a-b)/(c-d) ) }' 2>/dev/null)
    lastRem=$(awk -va="$lastRem" -vb="$timePer" 'BEGIN { printf "%.0f", (a+b)/2 }' 2>/dev/null)
    lastTime=$(date +%s)
    lastPer=$progress
    lastFrm=$curFrm
    
    timeRem=$(displaytime $lastRem)
    timeEla=$(displaytime $(( $(date +%s) - STARTTIME )) )
    #timeRem=$(date -d@$lastRem -u +%Hh\ %Mm\ %Ss 2>/dev/null)
    #timeEla=$(date -d@$(( $(date +%s) - STARTTIME )) -u +%-Hh\ %-Mm\ %-Ss 2>/dev/null )
    [ $LOG_FORMAT -eq 0 ] && printf "\r %d%% completed, %d FPS, %s elapsed, %s remaining" "$rprog" "$fps" "$timeEla" "$timeRem" || printf "%d%%, %d, %s, %s\n" "$rprog" "$fps" "$timeEla" "$timeRem"
    for pid in "$@"; do
      shift
      if kill -0 "$pid" 2>/dev/null; then
        set -- "$@" "$pid"
      elif wait "$pid"; then
        printf "\r"
      else
        eecho "\nError: ffmpeg ($pid) exited with non-zero exit status."
        ((++errors))
      fi
    done
    (("$#" > 0)) || break
    sleep 1
  done
  ((errors == 0))
}

#CORES=$(grep -c ^processor /proc/cpuinfo)
CORES=$(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}')
OUTPUT_FOLDER="TEMP/"
OUTPUT_FILE="frame%06d.jpg"
INPUT_FILE=""
CLEAR_OUTPUT=1
LOG_FORMAT=0

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
  -n | --no-clear)
    CLEAR_OUTPUT=0
    shift # past argument
    ;;
  -l | --log-format)
    LOG_FORMAT=1
    shift # past argument
    ;;
  -o | --output-folder)
    OUTPUT_FOLDER="$2"
    shift # past argument
    shift # past value
    ;;
  -of | --output-filename)
    OUTPUT_FILE="$2"
    shift # past argument
    shift # past value
    ;;
  -i | --input-file)
    INPUT_FILE="$2"
    shift # past argument
    shift # past value
    ;;
  -t | --threads)
    CORES="$2"
    shift # past argument
    shift # past value
    ;;
  -h | --help)
    echo -e "conv-parallel.sh -i (input file) [options] [FFMPEG OPTIONS]\n"
    echo ""
    echo "options:"
    echo "  -i,  --input-file         Input file"
    echo "  -o,  --output-folder      Output folder"
    echo "  -of, --output-filename    Output filename (with ffmpeg placeholder, e.g. image%06d.jpg)"
    echo "  -t,  --threads            Use n threads, default: all cpu cores"
    echo "  -n,  --no-clear           Don't clear output folder"
    echo "  -l,  --log-format         Print newlines + datestamps for csv log"
    echo ""
    echo "All other arguments will be passed to ffmpeg"
    exit 0
    ;;
  *)
    POSITIONAL_ARGS+=("$1") # save positional arg
    shift                   # past argument
    ;;
  esac
done

[ -z "$INPUT_FILE" ] && {
  eecho "Error: no input file specified!"
  exit 1
}
[ ! -f "$INPUT_FILE" ] && {
  eecho "Error: Input file not found!"
  exit 1
}

OUTPUT_FOLDER=$(echo "$OUTPUT_FOLDER" | sed 's:/*$::')
OUTPUT_FOLDER=${OUTPUT_FOLDER%/}

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

FPS=$(ffmpeg -i "$INPUT_FILE" 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p")
FRMS=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$INPUT_FILE")
FPC=$((FRMS / CORES))
pids=""

[ $LOG_FORMAT -eq 0 ] && echo "Using $CORES threads to convert $FRMS frames"

[ ! -d $OUTPUT_FOLDER/ ] && mkdir -p $OUTPUT_FOLDER
[ $CLEAR_OUTPUT -eq 1 -a "$(ls $OUTPUT_FOLDER/*.${OUTPUT_FILE##*.} | wc -l)" -gt 0 ] && { rm $OUTPUT_FOLDER/*.${OUTPUT_FILE##*.} || {
  eecho "Error: Failed to clear output directory!"
  exit 1
}; }

for c in $(eval echo {1..$((CORES - 1))}); do
  SFRME=$(( ( (c - 1) * FPC ) ))
  STIME=$(awk -va="$SFRME" -vb="$FPS" 'BEGIN{print a/b}')
  ffmpeg -threads 1 -ss $STIME -i "$INPUT_FILE" -threads 1 -start_number $SFRME -vframes $FPC ${POSITIONAL_ARGS[@]} $OUTPUT_FOLDER/$OUTPUT_FILE -hide_banner -loglevel error &
  pids="$pids $!"
done

SFRME=$(( ( (CORES - 1) * FPC ) ))
STIME=$(awk -va="$SFRME" -vb="$FPS" 'BEGIN{print a/b}')
ffmpeg -threads 1 -ss $STIME -i "$INPUT_FILE" -threads 1 -start_number $SFRME ${POSITIONAL_ARGS[@]} $OUTPUT_FOLDER/$OUTPUT_FILE -hide_banner -loglevel error &
pids="$pids $!"

waitall $pids || { echo ""; echo ""; echo "Error: Some conversions failed :("; exit 1; }

echo ""
echo "Finished in $(($(date +%s) - STARTTIME)) seconds"