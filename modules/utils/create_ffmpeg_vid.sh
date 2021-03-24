#!/bin/bash

# example script, will need modifications for specific use case
# -r means frame rate
# -start_number means startin num for frames
# -i means identifier formats, for frame_000.jpg use frame_%3d.jpg
# -vf .... in this case ensures frame odd dimensions are converted to even
ffmpeg -r 24 \
       -start_number 000 \
       -i frame_%3d.jpg \
       -vcodec libx264 \
       -vf "pad=ceil(iw/2)\_2:ceil(ih/2)\_2"  \
       -y -an video.mp4
