ffmpeg -i $1 -vf "crop=1080:1080:420:0" cropped.mp4
ffmpeg -i cropped.mp4 -start_number 0  -vf "fps=25" /root/CoDeF/all_sequences/heygen/heygen/%05d.png