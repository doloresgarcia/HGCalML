import os
import matplotlib.pyplot as plt
import functools
import pandas as pd
import matplotlib.animation as animation


def cmp_func(a, b):
    a0, a1 = a.split("_")
    b0, b1 = b.split("_")
    a0, a1 = int(a0), int(a1)
    b0, b1 = int(b0), int(b1)
    if a0 > b0:
        return True
    elif a0 < b0:
        return False
    else:
        return a1 > b1


# TODO: change in_folder to the output of plot_clustering_space_1008.py
IN_FOLDER = "/eos/user/g/gkrzmanc/ds2507results_bk/philip-dataset-140823/cluster_coords_by_epoch"


# TODO: Change this to your event IDs
events = [1, 2, 3]   # what events to animate
events_render = [1, 2, 3] # for what events to render the animation

from pathlib import Path

files = os.listdir(IN_FOLDER)
files = [x for x in files if not (x.startswith("frames") or x.startswith("anim"))]
files_ids = [(int(f.split("_")[0]), int(f.split("_")[1])) for f in files]
files.sort(key=functools.cmp_to_key(cmp_func))

for event in events:
    print("  *********  Event", event, "  *********  ")
    frames = []
    epoch_texts = []
    frame_folder = os.path.join(IN_FOLDER, "frames", "frames_event_" + str(event))
    Path(frame_folder).mkdir(parents=True, exist_ok=True)
    for epoch in files:
        event_folder = os.path.join(IN_FOLDER, epoch)
        if str(event) not in os.listdir(event_folder):
            continue
        event_folder = os.path.join(event_folder, str(event))
        frames.append(event_folder)
        epoch_texts.append(epoch)
    for i in range(len(frames)):
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        fold = frames[i]
        truth_img = os.path.join(fold, "0_truth.png")
        coords_betasize = os.path.join(fold, "0_ccoords_betasize.png")
        # just "plot" an image to the right
        ax[1].imshow(plt.imread(truth_img))
        ax[1].axis("off")
        ax[1].set_title("Truth")
        ax[0].set_title("Event " + str(event) + " epoch " + epoch_texts[i])
        ax[0].imshow(plt.imread(coords_betasize))
        fig.tight_layout()
        fig.savefig(os.path.join(frame_folder, str(i) + ".png"))
        plt.close(fig)

import glob
from PIL import Image

for event in events_render:
    print("  *********  Rendering event ", event, "  *********  ")
    frame_folder = os.path.join(IN_FOLDER, "frames", "frames_event_" + str(event))
    output_folder = os.path.join(IN_FOLDER, "animations")
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_folder, "filelist.txt"), "w") as f:
        for file in sorted(os.listdir(frame_folder), key=lambda x: int(x.split(".")[0])):
            f.write("file " + os.path.join(frame_folder, file) + "\n")
    out_file = os.path.join(output_folder, "event_" + str(event) + ".mp4")
    cmd = "ffmpeg   -f concat  -r 14 -safe 0 -i '{}/filelist.txt' -c:v libx264 -pix_fmt yuv420p {}".format(output_folder, out_file)
    os.system(cmd)
    print("  *********  Done rendering event ", event, "  *********  ")

