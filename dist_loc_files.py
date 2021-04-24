import pandas as pd
import shutil

import tkinter as tk
from tkinter import filedialog
import easygui
import sys




cyto = easygui.fileopenbox("Please select the VAMPIRE directory for cytoplasm images", "cyto_segmented")

nuc = easygui.fileopenbox("Please select the VAMPIRE directory for nucleus images", "nuc_segmented")

prop_cyt = easygui.fileopenbox("Please select the CellProfiler datasheet for propagated cytoplasm", "expt_cyt_prop")

prop_nuc = easygui.fileopenbox("Please select the CellProfiler datasheet for propagated nucleus", "expt_cyt_prop")

watershed_cyt = easygui.fileopenbox("Please select the CellProfiler datasheet for watershed cytoplasm",
                                    "expt_cyt_watershed")
watershed_nuc = easygui.fileopenbox("Please select the CellProfiler datasheet for watershed nucleus",
                                    "expt_cyt_watershed")


propagation_tags = ['CYT_propagation_segmented', 'DAPI_propagation_segmented']

watershed_tags = ['CYT_watershed_segmented', 'DAPI_watershed_segmented']

cyto_df = pd.read_csv(cyto)
nuc_df = pd.read_csv(nuc)

for path, tags in zip(cyto_df['set location'],cyto_df['tag']):
    if tags in propagation_tags:
        shutil.copy(prop_cyt,path)
    elif tags in watershed_tags:
        shutil.copy(watershed_cyt,path)
    else:
        print("TAG NOT FOUND")


for path, tags in zip(nuc_df['set location'],nuc_df['tag']):
    if tags in propagation_tags:
        shutil.copy(prop_nuc,path)
    elif tags in watershed_tags:
        shutil.copy(watershed_nuc,path)
    else:
        print("TAG NOT FOUND")

