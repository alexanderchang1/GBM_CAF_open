# clear and deletes old pickle and analysis files so that the vampire package doesn't get confused.

import pandas as pd
import shutil
import os
import sys
cyto = r'C:\Users\super\Documents\Aghi Lab PC\CAF Project\Grouped Segmentation\CAF_open' \
       r'\Segmented image sets propagation cyt.csv'
nuc = r'C:\Users\super\Documents\Aghi Lab PC\CAF Project\Grouped Segmentation\CAF_open' \
      r'\Segmented image sets propagation nuc.csv'

cyto_df = pd.read_csv(cyto)
nuc_df = pd.read_csv(nuc)
dataframes = [cyto_df, nuc_df]




for grids in dataframes:

    for path in grids['set location']:
        files = os.listdir(path)

        for item in files:
            if item.endswith(".pickle"):
                os.remove(os.path.join(path, item))
            if item.endswith("segmented.tiff"):
                os.remove(os.path.join(path, item))
            if item.endswith(".csv"):
                os.remove(os.path.join(path, item))






