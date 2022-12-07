# GBM_CAF_open
Public access scripts from Aghi Lab CAF projects


Work In Progress.

1. Taking the images and things to consider

Images are preferably taken on a confocal microscope. We use CellTrackerGreen as our dye because it is most compatible with our CellProfiler pipeline. The original VAMPIRE paper uses phalloidin conjugates to dye actin, however we found that is only useful for recognizing the cytoplasm of cells as primary objects. The capabilities of the pipeline rapidly degenerate once actin-stained cells are adjacent to each other, which is common in fibroblasts and many other cell types. Instead we use a DAPI/Fluoromount stain combined with Cell Tracker Green to seed the nuclei (which are easily recognized as they are almost never touching) and then use those as points to initiative secondary object recognition. In order to be fair, it's important to try and maintain the same power and laser exposure settings for all image sets, however, we also found in our data that minor variations in power settings did not largely affect the data points, as long as the the cells were still reasonably visible to the naked eye, the computational pipeline would also have no issues due to the robust nature of the algorithm in recognizing fluorescent signals. 

Other miscellanous things to consider: full z-stacks are typically not necessary as long as the image is captured on the z coordinate in which the nucleus is the brightest (which is typically the middle of the cell). Additionally, larger magnifications are typically preferable, as we found that at 20x, often-times larger fibroblasts would touch the edges of the field of view, and be removed by CellProfiler downstream, while such issues would not occur as frequently if our cells were imaged at 10x magnification instead. 

2. Setting up the Cell Profiler Pipeline

We encourage that each condition of pictures to be kept in their own folder, as seen in our example data. Secondly, we have also uploaded our preferred ImageJ macro DAPI_CYT_SPLIT.ijm for splitting the images into their respective DAPI and green fluorescent sub images. It is recommeneded that images be converted into .tiff format using the native ImageJ batch convert function. For exmaple, Zeiss microscopes running Zen software default save image files as .czi files, which can interfere with macros that are designed to batch process images. Converting all files into Tiff versions first can minimize the risk of such software issues. The uploaded pipeline GroupedSegmentation.cpipe is intended to be used in conjunction with this data set, and uses the automatically added DAPI and CYT tags at the end of the images to pair the images for primary and secondary object identification - a feature already coded into CellProfiler, the documentation of which can be found on their website, making the pipeline as simple as a drag and drop of the folder containing all the images, as the software will internally check for only CYT and DAPI matching pairs to analyze. 

Note. Our pipeline processes secondary objects using both watershed and propagation algorithms. Additionally, manual adjustment of the thresholds may be necessary to achieve optimum results for your images. The reason we choose to double calculate watershed and propagation algorithms is because different cell types have different affinities for CellTrackerGreen as a dye. GBM6 for example takes to it quite happily, while GBM43 will have dimmer signals. In an experimental design, the researcher is thus faced with the decision to introduce variance by extending the CellTrackerGreen incubation time for GBM43 beyond the 30 minutes that we used for all our cell times, or to change the power settings on the microscope. We experimented with both and found that changing incubation times could lead to overexposure of cell images. Additionally, even with increased incubation some cells would have secondary objects that covered the whole screen due to the algorithm being unable to differentiate clear cellular borders in the dimmer fluorescent signal. Doubling up on secondary object identification algorithms corrected for that, as propagation algorithms are more liberal and allowed clearer capture of the spindly structures common to fibroblast cell lines, while watershed methods allowed for clearer defined capture of the dim smaller shapes of certain glioblastoma cell lines. Image sets were then manually checked to see which method was suitable for each cell type, and then tags in the next section of the instructions were submitted to select the right set of segmented images. 

3. Installing and utilizing the VAMPIRE package

The VAMPIRE package is available as a python module at https://github.com/kukionfr/VAMPIRE_open, or as an .exe file at https://github.com/kukionfr/VAMPIRE_open/releases/download/executable/vampire.exe.

AS OF December 2022, you must use the .exe and run it as administrator. The standard python package cannot run the functions necessary to save and record boundary items and the code will run into numerous errors. 

Common early mistakes is many ransomware anti-virus softwares such as Avast will prevent the .exe files from creating the requisite changes in your hard-drive such as creating the .pickle and .csv files necessary to represent the package outcomes. It is recommended to either disable your ransomware protections or exempt your python IDE or the vampire.exe file in order for the program to run smoothly. 

The VAMPIRE software is relatively easy to use. You direct the software by uploading a .csv file with the condition, path, and tags of the images you are interested in. i.e. condition: GBM6 Cells, path: typically a C:/ string pointed at the folder with the SEGMENTED images of your cells, and lastly tag: i.e. CYT_propagation_segmented. It is this last part that allows us to apply our double algorithm technique above, by changing it to either CYT_propagation_segmented or CYT_watershed_segmented, we are thus able to selectively choose the best secondary object identification algorithm for downstream analysis with minimal hassle. 

CellProfiler will also generate .csv files that represent the locations of each of your segmented cells on each image. In our pipline, this file is named for example "MyExpt_Cytoplasm_propagation.csv", each file must be copied to the same folder as the images you seek to analyze in order for the VAMPIRE package to be able to recognize the locations of the segmented grayscale images and analyze them. i.e. "MyExpt_FilteredNuclei_propagation" in your BlueChannel folder, "MyExpt_Cytoplasm_watershed" in watershed propagated cell lines, etc. (FilteredNuclei means remaining nuclei once cells with cytoplasm touching the edges of the field of view are removed, along with their parent nuclei. For obvious reasons cells touching the edge of the field of view may only be partial, and are by default excluded from downstream analysis.). We have also uploaded a script called 'dist_loc_files.py' that automatically distributes the CellProfiler information files appropraitely to each subfolder, according to the master VAMPIRE.csv directory. 

After uploading this .csv file, you are then instructed to include settings for the number of coordinates in each cellular outline, as well as number of 'shape modes' - the number of 'species' of shape you want the program to divide your morphologies into. We chose 100 coordinates and 100 shape modes, doubling the default number of coordinates of 50 to increase detail in the tracings, and 100 shape modes so that specific variable was on a 100 point scale, which we felt struck an appropriate balance between granularity and specificity. 

You then use these settings to build the model. The resultant pickle file is then immediately used in the second half of the software window to 'apply' the model. Which then generates the 'VAMPIRE datasheet' as well as figures representing your shape modes. We edited the base VAMPIRE code to also export .csv files representing the values of the historgrams that represent their shape mode distributions, our version is also uploaded on this repo.

Important things to note and scripts of our own design to aid you. VAMPIRE analysis does not re-create pickle files that represent the X, Y coordinates of your objects if you've already run VAMPIRE analysis on that folder, even if the images themselves have changed. We have uploaded a 'clear_trash.py' script that automatically deletes all VAMPIRE analysis files in the relevant subfolders to ensure each run is freshly generating the requisite datapoints. 


4. Creating your own logistic regression and classifier.

Once these steps are complete, you are ready to build your logistic regression. We use the script 'train_test_caf.py' in order to achieve this. This script iterates through every cytoplasm cell in the VAMPIRE datasheets and then cross-references CellProfiler tables and the nuclei tables to match up cytoplasm and nuclei datapoints, it then joins them together and exports each cells data set as a .csv, as well as storing it in a dict that is exported in order to minimize processing requirements for downstream analysis so that you can simply re-import the pickle and save time. The reason the program has to do that is because each nuclei doesn't always spawn a cytoplasm in CellProfiler, and thus the software will almost have more nuclei than cytoplasms, however, without manually aligning by their original features and identifiers from Cell Profiler, brute force concatenation of the datafarames will lead to the nuclei of cell E being matched with the cytoplasm of cell D in downstream analysis. 

Once the data is processed, it is then put through a logistic regression, first it marks our fibroblast populations of choice as 1s in a binary outcome variable, and glioblastoma populations of choice as 0s. All other cellular populations are excluded and sequestered. 

The sum data is then randomly split 70/30 into training and testing data sets for the regression. 

5. Interpreting the data and fine tuning the general purposes. 

At this point we now have a functional statistical test for morphology, an extension of the original work of the VAMPIRE analysis code by Wu et.al. We then apply it by randomly splitting our non-trypsinized and serially trypsinized GBM patient samples into sub-populations, and applying the classifier to predict the identity of these cells by their morphological features. The proportions of each sub-population can then be compared using an unpaired t-test, with consistent proportions indicating that serially trypsinized glioblastoma patient samples are statsitically similar to breast cancer associated fibroblasts according to our model. 


***

All of our raw data has been also uploaded on this repository in order to help others apply this software as well as check our analysis. 
