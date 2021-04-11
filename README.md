# GBM_CAF_open
Public access scripts from Aghi Lab CAF projects


Work In Progress.

1. Taking the images and things to consider

Images are preferably taken on a confocal microscope. We use CellTrackerGreen as our dye because it is most compatible with our CellProfiler pipeline. The original VAMPIRE paper uses phalloidin conjugates to dye actin, however we found that is only useful for recognizing the cytoplasm of cells as primary objects. The capabilities of the pipeline rapidly degenerate once actin-stained cells are adjacent to each other, which is common in fibroblasts and many other cell types. Instead we use a DAPI/Fluoromount stain combined with Cell Tracker Green to seed the nuclei (which are easily recognized as they are almost never touching) and then use those as points to initiative secondary object recognition. In order to be fair, it's important to try and maintain the same power and laser exposure settings for all image sets, however, we also found in our data that minor variations in power settings did not largely affect the data points, as long as the the cells were still reasonably visible to the naked eye, the computational pipeline would also have no issues due to the robust nature of the algorithm in recognizing fluorescent signals. 

Other miscellanous things to consider: full z-stacks are typically not necessary as long as the image is captured on the z coordinate in which the nucleus is the brightest (which is typically the middle of the cell). Additionally, larger magnifications are typically preferable, as we found that at 20x, often-times larger fibroblasts would touch the edges of the field of view, and be removed by CellProfiler downstream, while such issues would not occur as frequently if our cells were imaged at 10x magnification instead. 

2. Setting up the Cell Profiler Pipeline

We encourage each condition of pictures to be kept in their own folder, as seen in our example data. Secondly, we have also uploaded our preferred ImageJ macro DAPI_CYT_SPLIT.ijm for splitting the images into their respective DAPI and green fluorescent sub images. It is recommeneded that images be converted into .tiff format using the native ImageJ batch convert function. For exmaple, Zeiss microscopes running Zen software default save image files as .czi files, which can interfere with macros that are designed to batch process images. Converting all files into Tiff versions first can minimize the risk of such software issues. The uploaded pipeline GroupedSegmentation.cpipe is intended to be used in conjunction with this data set, and uses the automatically added DAPI and CYT tags at the end of the images to pair the images for primary and secondary object identification. 

Note. Our pipeline attempts and processes secondary objects using both watershed and propagation algorithms. Additionally, manual adjustment of the thresholds may be necessary to achieve optimum results for your images. The reason we choose to double calculate watershed and propagation algorithms is because different cell types have different affinities for CellTrackerGreen as a dye. GBM6 for example takes to it quite happily, while GBM43 will have dimmer signals. In an experimental design, the researcher is thus faced with the decision to introduce variance by extending the incubation time for GBM43 beyond the 30 minutes that we used for all our cell times, or to change the power settings on the microscope. We experimented with both and found that changing incubation times could lead to overexposure of cell images. Additionally, even then some cells would have secondary objects that covered the whole screen due to the algorithm being unable to differentiate clear cellular borders in the dimmer fluorescent signal. Doubling up on secondary object identification algorithms corrected for that,as propagation algorithms are more liberal and allowed clearer capture of the spindly structures common to fibroblast cell lines, while watershed methods allowed for clearer defined capture of the dim smaller shapes of certain glioblastoma cell lines. Image sets were then manually checked to see which method was suitable for each cell type, and then tags in the next section of the instructions were submitted to select the write set of segmented images. 

3. Installing and utilizing the VAMPIRE package



4. Running the requsite scripts to make your life easier when creating your own logistic regression and classifier.

5. Interpreting the data and fine tuning the general purposes. 
