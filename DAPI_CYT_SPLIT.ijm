//SplitChannelMacro

        inputFolder= getDirectory("Choose a Directory");
        print(inputFolder);
        outputGreen= inputFolder + "/GreenChannel/";
        outputBlue   = inputFolder + "/BlueChannel/";
        images= getFileList(inputFolder);
        File.makeDirectory(outputGreen);
        File.makeDirectory(outputBlue);

        for (i=0; i<images.length; i++) {
                setBatchMode(true); //batch mode
                inputPath= inputFolder + images[i];
                open (inputPath);
                imagesName=getTitle();
                print("Splitting Image: " + imagesName);
                run("Split Channels");
                selectWindow("C2-" + imagesName);
                saveAs("Tiff", outputBlue + imagesName + "_DAPI");
                close();
                selectWindow("C1-" + imagesName);
                saveAs("Tiff", outputGreen +imagesName + "_CYT");
                close();
                write("Conversion Complete");
       
}
setBatchMode(false);
