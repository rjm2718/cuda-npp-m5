#include "imageproc.h"

// #include <Exceptions.h>
// #include <ImageIO.h>
// #include <ImagesCPU.h>
// #include <ImagesNPP.h>

#include <iostream>
#include <npp.h>
#include <cuda_runtime.h>
#include <opencv4/opencv2/opencv.hpp>

#include <string.h>
#include <fstream>
#include <filesystem>

using namespace std;

void checkCudaInit(int dev) {

    const NppLibraryVersion *libVer = nppGetLibVersion();
    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);

    if (deviceProp.major < 1) { // TODO
        printf("*** compute capability insufficient ***\n");
        exit(-1);
    }
}

void helpExit(int argc, char *argv[]) {
    printf("Usage: %s <input directory> <output directory>\n", argv[0]);
    printf("All files in input will be processed and written to output.\n");
    exit(EXIT_FAILURE);
}

bool accept(filesystem::directory_entry entry) {
    if (! entry.is_regular_file()) return false;
    string ext = entry.path().extension().string();
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
        printf("accepting: %s\n", ext.c_str());
        return true;
    }
    return false;
}

int main(int argc, char *argv[])
{
    try
    {
        string sFilename;
        char *filePath;

        // argv[1] is input directory, argv[2] is output
        if (argc != 3) {
            helpExit(argc, argv);
        }

        if (!filesystem::exists(argv[1]) || !filesystem::is_directory(argv[1])) {
            cout << "src directory doesn't exist" << "\n";
            helpExit(argc, argv);
        }
        if (!filesystem::exists(argv[2]) || !filesystem::is_directory(argv[2])) {
            cout << "target directory doesn't exist" << "\n";
            helpExit(argc, argv);
        }

        checkCudaInit(0);


        for (const auto & entry : filesystem::directory_iterator(argv[1])) {
            if (accept(entry)) {
                std::cout << entry.path() << std::endl;
            }
        }

/*
        // read files from input dir

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "Lena.pgm";
        }

        // if we specify the filename at the command line, then we only test
        // sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "nppiRotate opened: <" << sFilename.data()
                      << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "nppiRotate unable to open: <" << sFilename.data() << ">"
                      << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_rotate.pgm";

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output",
                                     &outputFilePath);
            sResultFilename = outputFilePath;
        }

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        // load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);
        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        // create struct with the ROI size
        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {0, 0};
        NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        // Calculate the bounding box of the rotated image
        NppiRect oBoundingBox;
        double angle = 45.0; // Rotation angle in degrees
        NPP_CHECK_NPP(nppiGetRotateBound(oSrcSize, angle, &oBoundingBox));

        // allocate device image for the rotated image
        npp::ImageNPP_8u_C1 oDeviceDst(oBoundingBox.width, oBoundingBox.height);

        // Set the rotation point (center of the image)
        NppiPoint oRotationCenter = {(int)(oSrcSize.width / 2), (int)(oSrcSize.height / 2)};

        // run the rotation
        NPP_CHECK_NPP(nppiRotate_8u_C1R(
            oDeviceSrc.data(), oSrcSize, oDeviceSrc.pitch(), oSrcOffset,
            oDeviceDst.data(), oDeviceDst.pitch(), oBoundingBox, angle, oRotationCenter,
            NPPI_INTER_NN));

        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());

*/
        exit(EXIT_SUCCESS);
    }
    // catch (npp::Exception &rException)
    // {
    //     std::cerr << "Program error! The following exception occurred: \n";
    //     std::cerr << rException << std::endl;
    //     std::cerr << "Aborting." << std::endl;
    //
    //     exit(EXIT_FAILURE);
    // }
    catch (const exception &e) {
        std::cout << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
