/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "imgproc.h"

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

inline int cudaDeviceInit(int argc, const char **argv) {
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name
            << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}


bool printfNPPinfo(int argc, char *argv[]) {
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

bool accept(fs::directory_entry entry) {
    if (!entry.is_regular_file()) return false;
    std::string ext = entry.path().extension().string();
    if (ext == ".pgm" || ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
        return true;
    }
    return false;
}

void convert(fs::path src, fs::path dst) {
    std::cout << src << " -> " << dst << std::endl;

    std::ifstream infile(src, std::ifstream::in);
    if (! infile.good()) {
        exit(EXIT_FAILURE);
    }


    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(src, oHostSrc);

    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    NppiSize oSrcSize = {(int) oDeviceSrc.width(), (int) oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize oSizeROI = {(int) oDeviceSrc.width(), (int) oDeviceSrc.height()};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

    int nBufferSize = 0;
    Npp8u *pScratchBufferNPP = 0;

    // get necessary scratch buffer size and allocate that much device memory
    NPP_CHECK_NPP(nppiFilterCannyBorderGetBufferSize(oSizeROI, &nBufferSize));

    checkCudaErrors( cudaMalloc((void **) &pScratchBufferNPP, nBufferSize) );

    // now run the canny edge detection filter
    // Using nppiNormL2 will produce larger magnitude values allowing for finer
    // control of threshold values while nppiNormL1 will be slightly faster.
    // Also, selecting the sobel gradient filter allows up to a 5x5 kernel size
    // which can produce more precise results but is a bit slower. Commonly
    // nppiNormL2 and sobel gradient filter size of 3x3 are used. Canny
    // recommends that the high threshold value should be about 3 times the low
    // threshold value. The threshold range will depend on the range of
    // magnitude values that the sobel gradient filter generates for a
    // particular image.

    Npp16s nLowThreshold = 72;
    Npp16s nHighThreshold = 256;

    if ((nBufferSize > 0) && (pScratchBufferNPP != 0)) {
        NPP_CHECK_NPP(nppiFilterCannyBorder_8u_C1R(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
            oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, NPP_FILTER_SOBEL,
            NPP_MASK_SIZE_3_X_3, nLowThreshold, nHighThreshold, nppiNormL2,
            NPP_BORDER_REPLICATE, pScratchBufferNPP));
    }

    cudaDeviceSynchronize();

    // free scratch buffer memory
    checkCudaErrors( cudaFree(pScratchBufferNPP) );

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(dst, oHostDst);
    std::cout << "Saved image: " << dst << std::endl;

    // nppiFree(oDeviceSrc.data());
    // nppiFree(oDeviceDst.data());
    //
    // nppiFree(oHostSrc.data());
    // nppiFree(oHostDst.data());
}

int main(int argc, char *argv[]) {
    printf("%s Starting...\n\n", argv[0]);

    try {
        char *srcDir;
        char *dstDir;

        cudaDeviceInit(argc, (const char **) argv);

        if (printfNPPinfo(argc, argv) == false) {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "input")) {
            getCmdLineArgumentString(argc, (const char **) argv, "input", &srcDir);
        } else {
            srcDir = (char *) "inputs";
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "output")) {
            getCmdLineArgumentString(argc, (const char **) argv, "output", &dstDir);
        } else {
            dstDir = (char *) "outputs";
        }

        for (const auto &src: fs::directory_iterator(srcDir)) {
            if (accept(src)) {
                std::string dstFn = src.path().stem().string() + std::string(".pgm"); // filename();
                fs::path dst = fs::path(dstDir) / dstFn;
                std::cout << dst << std::endl;
                convert(src.path(), dst);
            }
        }

        exit(EXIT_SUCCESS);
    } catch (npp::Exception &rException) {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    } catch (...) {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
