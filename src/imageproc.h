//
// Created by ryan on 1/7/25.
//

#ifndef IMAGEPROC_H
#define IMAGEPROC_H

#include <cuda.h>
#include <stdio.h>


// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

#endif //IMAGEPROC_H
