/*
 * Copyright (c) 2014  Balint Cristian (cristian dot balint at gmail dot com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* pr-learn.cu */
/* PR lean cuda kernels */

#include "stdio.h"

#include <opencv2/core/core.hpp>
#include "opencv2/cudev.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/utility.hpp"


using namespace cv::cuda;
using namespace cv::cudev;

namespace cv { namespace cuda {
namespace dlco {

__global__ static void kSubtractVectorsByRows( const GlobPtrSz<float> src1,
                                               const GlobPtrSz<float> src2,
                                               GlobPtr<float> dst )
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while ( idx < src1.rows )
    {
      register float tsum = 0;
      for ( int i = 0; i < src2.rows; i++ )
      {
        register float rsum = src1.data[ idx ] + 1 - src2.data[ i ];
        tsum += (rsum > 0) ? rsum : 0;
      }
      dst.data[idx] = tsum;
      idx += gridDim.x * blockDim.x;
    }
}

void SubtractVectorsByRows( const cuda::GpuMat& src1, const cuda::GpuMat& src2, cuda::GpuMat& dst, Stream& _stream )
{
    const dim3 grid ( 4096, 1, 1 );
    const dim3 block(  512, 1, 1 );

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    dst = cuda::GpuMat( src1.rows, 1, CV_32F, Scalar::all(0) );
    kSubtractVectorsByRows<<< grid, block, 0, stream >>>( globPtr<float>(src1), globPtr<float>(src2), globPtr<float>(dst) );

    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

} // end namespace dlco
}} // end namespaces
