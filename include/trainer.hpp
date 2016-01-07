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

/* trainer.hpp */

#ifndef TRAINER_H
#define TRAINER_H


using namespace cv;

/*
 * Common routines
 */

Mat SelectPRFilters( const Mat PRFilters, const Mat w );
Mat get_desc( Mat Patch, int nAngleBins, float InitSigma, bool bNorm );
void ComputeStats( const int nChannels, const Mat& PRParams,
                   const Mat& Dists, const Mat& Labels, const Mat& w,
                   int& nPR, int &Dim, int& nzDim, float& FPR95, double& AUC, int MaxDim = -1 );
int TermProgress( double dfComplete , int nLastTick = -1 );

/*
 * CUDA
 */

namespace cv { namespace cuda {
namespace dlco {

void SubtractVectorsByRows( const cuda::GpuMat& src1, const cuda::GpuMat& src2, 
                            cuda::GpuMat& dst, cuda::Stream& _stream );

} // end namespace dlco
}} // end namespace cv & dlco

#endif
