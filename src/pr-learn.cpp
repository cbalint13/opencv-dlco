/*
 * Copyright (c) 2014, Karen Simonyan
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
 *
 * C++ code by:  2014, Balint Cristian (cristian dot balint at gmail dot com)
 *
 */

/* pr-learn.cpp */
/* Learn Pool Region Filters */
// C++ implementation of "script_learn_PR.m"


#include <fenv.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/hdf/hdf5.hpp>

#include "trainer.hpp"


using namespace std;
using namespace cv;
using namespace hdf;

int main( int argc, char **argv )
{

    feenableexcept( FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW );

    // size of chunks
    int sChunk = 128;
    int nLastTick = -1;

    // % TrainSetName = 'yosemite';
    // % mu = 0.25;
    // % gamma = 4;

    // % TrainSetName = 'notredame';
    // % mu = 0.15;
    // % gamma = 0.5;

    // % TrainSetName = 'liberty';
    // % mu = 0.5;
    // % gamma = 8;

    // train set
    const float nDiv = 0.80; // 80% of data

    // hyperparams
    float mu = 0.025f;
    float gamma = 0.10f;
    unsigned int nIter = 5000000;
    unsigned int LogStep = 100000;

    bool help = false;
    const char *FltH5Filename = NULL;
    const char *DistsH5Filename = NULL;
    const char *OutputH5Filename = NULL;

    double frequency = getTickFrequency();

    if ( argc < 1 )
        exit( -argc );

    // parse arguments
    for ( int i = 1; i < argc; i++ )
    {
      if ( argv[i][0] == '-' )
      {
        if ( strcmp(argv[i], "-help") == 0 )
        {
            help = true;
            continue;
        }
        if ( strcmp(argv[i], "-mu") == 0 )
        {
            mu = atof(argv[i+1]);
            i++;
            continue;
        }
        if ( strcmp(argv[i], "-gamma") == 0 )
        {
            gamma = atof(argv[i+1]);
            i++;
            continue;
        }
        if ( strcmp(argv[i], "-iters") == 0 )
        {
            nIter = atoi(argv[i+1]);
            i++;
            continue;
        }
        cout << "ERROR: Invalid " << argv[i] << " option." << endl;
        help = true;
      } else if ( argv[i][0] != '-' )
      {
         if ( ! FltH5Filename )
         {
            FltH5Filename = argv[i];
            continue;
         }
         if ( ! DistsH5Filename )
         {
            DistsH5Filename = argv[i];
            continue;
         }
         if ( ! OutputH5Filename )
         {
            OutputH5Filename = argv[i];
            continue;
         }
      }
    }

    if ( ( ! DistsH5Filename ) ||
         ( ! FltH5Filename ) ||
         ( ! OutputH5Filename ) )
      help = true;

    if ( help )
    {
        cout << endl;
        cout << "Usage: pr-learn  src_h5_filter_file" << endl;
        cout << "       src_h5_dist_file dst_h5_output_file" << endl;
        cout << "       -mu <0.0-1.0, 0.025=default> " << endl;
        cout << "       -gamma <0.0-10.0, 0.10=default> " << endl;
        cout << "       -iters <0-N, 5000000=default> " << endl;
        cout << endl;
        exit( 1 );
    }
    cout << "mu: " << mu
              << " gamma: " << gamma
              << " nIters: " << nIter
    << endl;


    Mat PRParams, RingParams;
    Ptr<HDF5> h5fl = open( FltH5Filename );

    cout << "Load PRParams." << endl;
    h5fl->dsread( PRParams, "PRParams" );

    cout << "Load RingParams." << endl;
    h5fl->dsread( RingParams, "RingParams" );

    h5fl->close();

    // open hdf5 files
    Ptr<HDF5> h5io = open( DistsH5Filename );

    // get dimensions
    vector<int> DSize = h5io->dsgetsize( "Distance" );

    int nDists  = DSize[0];
    int FeatDim = DSize[1];

    Mat w =     Mat::zeros( 1, FeatDim, CV_32F );
    Mat dfAvg = Mat::zeros( 1, FeatDim, CV_32F );

    // create storage space
    Mat Labels( nDists, 1, CV_8U );
    Mat Dists( nDists, FeatDim, CV_32F );

    cout << "Load Labels: " << nDists << endl;
    cout << "Load Distances: " << nDists << " x " << FeatDim << endl;

    Mat Label, Dist;
    for ( int i = 0; i < nDists; i = i + sChunk )
    {
      int count = sChunk;
      // for the last incomplete chunk
      if ( i + sChunk > nDists )
        count = nDists - i;

      int loffset[2] = {     i, 0 };
      int lcounts[2] = { count, 1 };
      h5io->dsread( Label, "Label", loffset, lcounts );

      int doffset[2] = {     i,       0 };
      int dcounts[2] = { count, FeatDim };
      h5io->dsread( Dist, "Distance", doffset, dcounts );

      memcpy( &Labels.at<uchar>( i, 0 ), Label.data,  Label.total() );
      memcpy( &Dists.at<float>( i, 0 ), Dist.data, Dist.total() * Dist.elemSize() );

      nLastTick = TermProgress( (double)i / (double)nDists, nLastTick );
    }
    nLastTick = TermProgress( 1.0f, nLastTick );

    // close
    h5io->close();

    vector<int> IdxPos, IdxNeg;
    // aquire index of positive/negative samples
    for ( int i = 0; i < nDists; i++ )
    {
      if ( Labels.at<uchar>(i) == 1 ) IdxPos.push_back(i);
      if ( Labels.at<uchar>(i) == 0 ) IdxNeg.push_back(i);
    }

    cout << "Positive samples #" << IdxPos.size() << endl;
    cout << "Negative samples #" << IdxNeg.size() << endl;

    RNG rng( 2215 );
    randShuffle( IdxPos );
    randShuffle( IdxNeg );

    Mat f, df;
    Mat FeatDiff( 1, FeatDim, CV_32F );

    float Obj_Best = FLT_MAX;
    Mat w_Best =  Mat::zeros( 1, FeatDim, CV_32F );

    size_t nPosTrn = IdxPos.size() * nDiv;
    size_t nNegTrn = IdxNeg.size() * nDiv;
    size_t nPosVal = IdxPos.size() - nPosTrn;
    size_t nNegVal = IdxNeg.size() - nNegTrn;

    cout << "Positive train #" << nPosTrn << endl;
    cout << "Negative train #" << nNegTrn << endl;
    cout << "Positive valid #" << nPosVal << endl;
    cout << "Negative valid #" << nNegVal << endl;

    // prepare validation subset
    Mat PosVal( nPosVal, FeatDim, CV_32F );
    Mat NegVal( nNegVal, FeatDim, CV_32F );

    // construct validation set
    for ( unsigned int i = nPosTrn; i < IdxPos.size(); i++ )
      memcpy( &PosVal.at<float>( i - nPosTrn, 0 ),
              &Dists.at<float>( IdxPos[i], 0, 0 ),
              FeatDim * Dists.elemSize() );
    for ( unsigned int i = nNegTrn; i < IdxNeg.size(); i++ )
      memcpy( &NegVal.at<float>( i - nNegTrn, 0 ),
              &Dists.at<float>( IdxNeg[i], 0, 0 ),
              FeatDim * Dists.elemSize() );

    // query GPU
    cuda::DeviceInfo info( cuda::getDevice() );
    cout << endl;
    cout << "Found GPU: "<< info.name() << endl;
    cout << "Compute Capability: " << info.majorVersion()
              << "." <<info.minorVersion() <<  endl;
    cout << endl;

    // set active GPU
    cuda::setDevice(0);

    cuda::GpuMat W;
    cuda::Stream stream;
    cuda::GpuMat POSVal, NEGVal;
    cuda::GpuMat POSDist, NEGDist;

    POSVal.upload( PosVal, stream );
    NEGVal.upload( NegVal, stream );

    stream.waitForCompletion();

    unsigned int step = 0;
    unsigned int iPos, iNeg;
    int64 trainStartTime = getTickCount();
    for ( unsigned int t = 0; t <= nIter; t++ )
    {
      iPos = rng.uniform( 0, nPosTrn );
      iNeg = rng.uniform( 0, nNegTrn );

      subtract( Dists.row( IdxPos[iPos] ),
                Dists.row( IdxNeg[iNeg] ),
                FeatDiff );

      // % compute loss subgradient
      gemm( w, FeatDiff, 1.0f, noArray(), 0.0f, f, GEMM_2_T );

      // % subgradient average
      dfAvg = t * dfAvg / ( t + 1 );

      if ( f.at<float>(0) > -1 )
        scaleAdd( FeatDiff, (double) 1 / ( t + 1 ), dfAvg, dfAvg );

      // % update w
      w = -sqrt( t + 1 ) / gamma * ( dfAvg + mu );
      w = max( w, 0.0f );

      if ( step == LogStep )
      {
        int64 trainEndTime = getTickCount();
        int64 validStartTime = getTickCount();

        float Regul;
        float Loss = 0.0f;
        float LossVal = 0.0f;

        W.upload( w, stream );

        // % compute the objective on the validation set subsample
        Mat PosDist, NegDist;

        cuda::gemm( POSVal, w, 1, cuda::GpuMat(), 0, POSDist, GEMM_2_T, stream );
        cuda::gemm( NEGVal, w, 1, cuda::GpuMat(), 0, NEGDist, GEMM_2_T, stream );

        cuda::GpuMat dst;
        // % sum( max( (PosDist.at<float>(i) + 1.0f - NegDist), 0.0f ) )
        cuda::dlco::SubtractVectorsByRows( POSDist, NEGDist, dst, stream );

        // reduce final  sums
        Loss = cuda::sum( dst )[0];

        stream.waitForCompletion();

        LossVal = Loss / (nPosVal * nNegVal);

        // % compute the regulariser
        Regul = mu * sum( abs( w ) )[0];

        int64 validEndTime = getTickCount();

        // % save w if it's the current minimiser of Obj
        if ( ( LossVal + Regul ) < Obj_Best )
        {
            Obj_Best = LossVal + Regul;

            w_Best = w.clone();

            printf("Best: %i  Loss: %.6f Regul: %.6f Obj: %.6f (%.6f)  NNZ: %i (%i)  Ttime: %.4f Vtime: %.4f\n",
                   t, LossVal, Regul, (LossVal + Regul), Obj_Best, countNonZero(w), countNonZero(w_Best),
                   ( trainEndTime - trainStartTime ) / frequency,
                   ( validEndTime - validStartTime ) / frequency );

            /*
             *  best model full statistics
             */

            const int MaxDim = 640;
            const int nChannels = 8;

            double AUC;
            float FPR95;
            int nPR, Dim, nzDim;

            ComputeStats( nChannels, PRParams, Dists, Labels, w, nPR, Dim, nzDim, FPR95, AUC );

            /*
             * save best results
             * if fit max dim
             */

            if ( Dim <= MaxDim )
            {
              // open hdf5 files
              Ptr<HDF5> h5iw = open( OutputH5Filename );

              // create set with unlimited rows
              if ( ! h5iw->hlexists( "w" ) )
              {
                int chunks[2] = { 1, w.cols };
                h5iw->dscreate( HDF5::H5_UNLIMITED, w.cols, CV_32F, "w", 9, chunks );
              }

              // get actual size
              vector<int> wsize = h5iw->dsgetsize( "w" );
              // append to last row
              int offset[2] = { wsize[0], 0 };
              h5iw->dsinsert( w_Best, "w", offset );

              // close
              h5iw->close();

              // log as saved
              printf( "Stat: nPR #%i (#%i) Dim/MaxDim [%i/%i] AUC: %f FPR95: %.2f [saved]\n",
                      nPR, nzDim, Dim, MaxDim, AUC, FPR95*100 );
            } else {
              // log without save
              printf( "Stat: nPR #%i (#%i) Dim/MaxDim [%i/%i] AUC: %f FPR95: %.2f\n",
                      nPR, nzDim, Dim, MaxDim, AUC, FPR95*100 );
            }
        } else {
            printf("Step: %i  Loss: %.6f Regul: %.6f Obj: %.6f (%.6f)  NNZ: %i (%i)  Ttime: %.4f Vtime: %.4f\n",
                   t, LossVal, Regul, (LossVal + Regul), Obj_Best, countNonZero(w), countNonZero(w_Best),
                   ( trainEndTime - trainStartTime ) / frequency,
                   ( validEndTime - validStartTime ) / frequency );
        }
        // flush i/o
        cout << flush;

        step = 0;
        trainStartTime = getTickCount();
      }
      step++;
    }

    IdxPos.clear();
    IdxNeg.clear();
    Dists.release();
    PosVal.release();
    NegVal.release();

    return 0;
}
