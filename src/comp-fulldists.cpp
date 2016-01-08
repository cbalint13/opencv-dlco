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

/* comp-fulldists.cpp */
/* Compute descriptor distances */
// C++ implementation of "script_comp_desc.m"

#include <fenv.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/hdf/hdf5.hpp>

#include "hdf5.h"

#include "trainer.hpp"


using namespace std;
using namespace cv;
using namespace hdf;

int main( int argc, char **argv )
{

    feenableexcept( FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW );

    // write chunks
    int sChunk = 128;
    int nLastTick = -1;

    // hyperparams
    bool bNorm = true;
    int nAngleBins = 8;
    float InitSigma = 1.4;

    bool help = false;
    const char *FltH5Filename = NULL;
    const char *ImgH5Filename = NULL;
    const char *DstH5Filename = NULL;

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
        if ( strcmp(argv[i], "-anglebins") == 0 )
        {
            nAngleBins = atoi(argv[i+1]);
            i++;
            continue;
        }
        if ( strcmp(argv[i], "-sigma") == 0 )
        {
            InitSigma = atof(argv[i+1]);
            i++;
            continue;
        }
        if ( strcmp(argv[i], "-norm") == 0 )
        {
            bNorm = atoi(argv[i+1]);
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
         if ( ! ImgH5Filename )
         {
            ImgH5Filename = argv[i];
            continue;
         }
         if ( ! DstH5Filename )
         {
            DstH5Filename = argv[i];
            continue;
         }
      }
    }

    if ( ( ! FltH5Filename ) ||
         ( ! ImgH5Filename ) ||
         ( ! DstH5Filename ) )
      help = true;

    if ( help )
    {
        cout << endl;
        cout << "Usage: comp-fulldists src_h5_filter_file src_h5_patches_file dst_h5_dist_file" << endl;
        cout << "       -anglebins <0-255, 8=default> " << endl;
        cout << "       -sigma <0.0-10.0, 1.4=default> " << endl;
        cout << "       -norm <0,1 1=default> " << endl;
        cout << endl;
        exit( 1 );
    }
    cout << "Bins: #" << nAngleBins
              << " Sigma: " << InitSigma
              << " bNorm: " << bNorm
    << endl;

    Mat TrainPairs;
    Ptr<HDF5> h5im = open( ImgH5Filename );
    cout << "Load Indices." << endl;
    h5im->dsread( TrainPairs, "Indices" );

    // get dimensions
    vector<int> ISize = h5im->dsgetsize( "Patches" );

    // should be rank 3
    CV_Assert( ISize.size() == 3 );

    // rank3 to rank2
    int nPatches = ISize[0];

    // create storage
    int pdims[3] = { ISize[0], ISize[1], ISize[2] };
    Mat Patches( 3, pdims, CV_8U );

    Mat Patch;
    cout << "Load Patches." << endl;
    for ( int i = 0; i < nPatches; i = i + sChunk )
    {
      int count = sChunk;
      // for the last incomplete chunk
      if ( i + sChunk > nPatches )
        count = nPatches - i;

      int ioffset[3] = {     i,        0,        0 };
      int icounts[3] = { count, ISize[1], ISize[2] };
      h5im->dsread( Patch, "Patches", ioffset, icounts );

      memcpy( &Patches.at<uchar>( i, 0, 0 ), Patch.data, Patch.total() );

      nLastTick = TermProgress( (double)i / (double)nPatches, nLastTick );
    }
    nLastTick = TermProgress( 1.0f, nLastTick );
    // close
    h5im->close();

    Ptr<HDF5> h5fl = open( FltH5Filename );

    // get dimensions
    vector<int> PSize = h5fl->dsgetsize( "PRFilters" );

    // should be rank 3
    CV_Assert( PSize.size() == 3 );

    // rank3 to rank2
    int nPFilters  = PSize[0];
    int FilterSize = PSize[1] * PSize[2];

    // create storage
    Mat PRFilters( nPFilters, FilterSize, CV_32F );
    Mat Filter;
    cout << "Load PRFilters." << endl;
    for ( int i = 0; i < nPFilters; i = i + sChunk )
    {
      int count = sChunk;
      // for the last incomplete chunk
      if ( i + sChunk > nPFilters )
        count = nPFilters - i;

      int poffset[3] = {     i,        0,        0 };
      int pcounts[3] = { count, PSize[1], PSize[2] };
      h5fl->dsread( Filter, "PRFilters", poffset, pcounts );

      memcpy( &PRFilters.at<float>( i, 0 ), Filter.data, Filter.total() * Filter.elemSize() );

      nLastTick = TermProgress( (double)i / (double)nPFilters, nLastTick );
    }
    nLastTick = TermProgress( 1.0f, nLastTick );

    // close
    h5fl->close();

    // create hdf5
    Ptr<HDF5> h5io = open( DstH5Filename );

    // create label space
    Mat Label( sChunk, 1, CV_8U );

    // create storage
    int lchunks[2] = { sChunk, 1 };
    h5io->dscreate( TrainPairs.rows, 1, CV_8U, "Label", 9, lchunks );

    // iterate through image patches
    cout << "Export Pair Labels: #" << TrainPairs.rows << endl;

    // Export thruth tables
    for ( int i = 0; i < TrainPairs.rows; i = i + sChunk )
    {
      int chunk = 0;
      #pragma omp parallel for schedule(dynamic) shared(chunk)
      for ( int k = 0; k < sChunk; k++ )
      {
         if ( ( i + k ) >= TrainPairs.rows )
           continue;
         // 3DPointID equals
         if ( TrainPairs.at<int32_t>( i + k, 1 )
          ==  TrainPairs.at<int32_t>( i + k, 3 ) )
           Label.at<uchar>( k ) = 1;
         else
           Label.at<uchar>( k ) = 0;
         {
           #pragma omp atomic
           chunk++;
         }
      }
      int offset[2] = {     i, 0 };
      int counts[2] = { chunk, 1 };
      h5io->dswrite( Label, "Label",  offset, counts );

      nLastTick = TermProgress( (double) ( i + chunk )
                / (double) TrainPairs.rows, nLastTick );
    }
    nLastTick = TermProgress( 1.0f, nLastTick );

    cuda::GpuMat prfilters;
    Mat Dist( 1, PRFilters.rows / 8, CV_32F );

    // query GPU
    cuda::DeviceInfo info( cuda::getDevice() );
    cout << endl;
    cout << "Found GPU: "<< info.name() << endl;
    cout << "Compute Capability: " << info.majorVersion()
              << "." <<info.minorVersion() <<  endl;
    cout << endl;

    // set active GPU
    cuda::setDevice(0);

    // filters on GPU are permanent
    prfilters.upload( PRFilters );

    cout << "Start Compute L2 distances." << endl;

    // create dataset
    Mat Dists( sChunk, PRFilters.rows / 8, CV_32F );

    // create storage
    int dchunks[2] = { sChunk, PRFilters.rows / 8 };
    h5io->dscreate( TrainPairs.rows, PRFilters.rows / 8,
                    CV_32F, "Distance", 9, dchunks );

    int64 startTime = getTickCount();
    for ( int i = 0; i < TrainPairs.rows; i = i + sChunk )
    {
      int chunk = 0;
      #pragma omp parallel for schedule(dynamic) shared(chunk)
      for ( int k = 0; k < sChunk; k++ )
      {

        if ( ( i + k ) >= TrainPairs.rows )
          continue;

        Mat Patch1( Patches.size[1], Patches.size[2],
                    CV_8U, &Patches.at<uchar>(
                    TrainPairs.at<int32_t>( i + k, 0 ), 0, 0) );

        Mat Patch2( Patches.size[1], Patches.size[2],
                    CV_8U, &Patches.at<uchar>(
                    TrainPairs.at<int32_t>( i + k, 2 ), 0, 0) );

        cuda::GpuMat dist;
        cuda::GpuMat L1, L2;
        cuda::GpuMat dst1, dst2;
        cuda::GpuMat dsc1, dsc2;

        cuda::Stream stream;

        // % patch descriptors (each column corresponds to a PR)
        Mat Desc1 = get_desc( Patch1, nAngleBins, InitSigma, bNorm );
        Mat Desc2 = get_desc( Patch2, nAngleBins, InitSigma, bNorm );

        // upload data
        dsc1.upload( Desc1, stream );
        dsc2.upload( Desc2, stream );

        // % Desc = (PR * PatchTrans)';
        cuda::gemm( prfilters, dsc1, 1.0f, cuda::GpuMat(), 0, dst1, 0, stream );
        cuda::gemm( prfilters, dsc2, 1.0f, cuda::GpuMat(), 0, dst2, 0, stream );

        //% Desc = min(Desc, 1);
        cuda::threshold( dst1, dsc1, 1.0f, 1.0f, CV_THRESH_TRUNC, stream );
        cuda::threshold( dst2, dsc2, 1.0f, 1.0f, CV_THRESH_TRUNC, stream );

        // % PR distances
        cuda::subtract( dsc2, dsc1, L1, cuda::GpuMat(), CV_32F, stream );
        cuda::pow( L1, 2, L2, stream );

        // % PR weights are shared across PR rings
        cuda::reduce( L2, L1, 1, CV_REDUCE_SUM, CV_32F, stream );
        cuda::reduce( L1.reshape( 0, PRFilters.rows / 8 ),
                      dist, 1, CV_REDUCE_SUM, CV_32F, stream );

        // download data
        dist.download( Dist , stream );

        stream.waitForCompletion();

        memcpy( &Dists.at<float>( k, 0 ),
                 Dist.data, Dist.total() * Dist.elemSize() );

        {
          #pragma omp atomic
          chunk++;
        }
      }
      // save chunk in HDF5
      int offset[2] = {     i,                  0 };
      int counts[2] = { chunk, PRFilters.rows / 8 };
      h5io->dswrite( Dists, "Distance",  offset, counts );

      if ( !checkRange(Dists) )
      {
        cout << "\nDist contains NaN\n";
        exit(-1);
      }
      printf( "\rStep: %i / %i", i + chunk, TrainPairs.rows );
      fflush(stdout);
      chunk = 0;
    }

    int64 endTime = getTickCount();
    cout << "\nDone." << endl << endl ;
    printf( "Total: %.09f sec\n\n", ( endTime - startTime ) / frequency );

    // close
    h5io->close();

    return 0;
}
