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
 * C++ code by:  2015, Balint Cristian (cristian dot balint at gmail dot com)
 *
 */

/* compute-unprojdists.cpp */
/* Compute unprojected descriptors */
// C++ implementation of "script_comp_unproj_desc.m"

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

    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);

    int widx = 0;

    // hyperparams
    const bool bNorm = true;
    const int nAngleBins = 8;
    const float InitSigma = 1.4;

    // write chunks
    int sChunk = 128;
    int nLastTick = -1;

    bool help = false;
    const char *FltH5Filename = NULL;
    const char *ImgH5Filename = NULL;
    const char *PrjH5Filename = NULL;
    const char *OutH5Filename = NULL;

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
        if ( strcmp(argv[i], "-prj") == 0 )
        {
            PrjH5Filename = argv[i+1];
            i++; continue;
        }
        if ( strcmp(argv[i], "-id") == 0 )
        {
            widx = atoi(argv[i+1]);
            i++; continue;
        }
        if ( strcmp(argv[i], "-out") == 0 )
        {
            OutH5Filename = argv[i+1];
            i++; continue;
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
         help = true;
      }
    }

    if ( ( ! FltH5Filename ) ||
         ( ! ImgH5Filename ) ||
         ( ! OutH5Filename ) ||
         ( ! PrjH5Filename ) )
      help = true;


    if ( help )
    {
        cout << endl;
        cout << "Usage: comp-uprjdists src_h5_filter_file src_h5_img_file" << endl;
        cout << "        -prj src_h5_prj_file -id src_h5_prj_matrix_rowid -out dsc_h5_filename" << endl;
        cout << endl;
        exit( 1 );
    }

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

    printf( "ImageSet: [%s]\n", ImgH5Filename );

    /*
     * Image Set
     */

    Ptr<HDF5> h5im = open( ImgH5Filename );

    // get dimensions
    vector<int> ISize = h5im->dsgetsize( "Patches" );

    // should be rank 3
    CV_Assert( ISize.size() == 3 );

    // rank3 to rank2
    int nPatches  = ISize[0];
    int PatchSize = ISize[1] * ISize[2];

    // create storage
    Mat Patches( nPatches, PatchSize, CV_8U );

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

      memcpy( &Patches.at<uchar>( i, 0 ), Patch.data, Patch.total() );

      nLastTick = TermProgress( (double)i / (double)nPatches, nLastTick );
    }
    nLastTick = TermProgress( 1.0f, nLastTick );

    // close
    h5im->close();

    printf( "Load Learnt Filters: [%s]#%i\n", PrjH5Filename, widx );

    // % w = repmat(w', 8, 1);
    // % w = w(:);
    // % NZIdx = (w > 0) & any(PRFilters, 2);
    // % w = w(NZIdx);
    // % PRFilters = PRFilters(NZIdx, :);

    Mat w, sPRFilters;

    Ptr<HDF5> h5io = open( PrjH5Filename );
    vector<int> dims = h5io->dsgetsize("w");

    // only specified w matrix
    int offset[2] = { widx,       0 };
    int counts[2] = {    1, dims[1] };
    h5io->dsread( w, "w", offset, counts );

    h5io->close();

    sPRFilters = SelectPRFilters( PRFilters, w );

    printf("PRFilters: %i x %i\n", sPRFilters.rows, sPRFilters.cols);
    printf("Descriptor size: %i\n", sPRFilters.rows * 8);

    // create storage chunk
    Mat Descs( sChunk, sPRFilters.rows * 8, CV_32F );

    Ptr<HDF5> h5id = open( OutH5Filename );

    // create hdf storage
    int dchunks[2] = { sChunk, 1 };
    h5id->dscreate( Patches.rows, Descs.cols, CV_32F, "DescUnproj", 9, dchunks );

    int64 startTime = getTickCount();
    for ( int i = 0; i < Patches.rows; i = i + sChunk )
    {
      int chunk = 0;
      #pragma omp parallel for schedule(dynamic) shared(chunk)
      for ( int k = 0; k < sChunk; k++ )
      {

        if ( ( i + k ) >= Patches.rows )
          continue;

        Mat Patch( 64, 64, CV_8U, Patches.ptr( i + k, 0 ) );
        Mat PatchTrans = get_desc( Patch, nAngleBins, InitSigma, bNorm );

        Mat Desc = sPRFilters * PatchTrans;

        Desc = min( Desc.reshape( 0, 1 ), 1.0f );

        memcpy( &Descs.at<float>( k, 0 ),
                 Desc.data, Desc.total() * Desc.elemSize() );


        {
          #pragma omp atomic
          chunk++;
        }
      }
      // save chunk in HDF5
      int offset[2] = {     i,                   0 };
      int counts[2] = { chunk, sPRFilters.rows * 8 };
      h5id->dswrite( Descs, "DescUnproj",  offset, counts );

      if ( !checkRange(Descs) )
      {
        cout << "\nDesc contains NaN\n";
        exit(-1);
      }
      printf( "\rStep: %i / %i", i + chunk, Patches.rows );
      fflush(stdout);
      chunk = 0;
    }

    int64 endTime = getTickCount();
    cout << "\nDone." << endl << endl ;
    printf( "Total: %.09f sec\n\n", ( endTime - startTime ) / frequency );

    h5id->close();

    return 0;
}
