/*
 * Copyright (c) 2016, Balint Cristian (cristian dot balint at gmail dot com)
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
 */

/* export-opencv.cpp */
/* Export .C header to OpenCV */

#include <fenv.h>
#include <stdio.h>
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

    int widx = -1;
    bool help = false;
    const char *FltH5Filename = NULL;
    const char *PrgH5Filename = NULL;
    const char *PrjH5Filename = NULL;
    const char *OutHeadername = NULL;

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
        if ( strcmp(argv[i], "-flt") == 0 )
        {
            FltH5Filename = argv[i+1];
            i++; continue;
        }
        if ( strcmp(argv[i], "-prg") == 0 )
        {
            PrgH5Filename = argv[i+1];
            i++; continue;
        }
        if ( strcmp(argv[i], "-id") == 0 )
        {
            widx = atoi(argv[i+1]);
            i++; continue;
        }
        if ( strcmp(argv[i], "-prj") == 0 )
        {
            PrjH5Filename = argv[i+1];
            i++; continue;
        }
        cout << "ERROR: Invalid " << argv[i] << " option." << endl;
        help = true;
      } else if ( argv[i][0] != '-' )
      {
         if ( ! OutHeadername )
         {
            OutHeadername = argv[i];
            continue;
         }
         help = true;
      }
    }

    if ( ( widx == -1 ) ||
         ( ! FltH5Filename ) ||
         ( ! PrgH5Filename ) ||
         ( ! PrjH5Filename ) ||
         ( ! OutHeadername ) )
      help = true;


    if ( help )
    {
        cout << endl;
        cout << "Usage: export-opencv -flt src_h5_filter_file" << endl;
        cout << "       -prg src_h5_prg_file -id src_h5_prg_matrix_rowid" << endl;
        cout << "       -prj src_h5_prj_file out_cc_headerfile" << endl;
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
    cout << "Load PRFilters:" << endl;
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

    printf( "Load Learnt Filters: [%s]#%i\n", PrgH5Filename, widx );

    Mat w;
    Ptr<HDF5> h5ir = open( PrgH5Filename );
    vector<int> dims = h5ir->dsgetsize("w");

    // only specified w matrix
    int offset[2] = { widx,       0 };
    int counts[2] = {    1, dims[1] };
    h5ir->dsread( w, "w", offset, counts );

    h5ir->close();

    printf( "Load Learnt Projections: [%s]\n", PrjH5Filename );

    Mat W;
    Ptr<HDF5> h5ip = open( PrjH5Filename );
    h5ip->dsread( W, "W" );
    h5ip->close();

    // % w = repmat(w', 8, 1);
    // % w = w(:);
    // % NZIdx = (w > 0) & any(PRFilters, 2);
    // % w = w(NZIdx);
    // % PRFilters = PRFilters(NZIdx, :);

    Mat sPRFilters = SelectPRFilters( PRFilters, w );

    printf( "PRFilters: %i x %i [%i]\n", sPRFilters.rows, sPRFilters.cols, sPRFilters.rows * 8 );
    printf( "PJFilters: %i x [%i]\n", W.rows, W.cols );

    if ( W.cols != sPRFilters.rows * 8 )
    {
      printf("ERROR: PJFilters [%i] not agree PRFilters [%i].\n", sPRFilters.rows * 8, W.cols);
      exit(0);
    }

    // open C header file for write
    FILE * out = fopen( OutHeadername, "w" );

    fprintf( out, "// generated VGG pooling region filters & projection parameters\n" );

    fprintf( out, "\n" );

    fprintf( out, "// PR: [%s]#%i\n", PrgH5Filename, widx );
    fprintf( out, "// PJ: [%s]\n", PrjH5Filename );

    fprintf( out, "\n" );
    fprintf( out, "\n" );

    fprintf( out, "// PR orig rows\n" );
    fprintf( out, "static const int PRrows = %i;\n", sPRFilters.rows );

    fprintf( out, "\n" );

    fprintf( out, "// PR orig cols\n" );
    fprintf( out, "static const int PRcols = %i;\n", sPRFilters.cols );

    fprintf( out, "\n" );

    int idx, start, count, total;

    fprintf( out, "// PR indexes & len\n" );
    fprintf( out, "static const unsigned int PRidx[] =\n" );

    idx = 0;
    start = -1; count = 0;
    total = countNonZero( sPRFilters );
    for( int r = 0; r < sPRFilters.rows; r++ )
    {
      for( int c = 0; c < sPRFilters.cols; c++ )
      {
        // nonzero element
        if ( sPRFilters.at<float>( r, c ) != 0.0f )
        {
          // mark start
          total--; count++;
          if ( start == -1 ) start = r*sPRFilters.cols + c;
        }
        if ( ( total == 0 ) ||
             ( sPRFilters.at<float>( r, c ) == 0.0f ) )
        {
          // dump block
          if ( count != 0 )
          {
            idx++;
            if ( idx == 1 ) fprintf( out, "{\n " );
            fprintf( out, "0x%x,0x%X", start, count );

            if ( total == 0 )
            {
              fprintf( out, "\n};\n" );
              break;
            }
            else fprintf( out, "," );

            if ( idx % 8 == 0 ) fprintf( out, "\n " );
            // cancel marker
            start = -1; count = 0;
          }
        }
      }
    }

    fprintf( out, "\n" );

    fprintf( out, "// PR matrix\n" );
    fprintf( out, "static const unsigned int PR[] =\n" );

    count = 1;
    total = countNonZero( sPRFilters );
    for( int r = 0; r < sPRFilters.rows; r++ )
    {
      for( int c = 0; c < sPRFilters.cols; c++ )
      {
        if ( r + c == 0 ) fprintf( out, "{\n " );
        if ( sPRFilters.at<float>( r, c ) != 0.0f )
        {
          fprintf( out, "0x%08x", *(uint *) sPRFilters.ptr<float>( r, c ) );

          // close matrix
          if ( count == total )
          {
            fprintf( out, "\n};\n" );
            break;
          }
          else fprintf( out, "," );

          if ( count % 8 == 0 ) fprintf( out, "\n " );

          count++;
        }
      }
    }

    fprintf( out, "\n" );
    fprintf( out, "\n" );

    fprintf( out, "// PJ orig rows\n" );
    fprintf( out, "static const int PJrows = %i;\n", W.rows );

    fprintf( out, "\n" );

    fprintf( out, "// PJ orig cols\n" );
    fprintf( out, "static const int PJcols = %i;\n", W.cols );

    fprintf( out, "\n" );

    fprintf( out, "// PJ indexes & len\n" );
    fprintf( out, "static const unsigned int PJidx[] =\n" );

    idx = 0;
    start = -1; count = 0;
    total = countNonZero( W );
    for( int r = 0; r < W.rows; r++ )
    {
      for( int c = 0; c < W.cols; c++ )
      {
        // nonzero element
        if ( W.at<float>( r, c ) != 0.0f )
        {
          // mark start
          total--; count++;
          if ( start == -1 ) start = r*W.cols + c;
        }
        if ( ( total == 0 ) ||
             ( W.at<float>( r, c ) == 0.0f ) )
        {
          // dump block
          if ( count != 0 )
          {
            idx++;
            if ( idx == 1 ) fprintf( out, "{\n " );
            fprintf( out, "0x%x,0x%X", start, count );

            if ( total == 0 )
            {
              fprintf( out, "\n};\n" );
              break;
            }
            else fprintf( out, "," );

            if ( idx % 8 == 0 ) fprintf( out, "\n " );
            // cancel marker
            start = -1; count = 0;
          }
        }
      }
    }

    fprintf( out, "\n" );

    fprintf( out, "// PJ sparse elements\n" );
    fprintf( out, "static const unsigned int PJ[] =\n" );

    count = 1;
    total = countNonZero( W );
    for( int r = 0; r < W.rows; r++ )
    {
      for( int c = 0; c < W.cols; c++ )
      {
        if ( r + c == 0 ) fprintf( out, "{\n " );
        if ( W.at<float>( r, c ) != 0.0f )
        {
          fprintf( out, "0x%08x", *(uint *) W.ptr<float>( r, c ) );

          // close matrix
          if ( count == total )
          {
            fprintf( out, "\n};\n" );
            break;
          }
          else fprintf( out, "," );

          if ( count % 8 == 0 ) fprintf( out, "\n " );

          count++;
        }
      }
    }

    fprintf( out, "\n" );

    // close file
    fclose( out );

    return 0;
}
