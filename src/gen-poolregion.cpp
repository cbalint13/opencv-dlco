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

/* generate-poolregions.cpp */
/* Generates a candidate set of pooling region Gaussian filters */
// C++ implementation of "script_gen_PR.m"

#include <fenv.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/hdf/hdf5.hpp>

#include "trainer.hpp"


using namespace std;
using namespace cv;
using namespace hdf;

static Mat get_PR_filter( int PatchSize, double x0, double y0, double sigma )
{
    int rExt = ceil(3.0f * sigma);

    // % extended patch
    Mat PR = Mat::zeros( PatchSize + 2*rExt, PatchSize + 2*rExt, CV_32F );

    x0 = x0 + 0.5f * (1.0f + PatchSize) + rExt;
    y0 = y0 + 0.5f * (1.0f + PatchSize) + rExt;

    // % compute weights
    for ( int y = floor(y0 - 3.0f * sigma); y <= ceil(y0 + 3.0f * sigma); y++ )
        for ( int x = floor(x0 - 3.0f * sigma); x <= ceil(x0 + 3.0f * sigma); x++ )
        {
            double dx = (x - x0);
            double dy = (y - y0);
            double r2 = dx*dx + dy*dy;
            PR.at<float>(x-1, y-1) = (float) exp(-r2 / (2 * sigma*sigma));
         }

    // % crop the original patch borders
    // PR = PR(rExt + 1 : PatchSize + rExt, rExt + 1 : PatchSize + rExt);
    Rect Box( rExt, rExt, PatchSize, PatchSize );
    PR = PR( Box ).clone();

    // % normalise to a unit sum
    PR /= sum(PR)[0];

    return PR;
}

int main( int argc, char **argv )
{

    feenableexcept( FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW );

    // hyperparams
    int nr = 32;
    int nphi = 5;
    int nsigma = 32;
    uchar PatchSize = 64;

    int sChunk = 128;
    bool help = false;
    int nLastTick = -1;

    const char *OutH5Filename = NULL;

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
        if ( strcmp(argv[i], "-patchsize") == 0 )
        {
            PatchSize = atoi(argv[i+1]);
            i++;
            continue;
        }
        if ( strcmp(argv[i], "-nr") == 0 )
        {
            nr = atoi(argv[i+1]);
            i++;
            continue;
        }
        if ( strcmp(argv[i], "-nphi") == 0 )
        {
            nphi = atoi(argv[i+1]);
            i++;
            continue;
        }
        if ( strcmp(argv[i], "-nsigma") == 0 )
        {
            nsigma = atoi(argv[i+1]);
            i++;
            continue;
        }
        cout << "ERROR: Invalid " << argv[i] << " option." << endl;
        help = true;
      } else if ( argv[i][0] != '-' )
      {
         if ( ! OutH5Filename )
         {
            OutH5Filename = argv[i];
            continue;
         }
      }
    }

    if ( ! OutH5Filename )
      help = true;

    if ( help )
    {
        cout << endl;
        cout << "Usage: gen-poolregion dst_h5_file" << endl;
        cout << "       -patchsize <0-255, 64=default> " << endl;
        cout << "       -nr <0-255, 32=default> " << endl;
        cout << "       -nphi <0-255, 5=default> " << endl;
        cout << "       -nsigma <0-255, 32=default> " << endl;
        cout << endl;
        exit( 1 );
    }

    int PatchRad = floor((PatchSize - 1) / 2);

    double r0 = 0.0f;
    double r1 = (double) PatchRad;
    double phi0 = 0.0f;
    double phi1 = CV_PI / 4.0f;
    double sigma0 = 0.5f;
    double sigma1 = ceil(PatchRad / 2.0f);

    int nParams = nr*nphi*nsigma;

    // create hdf5 file
    Ptr<HDF5> h5io = open( OutH5Filename );

    cout << "Export RingParams: #" << nParams << endl;

    // create RingParams
    Mat RingParams( nParams, 3, CV_64F );

    // create storage
    int rngchunks[2] = { sChunk, 3 };
    h5io->dscreate( nParams, 3, CV_64F, "RingParams", 9, rngchunks );

    int k = 0;
    int chunk = 0;
    // sigma0 : ((sigma1 - sigma0)/(nsigma - 1)) : sigma1
    for ( double s = sigma0; s <= sigma1; s = s + ((sigma1 - sigma0) / (nsigma - 1)) )
    {
      // phi0 : ((phi1 - phi0)/(nphi - 1)) : phi1
      for ( double p = phi0; p <= phi1; p = p + ((phi1 - phi0) / (nphi - 1)) )
        // r0 : ((r1 - r0)/(nr - 1)) : r1
        for ( double r = r0; r <= r1; r = r + ((r1 - r0)/(nr - 1)) )
        {
          // % RingParams = [r(:), phi(:), sigma(:)]
          RingParams.at<double>( chunk, 0 ) = r;
          RingParams.at<double>( chunk, 1 ) = p;
          RingParams.at<double>( chunk, 2 ) = s;

          k++;
          chunk++;

          if ( ( chunk == sChunk ) ||
               ( k == nParams ) )
          {
            int offset[2] = { k - chunk, 0 };
            int counts[2] = {     chunk, 3 };
            h5io->dswrite( RingParams, "RingParams", offset, counts );
            nLastTick = TermProgress( (double)s / (double)sigma1, nLastTick );
          }
        }
    }
    nLastTick = TermProgress( 1.0f, nLastTick );

    // total projections
    int nPR = nParams * 8;

    // single slice of PRParams
    Mat PRParams = Mat::zeros( sChunk, 3, CV_32F );

    // create storage
    int prpchunks[2] = { sChunk, 3 };
    h5io->dscreate( nPR, 3, CV_32F, "PRParams", 9, prpchunks );

    // single slice of PRFilter
    int dsdims[3] = { sChunk, PatchSize, PatchSize };
    Mat PRFilter( 3, dsdims, CV_32F );

    // create storage
    int prfdim[3] = { nPR,    PatchSize, PatchSize };
    int chunks[3] = { sChunk, PatchSize, PatchSize };
    h5io->dscreate( 3, prfdim, CV_32F, "PRFilters", 9, chunks );

    // %% symmetric PRs
    cout << "Export Projection: #" << nPR << endl;

    Mat PR;
    k = 0, chunk = 0;
    // % loop over rings
    for ( int iCenter = 0; iCenter < nParams; iCenter++ )
    {
      double RadRing = RingParams.at<double>( iCenter, 0);
      double phi =     RingParams.at<double>( iCenter, 1);
      double sigma =   RingParams.at<double>( iCenter, 2);

      Mat Offsets( 8, 2, CV_64F );
      double xc = RadRing * cos( phi );
      double yc = RadRing * sin( phi );
      Offsets.at<double>(0,0) =  yc; Offsets.at<double>(0,1) =  xc;
      Offsets.at<double>(1,0) =  yc; Offsets.at<double>(1,1) = -xc;
      Offsets.at<double>(2,0) = -yc; Offsets.at<double>(2,1) =  xc;
      Offsets.at<double>(3,0) = -yc; Offsets.at<double>(3,1) = -xc;
      Offsets.at<double>(4,0) =  xc; Offsets.at<double>(4,1) =  yc;
      Offsets.at<double>(5,0) =  xc; Offsets.at<double>(5,1) = -yc;
      Offsets.at<double>(6,0) = -xc; Offsets.at<double>(6,1) = -yc;
      Offsets.at<double>(7,0) = -xc; Offsets.at<double>(7,1) =  yc;

      // % loop over PRs in a ring
      for ( int iOffset = 0; iOffset < Offsets.rows; iOffset++ )
      {
        // % PR = get_PR_filter(PatchSize, Offsets(iOffset, 2), Offsets(iOffset, 1), sigma);
        PR = get_PR_filter( PatchSize,
                            Offsets.at<double>(iOffset, 1),
                            Offsets.at<double>(iOffset, 0),
                            sigma );
        // % Gaussian centred at (x0, y0)
        // PRFilters(k, :) = PR(:)';
        memcpy( &PRFilter.at<float>( chunk, 0, 0), PR.clone().data, PR.total() * PR.elemSize() );

        // % save PR params
        // PRParams(k, :) = [Offsets(iOffset, 2), Offsets(iOffset, 1), sigma];
        PRParams.at<float>(chunk, 0) = Offsets.at<double>(iOffset, 1);
        PRParams.at<float>(chunk, 1) = Offsets.at<double>(iOffset, 0);
        PRParams.at<float>(chunk, 2) = (float) sigma;

        k++;
        chunk++;

        if ( ( chunk == sChunk ) ||
             ( k == nPR ) )
        {
           int prpoffset[2] = { k - chunk, 0 };
           int prpcounts[2] = {     chunk, 3 };
           h5io->dswrite( PRParams, "PRParams",  prpoffset, prpcounts );

           int prfoffset[3] = { k - chunk,         0,         0 };
           int prfcounts[3] = {     chunk, PatchSize, PatchSize };
           h5io->dswrite( PRFilter, "PRFilters", prfoffset, prfcounts );

           nLastTick = TermProgress( (double)iCenter / (double)nParams, nLastTick );
           chunk = 0;
        }
      }
    }
    nLastTick = TermProgress( 1.0f, nLastTick );

    // close
    h5io->close();

    PR.release();
    RingParams.release();
    PRParams.release();

    cout << "Done." << endl << endl ;

    return 0;
}
