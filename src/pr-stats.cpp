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

/* pr-stats.cpp */
/* Select best learned PR filter */
// C++ implementation of "script_select_PR.m"

#include <map>
#include <fenv.h>
#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/hdf/hdf5.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

    // params
    int MaxDim = 640;
    const int nChannels = 8;
    const bool bModelSelect = true;

    bool help = false;
    vector<string> dst_files;
    vector<string> prj_files;

    const char *FltH5Filename = NULL;

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
        if ( strcmp(argv[i], "-dst") == 0 )
        {
            dst_files.push_back( argv[i+1] );
            i++;
            continue;
        }
        if ( strcmp(argv[i], "-prj") == 0 )
        {
            prj_files.push_back( argv[i+1] );
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
      }
    }

    if ( help )
    {
        cout << endl;
        cout << "Usage: pr-stats src_h5_filter_file" << endl;
        cout << "        -dst src_h5_dst_file1 -dst src_h5_dst_file2 -dst ..." << endl;
        cout << "        -prj src_h5_prj_file1 -prj src_h5_prj_file2 -prj ..." << endl;
        cout << endl;
        exit( 1 );
    }
    cout << "MaxDim: " << MaxDim
              << " nChannels: " << nChannels
    << endl;

    Mat PRParams, RingParams;
    Ptr<HDF5> h5fl = open( FltH5Filename );

    cout << "Load PRParams." << endl;
    h5fl->dsread( PRParams, "PRParams" );

    cout << "Load RingParams." << endl;
    h5fl->dsread( RingParams, "RingParams" );

    h5fl->close();

    int iPR = 0;
    Mat nPRInRing(RingParams.rows, 1, CV_32F, cvScalar(0.0f));
    for (int iRing = 0; iRing < RingParams.rows; iRing++)
    {
      Mat RingPRs = PRParams.rowRange(iPR, iPR + 8).clone();

      // % nPRInRing(iRing) = size(unique(RingPRs, 'rows'), 1);

      int dup_rows = 0;
      for (int i = 0; i < RingPRs.rows; i++) {
        int isInside = false;
        for (int j = 0; j < RingPRs.rows; j++) {
          if (i == j) continue;
          int count = 0;
          for (int k = 0; k < RingPRs.cols; k++)
            if( RingPRs.at<float>(i, k) ==
                    RingPRs.at<float>(j, k) )
                count ++;
            if (count == RingPRs.cols)
                isInside = true;
        }
        if (isInside == true)
          dup_rows++;
      }

      nPRInRing.at<float>(iRing, 0) = RingPRs.rows - (dup_rows / 2);;

      iPR = iPR + 8;
    }


    // {FPR95, DIM, nPR, NZIDX, IDModelSet, IDTestSet}
    typedef array<float, 7> ModelParam;
    multimap<double, ModelParam> ModelsCollection;

    vector<string>::iterator TestSetName;

    // iterate combinations of sets
    int dst_idx = 0;
    for ( TestSetName = dst_files.begin() ; TestSetName != dst_files.end(); ++TestSetName)
    {
      printf("Set: [%s] {full}\n", TestSetName->c_str());

      /*
       * Test Set
       */

      Ptr<HDF5> h5ts = open( TestSetName[0] );
      vector<int> DSize = h5ts->dsgetsize( "Distance" );
      int nDists  = DSize[0]; int FeatDim = DSize[1];

      // allocate storage
      Mat Labels( nDists, 1, CV_8U );
      Mat Dists( nDists, FeatDim, CV_32F );

      // load TestSet
      Mat TestLabel, TestDist;
      printf("    Load TestLabel: ");
      for ( int i = 0; i < nDists; i = i + sChunk )
      {
        int count = sChunk;
        // for the last incomplete chunk
        if ( i + sChunk > nDists )
          count = nDists - i;

        int loffset[2] = {     i, 0 };
        int lcounts[2] = { count, 1 };
        h5ts->dsread( TestLabel, "Label", loffset, lcounts );

        int doffset[2] = {     i,       0 };
        int dcounts[2] = { count, FeatDim };
        h5ts->dsread( TestDist, "Distance", doffset, dcounts );

        memcpy( &Labels.at<uchar>( i ), TestLabel.data, TestLabel.total() );
        memcpy( &Dists.at<float>( i, 0 ), TestDist.data, TestDist.total() * TestDist.elemSize() );

        nLastTick = TermProgress( (double)i / (double)nDists, nLastTick );
      }
      nLastTick = TermProgress( 1.0f, nLastTick );

      // close
      h5ts->close();

      // best model;
      Mat w_best;
      int widx_best = -1;
      string model_best;

      int prj_idx = -1;
      double AUC_best = 0.0f;
      vector<string>::iterator PrjModelName;
      printf("    Validate Model: ");
      for ( PrjModelName = prj_files.begin() ; PrjModelName != prj_files.end(); ++PrjModelName )
      {
        prj_idx++;

        Mat W;

        Ptr<HDF5> h5io = open( PrjModelName[0] );
        h5io->dsread( W, "w" );
        h5io->close();

        CV_Assert( W.cols * 8 == PRParams.rows );

        // iterate all items in W
        for ( int widx = 0; widx < W.rows; widx++ )
        {
          // choose one w
          Mat w = W.row( widx );

          double AUC;
          float FPR95;
          int nPR, Dim, nzDim;

          // full statistics
          if ( ! bModelSelect ) MaxDim = -1;

          ComputeStats( nChannels, PRParams, Dists, Labels, w, nPR, Dim, nzDim, FPR95, AUC, MaxDim );

          nLastTick = TermProgress( (double)prj_idx / (double)prj_files.size(), nLastTick );

          // invalid result
          if ( bModelSelect )
          if ( Dim > MaxDim )
            continue;

          // [AUC , {FPR95, DIM, nPR, NZIDX, IDModelSet, IDTestSet}]
          ModelsCollection.insert(
               pair<double, ModelParam>(
                   AUC, ModelParam{
                      FPR95*100, (float)Dim,
                     (float)nPR, (float)nzDim,
                     (float)prj_idx, (float)dst_idx,
                     (float)widx
                   }
               )
          );

          // keep best w
          if ( AUC > AUC_best )
          {
            w_best = w;
            AUC_best = AUC;
            widx_best = widx;
            model_best = PrjModelName[0];
          }

        }
      }

      dst_idx++;
      nLastTick = TermProgress( 1.0f, nLastTick );

      multimap<double, ModelParam>::iterator it;
      for ( it = ModelsCollection.begin(); it != ModelsCollection.end() ; ++it )
      {
        if ( TestSetName[0].compare(dst_files[(int)it->second[5]]) == 0 )
        printf( "  ModelStat: AUC #%.07g  FPR95: %.2f Dim/MaxDim [%i/%i] nPR: %i (#%i) [%s](#%i)->[%s]\n",
                it->first, it->second[0], (int)it->second[1], MaxDim, (int)it->second[2],
                (int)it->second[3], prj_files[(int)it->second[4]].c_str(), (int)it->second[6],
                dst_files[(int)it->second[5]].c_str() );
      }
      printf( "\n  BestModel: [%s] #%i\n", model_best.c_str(), widx_best );

    }
    return 0;
}
