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
 * C++ code by:  2016, Balint Cristian (cristian dot balint at gmail dot com)
 *
 */

/* pj-learn.cpp */
/* Learn Proj Rank RDA */
// C++ implementation of "learn_proj_rank_RDA.m"


#include <fenv.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/hdf/hdf5.hpp>

#include "openblas/cblas.h"
#include "openblas/lapacke.h"

#include "trainer.hpp"


using namespace std;
using namespace cv;
using namespace hdf;

int main( int argc, char **argv )
{

    // size of chunks
    int sChunk = 128;
    int nLastTick = -1;

    // %% yosemite
    // % <=80-D
    // %mu = 0.001;
    // %gamma = 0.5;
    // % <=64-D
    // % mu = 0.002;
    // % gamma = 0.5;

    // %% notredame
    // % <=80-D
    // % mu = 0.001;
    // % gamma = 1;
    // % <=64-D
    // % mu = 0.001;
    // % gamma = 0.25;

    // %% liberty
    // % <=80-D
    // % mu = 0.001;
    // % gamma = 1;
    // % <=64-D
    // % mu = 0.002;
    // % gamma = 1;

    // hyperparams
    int MaxDim = 80;
    float mu = 0.001f;
    float gamma = 0.500;
    unsigned int nIter = 50000;
    unsigned int LogStep = 100;
    unsigned int szBatch = 200;

    // train set
    const float nDiv = 0.80; // 80% of data

    bool help = false;
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
        if ( strcmp(argv[i], "-maxdim") == 0 )
        {
            MaxDim = atoi(argv[i+1]);
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
         ( ! OutputH5Filename ) )
      help = true;

    if ( help )
    {
        cout << endl;
        cout << "Usage: pr-learn  src_h5_dist_file dst_h5_output_file" << endl;
        cout << "       -mu <0.0-1.0, 0.025=default> " << endl;
        cout << "       -gamma <0.0-10.0, 0.10=default> " << endl;
        cout << "       -maxdim <10-256, 80=default> " << endl;
        cout << "       -iters <0-N, 5000000=default> " << endl;
        cout << endl;
        exit( 1 );
    }
    cout << "mu: " << mu
         << " gamma: " << gamma
         << " maxdim: " << MaxDim
         << " nIters: " << nIter
    << endl;


    // open hdf5 files
    Ptr<HDF5> h5io = open( DistsH5Filename );

    // get dimensions
    vector<int> DSize = h5io->dsgetsize( "Distance" );

    int nDists  = DSize[0];
    int FeatDim = DSize[1];

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

    double AUC_Best = 0;
    float Obj_Best = FLT_MAX;
    float FPR95_Best = FLT_MAX;
    Mat W_Best, W_Save, A_Save;

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
              << "." << info.minorVersion() <<  endl;
    cout << endl;

    // set active GPU
    cuda::setDevice(0);

    cuda::GpuMat cW;
    cuda::Stream stream;
    cuda::GpuMat DISTs;
    cuda::GpuMat POSVal,  NEGVal;
    cuda::GpuMat POSDist, NEGDist;
    cuda::GpuMat POSFeatDiffProj;
    cuda::GpuMat NEGFeatDiffProj;

    DISTs.upload( Dists, stream );
    POSVal.upload( PosVal, stream );
    NEGVal.upload( NegVal, stream );

    stream.waitForCompletion();

    // release mem
    PosVal.release();
    NegVal.release();

    Mat W     = Mat::zeros( Dists.cols, Dists.cols, CV_32F );
    Mat A     = Mat::zeros( Dists.cols, Dists.cols, CV_32F );
    Mat dLoss = Mat::zeros( Dists.cols, Dists.cols, CV_32F );
    Mat dfAvg = Mat::zeros( Dists.cols, Dists.cols, CV_32F );
    Mat MuDiag = mu * Mat::eye( Dists.cols, Dists.cols, CV_32F );

    // batch train matrices
    Mat DescDiffPosBatch( szBatch, Dists.cols, CV_32F );
    Mat DescDiffNegBatch( szBatch, Dists.cols, CV_32F );

    Mat PosDist, NegDist;
    Mat PosFeatDiffProj, NegFeatDiffProj;

    Mat D, mul;
    Mat IdxNegViol; //, DescDiffPosCur, DescDiffNegCur;

    unsigned int step = 0;
    int64 trainStartTime = getTickCount();
    for ( unsigned int t = 0; t <= nIter; t++ )
    {
      // %% gradient computation

      // % sample a batch (with replacement)
      unsigned int iPos, iNeg;
      for ( unsigned int k = 0; k < szBatch; k++ )
      {
        iPos = rng.uniform( 0, nPosTrn );
        iNeg = rng.uniform( 0, nNegTrn );
        #pragma omp parallel num_threads(2)
        {
          #pragma omp single
          {
            memcpy( DescDiffPosBatch.ptr<float>( k, 0 ), Dists.ptr<float>( IdxPos[iPos], 0 ) ,
                         Dists.cols * Dists.elemSize() );
          }

          #pragma omp single
          {
            memcpy( DescDiffNegBatch.ptr<float>( k, 0 ), Dists.ptr<float>( IdxNeg[iNeg], 0 ),
                         Dists.cols * Dists.elemSize() );
          }
        }
      }

      // % compute the distances
      #pragma omp parallel num_threads(2)
      {
        #pragma omp single
        {
          //gemm( W, DescDiffPosBatch, 1.0f, noArray(), 0.0f, PosFeatDiffProj, GEMM_2_T );
          Mat PosFeatDiffProj( W.rows, DescDiffPosBatch.rows, CV_32F );
          cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasTrans,
                       PosFeatDiffProj.rows, PosFeatDiffProj.cols, W.cols,
                       1.0f,
                       W.ptr<float>(0,0), W.cols,
                       DescDiffPosBatch.ptr<float>(0,0), DescDiffPosBatch.cols,
                       0.0f,
                       PosFeatDiffProj.ptr<float>(0,0), PosFeatDiffProj.cols );


          pow( PosFeatDiffProj, 2, PosFeatDiffProj );
          reduce( PosFeatDiffProj, PosDist, 0, CV_REDUCE_SUM );
        }

        #pragma omp single
        {
          //gemm( W, DescDiffNegBatch, 1.0f, noArray(), 0.0f, NegFeatDiffProj, GEMM_2_T );
          Mat NegFeatDiffProj( W.rows, DescDiffNegBatch.rows, CV_32F );
          cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasTrans,
                       NegFeatDiffProj.rows, NegFeatDiffProj.cols, W.cols,
                       1.0f,
                       W.ptr<float>(0,0), W.cols,
                       DescDiffNegBatch.ptr<float>(0,0), DescDiffNegBatch.cols,
                       0.0f,
                       NegFeatDiffProj.ptr<float>(0,0), NegFeatDiffProj.cols );

          pow( NegFeatDiffProj, 2, NegFeatDiffProj );
          reduce( NegFeatDiffProj, NegDist, 0, CV_REDUCE_SUM );
        }
      }

      dLoss.setTo( 0.0f );
      // % sum of outer products
      #pragma omp parallel for schedule(dynamic,1) private(IdxNegViol, D, mul)
      for ( unsigned int iPos = 0; iPos < szBatch; iPos++ )
      {
        // % IdxNegViol = (PosDist(iPos) + 1 > NegDist);
        IdxNegViol = ( PosDist.at<float>( iPos ) + 1.0f ) > NegDist;

        // % nViol = size(DescDiffNegCur, 2);
        int nViol = countNonZero(IdxNegViol);

        if ( nViol == 0 ) continue;

        // % DescDiffPosCur = DescDiffPosBatch(:, iPos);
        Mat DescDiffPosCur = DescDiffPosBatch.row( iPos );

        Mat DescDiffNegCur;
        // % DescDiffNegCur = DescDiffNegBatch(:, IdxNegViol)
        for ( int k = 0; k < NegDist.cols; k++ )
          if ( IdxNegViol.at<uchar>( 0, k ) != 0 )
            DescDiffNegCur.push_back( DescDiffNegBatch.row( k ) );

        // % dLoss = dLoss + (nViol * DescDiffPosCur) * DescDiffPosCur' - DescDiffNegCur * DescDiffNegCur';
        if ( nViol > 0 )
        {
          Mat mul( DescDiffNegCur.cols, DescDiffNegCur.cols, CV_32F);
          //gemm( DescDiffNegCur, DescDiffNegCur, 1    , noArray(),  0, mul, GEMM_1_T );
          cblas_sgemm( CblasRowMajor, CblasTrans, CblasNoTrans,
                       mul.rows, mul.cols, DescDiffNegCur.rows,
                       1.0f,
                       DescDiffNegCur.ptr<float>(0,0), DescDiffNegCur.cols,
                       DescDiffNegCur.ptr<float>(0,0), DescDiffNegCur.cols,
                       0.0f,
                       mul.ptr<float>(0,0), mul.cols );

          //Mat D( mul.rows, mul.cols, CV_32F);
          //gemm( DescDiffPosCur, DescDiffPosCur, nViol,       mul, -1, D  , GEMM_1_T );
          cblas_sgemm( CblasRowMajor, CblasTrans, CblasNoTrans,
                       mul.rows, mul.cols, DescDiffPosCur.rows,
                       nViol,
                       DescDiffPosCur.ptr<float>(0,0), DescDiffPosCur.cols,
                       DescDiffPosCur.ptr<float>(0,0), DescDiffPosCur.cols,
                       -1.0f,
                       mul.ptr<float>(0,0), mul.cols );

          #pragma omp critical
          {
            //add( dLoss, D, dLoss);
            add( dLoss, mul, dLoss);
          }
        }
      }

      // % subgradient average
      // % dfAvg = ((t - 1) / t) * dfAvg + (1 / (nPos * nNeg * t)) * dLoss;
      addWeighted( dfAvg, (double) t / (t + 1), dLoss, 1.0f / ( szBatch*szBatch * (t + 1) ), 0, dfAvg );

      // % update A
      // A = (-sqrt(t) / Params.gamma) * (dfAvg + MuDiag);
      add( dfAvg, MuDiag, A );
      double div = -sqrt( (double)t + 1.0f ) / (double)gamma;
      A = A.mul( div );

      // % ensure A is symmetrical (cancel-out numerical errors)
      // % A = 0.5 * (A + A');
      A = 0.5f * ( A + A.t() );

      Mat Eval = Mat( A.rows,      1, CV_32F );
      Mat Evec = Mat( A.rows, A.cols, CV_32F );

      // ssyrevx ~0.235 sec
      // ssyrevr ~0.070 sec
      int ifail[A.rows*2], m;
      LAPACKE_ssyevr( LAPACK_ROW_MAJOR, 'V', 'A', 'U',
                      A.rows, A.ptr<float>(0,0), A.cols,
                      0.0, 0.0, 0, 0, 0.0f,
                      &m,
                      Eval.ptr<float>(0),
                      Evec.ptr<float>(0,0), Evec.rows,
                      ifail );

      Mat V = Evec.t();

      // % pos eigen-values
      // % diagDPos = max(single(diag(D)), 0);
      Mat diagDPos = max( Eval, 0 );

      // % A = V * bsxfun(@times, diagDPos, V');
      // % W = bsxfun(@times, sqrt(diagDPos), V');
      Mat Bmul( V.rows, V.cols, CV_32F );
      Mat sqBmul( V.rows, V.cols, CV_32F );
      #pragma omp parallel for
      for ( int r = 0; r < V.rows; r++ )
      {
        const float   e =       diagDPos.at<float>( r );
        const float sqe = sqrt( diagDPos.at<float>( r ) );
        for ( int c = 0; c < V.cols; c++ )
        {
          const float v = V.at<float>( r, c );
            Bmul.at<float>( r, c ) =    e * v;
          sqBmul.at<float>( r, c ) =  sqe * v;
        }
      }

      // A = Evec * Bmul;
      cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   Evec.rows, Bmul.cols, Evec.rows,
                   1.0f,
                   Evec.ptr<float>(0,0), Evec.cols,
                   Bmul.ptr<float>(0,0), Bmul.cols,
                   0.0f,
                   A.ptr<float>(0,0), A.cols );

      // %W = W(diagDPos ~= 0, :);
      W.release();
      for ( int k = 0; k < diagDPos.rows; k++ )
        if ( diagDPos.at<float>( k ) != 0.0f )
          W.push_back( sqBmul.row( k ) );

      // %if isempty(W)
      // %  W = 0;
      // %end
      if ( W.rows == 0 )
        W = Mat::zeros( Dists.cols, Dists.cols, CV_32F );

      if ( step == LogStep )
      {
        int64 trainEndTime = getTickCount();
        int64 validStartTime = getTickCount();

        float Regul;
        float Loss = 0.0f;
        float LossVal = 0.0f;

        cW.upload( W, stream );

        // % compute the objective on the validation set subsample
        cuda::gemm( cW, POSVal, 1, cuda::GpuMat(), 0, POSFeatDiffProj, GEMM_2_T, stream );
        cuda::gemm( cW, NEGVal, 1, cuda::GpuMat(), 0, NEGFeatDiffProj, GEMM_2_T, stream );

        cuda::pow( POSFeatDiffProj, 2, POSFeatDiffProj, stream );
        cuda::pow( NEGFeatDiffProj, 2, NEGFeatDiffProj, stream );

        cuda::reduce( POSFeatDiffProj, POSDist, 0, CV_REDUCE_SUM );
        cuda::reduce( NEGFeatDiffProj, NEGDist, 0, CV_REDUCE_SUM );

        cuda::GpuMat dst;
        // % sum( max( (PosDist.at<float>(i) + 1.0f - NegDist), 0.0f ) )
        cuda::dlco::SubtractVectorsByRows( POSDist.reshape(1, POSDist.cols),
                                           NEGDist.reshape(1, POSDist.cols),
                                           dst, stream );

        // reduce final sums
        Loss = cuda::sum( dst )[0];

        stream.waitForCompletion();

        LossVal = Loss / (float)nPosVal / (float)nNegVal;

        // % compute the regulariser
        Regul = mu * trace( A )[0];

        int64 validEndTime = getTickCount();

        // % save w if it's the current minimiser of Obj
        if ( ( LossVal + Regul ) < Obj_Best )
        {
          Obj_Best = LossVal + Regul;

          W_Best = W.clone();

          printf( "Best: %i  Loss: %.6f Regul: %.6f Obj: %.6f (%.6f) Rank: %i (%i) Ttime: %.4f Vtime: %.4f\n",
                  t, LossVal, Regul, (LossVal + Regul), Obj_Best, W.rows, W_Best.rows,
                  ( trainEndTime - trainStartTime ) / frequency,
                  ( validEndTime - validStartTime ) / frequency );

          /*
           *  best model full statistics
           */

          int Dim;
          double AUC;
          float FPR95;

          ComputePJStats( DISTs, Labels, W, Dim, FPR95, AUC );

          /*
           * save best results
           * for best auc & fpr95
           */

          if ( ( AUC_Best <= AUC ) &&
               ( FPR95_Best >= FPR95 ) )
          {

            AUC_Best = AUC;
            FPR95_Best = FPR95;

            W_Save = W.clone();
            A_Save = A.clone();

            // log as saved
            printf( "Stat: Dim/MaxDim [%i/%i] AUC: %.6f (%.6f) FPR95: %.2f (%.2f) [saved]\n",
                    Dim, MaxDim, AUC, AUC_Best, FPR95*100, FPR95_Best*100 );

          } else {
            printf( "Stat: Dim/MaxDim [%i/%i] AUC: %.6f (%.6f) FPR95: %.2f (%.2f)\n",
                    Dim, MaxDim, AUC, AUC_Best, FPR95*100, FPR95_Best*100 );
          }
        } else {
            printf( "Step: %i  Loss: %.6f Regul: %.6f Obj: %.6f (%.6f) Rank: %i (%i) Ttime: %.4f Vtime: %.4f\n",
                    t, LossVal, Regul, (LossVal + Regul), Obj_Best, W.rows, W_Best.rows,
                    ( trainEndTime - trainStartTime ) / frequency,
                    ( validEndTime - validStartTime ) / frequency );
        }
        // flush i/o
        cout << flush;

        step = 0;
        trainStartTime = getTickCount();
      } // end if
      step++;
    } // end for cycle

    // open hdf5 files
    Ptr<HDF5> h5iw = open( OutputH5Filename );
    h5iw->dswrite( W_Save, "W" );
    h5iw->dswrite( A_Save, "A" );

    // close
    h5iw->close();

    return 0;
}
