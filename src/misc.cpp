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

/* misc.cpp */
/* Miscelaneous routines */

#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;

int TermProgress( double dfComplete , int nLastTick = -1 )
{
    int nThisTick = (int) (dfComplete * 40.0);

    if (nThisTick < 0)
        nThisTick = 0;
    if (nThisTick > 40)
        nThisTick = 40;

    // Have we started a new progress run?
    if ( ( nThisTick < nLastTick )
           && ( nLastTick >= 39) )
        nLastTick = -1;

    if ( nThisTick <= nLastTick )
        return nLastTick;

    while ( nThisTick > nLastTick )
    {
        nLastTick = nLastTick + 1;
        if ( ( nLastTick % 4 ) == 0 )
            printf("%d" , ( ( nLastTick / 4 ) * 10 ) );
        else
            printf( "." );
    }
    if (nThisTick == 40)
        printf( " - done.\n" );
    else
        fflush( stdout );

    return nLastTick;
}

Mat SelectPRFilters( const Mat PRFilters, const Mat w )
{
    // % w = repmat(w', 8, 1);
    // % w = w(:);
    // % NZIdx = (w > 0) & any(PRFilters, 2);
    // % w = w(NZIdx);
    // % PRFilters = PRFilters(NZIdx, :);

    CV_Assert( w.cols * 8 == PRFilters.rows );

    Mat nPRFilters;
    for ( int i = 0; i < w.cols; i++ )
    {
      for ( int j = 0; j < 8; j++ )
      {
        int row = i*8 + j;
        if ( w.at<float>( 0, i ) > 0.0f )
        if ( countNonZero( PRFilters.row( row ) ) != 0 )
        {
          nPRFilters.push_back( PRFilters.row( row ) );
        }
      }
    }

    Mat mPRFilters;
    // % [PRFilters, ~, UniqueIdx2] = unique(PRFilters, 'rows');
    for ( int i = 0; i < nPRFilters.rows; i++ )
    {
      int isInside = false;
      for ( int j = 0; j < mPRFilters.rows; j++ )
      {
        int count = 0;
        for ( int k = 0; k < mPRFilters.cols; k++ )
          if ( nPRFilters.at<float>( i, k ) ==
               mPRFilters.at<float>( j, k ) )
            count++;
        if ( count == nPRFilters.cols )
        {
          isInside = true;
          break;
        }
      }
      if ( isInside == false )
        mPRFilters.push_back( nPRFilters.row( i ) );
    }

    // unique() does sortrows() too
    Mat sPRFilters = Mat( mPRFilters.rows, mPRFilters.cols, CV_32F );

    mPRFilters.row( 0 ).copyTo( sPRFilters.row( 0 ) );
    for ( int i = 1; i < mPRFilters.rows; i++ )
    {
      int idx = i;
      while ( idx > 0 )
      {
        bool cmp = false;
        for ( int j = 0; j < mPRFilters.cols; j++ )
        {
          if ( mPRFilters.at<float>( i, j ) ==
               sPRFilters.at<float>( idx-1, j ) )
            continue;
          else if ( mPRFilters.at<float>( i, j ) <
                    sPRFilters.at<float>( idx-1, j ) )
          {
            cmp = true;
            sPRFilters.row( idx-1 ).copyTo( sPRFilters.row( idx ) );
            break;
          }
          else if ( mPRFilters.at<float>( i, j ) >
                    sPRFilters.at<float>( idx-1, j ) )
          {
            cmp = false;
            mPRFilters.row( i ).copyTo( sPRFilters.row( idx ) );
            break;
          }
        }
        if ( cmp == true )
        {
          idx -= 1;
          if ( idx == 0 )
            mPRFilters.row( i ).copyTo( sPRFilters.row( idx ) );
          continue;
        }
        else
        {
          break;
        }
      }
    }

    return sPRFilters;
}

void ComputeStats( const int nChannels, const Mat& PRParams,
                   const Mat& Dists, const Mat& Labels, const Mat& w,
                   int& nPR, int &Dim, int& nzDim, float& FPR95, double& AUC, int MaxDim = -1 )
{

    // % w2 = repmat(reshape(w, 1, []), 8, 1);
    // % NZIdx = (w2(:) > 0) & any(PRParams, 2);
    // % nPR = size(unique(PRParams(NZIdx, :), 'rows'), 1);
    // % PRParamsNZIdx = PRParams(NZIdx, :)

    Mat PRParamsNZIdx;

    for ( int i = 0; i < w.cols; i++ )
    {
      for ( int j = 0; j < 8; j++ )
      {
        int row = i*8 + j;
        if ( w.at<float>( 0, i ) > 0 )
        if ( countNonZero( PRParams.row( row ) ) != 0 )
          PRParamsNZIdx.push_back( PRParams.row( row ) );
      }
    }

    // % nPR = size(unique(PRParamsNZIdx, 'rows'), 1);
    int dup_rows = 0;
    for ( int i = 0; i < PRParamsNZIdx.rows; i++ )
    {
      int isInside = false;
      for ( int j = 0; j < PRParamsNZIdx.rows; j++ )
      {
        if ( i == j ) continue;
        int count = 0;
        for ( int k = 0; k < PRParamsNZIdx.cols; k++ )
          if( PRParamsNZIdx.at<float>(i, k) ==
              PRParamsNZIdx.at<float>(j, k) )
            count++;
        if ( count == PRParamsNZIdx.cols )
          isInside = true;
      }
      if ( isInside == true )
        dup_rows++;
    }

    nzDim = PRParamsNZIdx.rows;
    nPR = nzDim - (dup_rows / 2);
    Dim = nPR * nChannels;

    // abort on MaxDim
    if ( ( MaxDim != -1 )
      && ( Dim > MaxDim ) )
      return;

    // % apply model
    Mat PatchRank;
    Mat PatchDist = w * Dists.t();
    sortIdx( PatchDist, PatchRank, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING );
    PatchDist.release();

    CV_Assert( PatchRank.type() == CV_32S );

    // % compute ROC curve
    float tplast = 0.0f, fplast = 0.0f;
    Mat TPR(1, PatchRank.cols, CV_32F);
    Mat FPR(1, PatchRank.cols, CV_32F);
    for (int i = 0; i < PatchRank.cols; i++)
    {
      int idx = PatchRank.at<int32_t>(0, i);
      if ( Labels.at<uchar>(idx, 0) == 1 ) tplast++;
      if ( Labels.at<uchar>(idx, 0) == 0 ) fplast++;
      TPR.at<float>(0, i) = tplast; FPR.at<float>(0, i) = fplast;
    }

    TPR /= tplast;
    FPR /= fplast;

    // % FPR @ 95% Recall
    FPR95 = -1.0f;
    Mat TFPR(PatchRank.cols+1, 2, CV_32F);
    for (int i = 0; i < PatchRank.cols; i++)
    {
      TFPR.at<float>(i, 0) = FPR.at<float>(0, i);
      TFPR.at<float>(i, 1) = TPR.at<float>(0, i);
      if ( ( FPR95 == -1.0f ) &&
           ( TPR.at<float>(0, i) >= 0.95f ) )
         FPR95 = FPR.at<float>(0, i);
    }
    TFPR.at<float>(PatchRank.cols, 0) = 1.0f;
    TFPR.at<float>(PatchRank.cols, 1) = 0.0f;

    // % area under ROC curve
    AUC = contourArea( TFPR );
}
