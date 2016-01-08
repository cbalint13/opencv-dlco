/*
 * Copyright (c) 2014,2015  Balint Cristian (cristian dot balint at gmail dot com)
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

/* vgg-desc.cpp */
/* Implementation of "get_patch.m" */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;

Mat get_desc( Mat Patch, int nAngleBins, float InitSigma, bool bNorm )
{
    // % Patch = single(Patch);
    Patch.convertTo( Patch, CV_32F );
    // % smooth
    GaussianBlur( Patch, Patch, Size( 0, 0 ), InitSigma, InitSigma, BORDER_REPLICATE);

    Mat Ix, Iy;
    // % compute gradient
    float kparam[3] = { -1, 0, 1 };
    Mat Kernel( 1, 3, CV_32F, &kparam );
    filter2D( Patch, Ix, CV_32F, Kernel,     Point( -1, -1 ), 0, BORDER_REPLICATE );
    filter2D( Patch, Iy, CV_32F, Kernel.t(), Point( -1, -1 ), 0, BORDER_REPLICATE );

    Mat GMag, GAngle;

    // % gradient magnitude
    // % GMag = sqrt(Ix .^ 2 + Iy .^ 2);
    magnitude( Ix, Iy, GMag );

    // % gradient orientation: [0; 2 * pi]
    // % GAngle = atan2(Iy, Ix) + pi;
    //phase( Ix, Iy, GAngle, false ); # <- opencv buggy
    GAngle = Mat( GMag.rows, GMag.cols, CV_32F );
    for ( unsigned int i = 0; i < GAngle.total(); i++ )
      GAngle.at<float>(i) = atan2( Iy.at<float>(i), Ix.at<float>(i) ) + CV_PI;

    // % soft-assignment of gradients to the orientation histogram
    float AngleStep = 2.0f * CV_PI / (float) nAngleBins;
    Mat GAngleRatio = GAngle / AngleStep - 0.5f;
    // % Offset1 = mod(GAngleRatio, 1);
    Mat Offset1( GAngleRatio.rows, GAngleRatio.cols, CV_32F );
    for ( unsigned int i = 0; i < GAngleRatio.total(); i++ )
      Offset1.at<float>(i) = GAngleRatio.at<float>(i) - floor( GAngleRatio.at<float>(i) );

    Mat w1 = 1.0f - Offset1.t();
    Mat w2 = Offset1.t();

    Mat Bin1( GAngleRatio.rows, GAngleRatio.cols, CV_8U );
    Mat Bin2( GAngleRatio.rows, GAngleRatio.cols, CV_8U );

    // % Bin1 = ceil(GAngleRatio);
    // % Bin1(Bin1 == 0) = Params.nAngleBins;
    for ( unsigned int i = 0; i < GAngleRatio.total(); i++ )
    {
      if ( ceil( GAngleRatio.at<float>(i) - 1.0f) == -1.0f )
        Bin1.at<uchar>(i) = nAngleBins - 1;
      else
        Bin1.at<uchar>(i) = (uchar)
                ceil( GAngleRatio.at<float>(i) - 1.0f );
    }

    // % Bin2 = Bin1 + 1;
    // % Bin2(Bin2 > Params.nAngleBins) = 1;
    for ( unsigned int i = 0; i < GAngleRatio.total(); i++ )
    {
      if ( ( Bin1.at<uchar>(i) + 1 ) > nAngleBins - 1 )
        Bin2.at<uchar>(i) = 0;
      else
        Bin2.at<uchar>(i) = Bin1.at<uchar>(i) + 1;
    }

    // normalize
    if ( bNorm )
    {
      // % Quantile = 0.8;
      // % T = quantile(GMag(:), Quantile);
      float q = 0.8f;
      Mat GMagSorted;
      cv::sort( GMag.reshape( 0, 1 ),
                GMagSorted, CV_SORT_ASCENDING );

      int n = GMagSorted.cols;
      // scipy/stats/mstats_basic.py#L1718 mquantiles()
      // m = alphap + p*(1.-alphap-betap)
      // alphap = 0.5 betap = 0.5 => (m = 0.5)
      // aleph = (n*p + m)
      float aleph = ( n * q + 0.5f );
      int k = cvFloor( aleph );
      if ( k >= n - 1 ) k = n - 1;
      if ( k <= 1 ) k = 1;

      float gamma = aleph - k;
      if ( gamma >= 1.0f ) gamma = 1.0f;
      if ( gamma <= 0.0f ) gamma = 0.0f;
      // quantile out from distribution
      float T = ( 1.0f - gamma ) * GMagSorted.at<float>( k - 1 )
              + gamma * GMagSorted.at<float>( k );

      // avoid NaN
      if ( T != 0.0f ) GMag /= ( T / nAngleBins );
    }

    Mat Bin1T = Bin1.t();
    Mat Bin2T = Bin2.t();
    Mat GMagT = GMag.t();

    // % feature channels
    Mat PatchTrans = Mat::zeros( Patch.total(), nAngleBins, CV_32F );
    for (int i = 0; i < nAngleBins; i++)
    {
      for ( int p = 0; p < (int)Patch.total(); p++ )
      {
        if ( Bin1T.at<uchar>(p) == i )
          PatchTrans.at<float>(p,i) = w1.at<float>(p) * GMagT.at<float>(p);
        if ( Bin2T.at<uchar>(p) == i)
          PatchTrans.at<float>(p,i) = w2.at<float>(p) * GMagT.at<float>(p);
      }
    }

    return PatchTrans;
}
