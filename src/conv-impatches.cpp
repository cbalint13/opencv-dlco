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

/* conv-impatches.cpp */
/* Convert patch datasets into HDF5 */
//   "Learning Local Image Descriptors Data", Matthew Brown
//   URL: http://www.cs.ubc.ca/~mbrown/patchdata/patchdata.html

#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/hdf/hdf5.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "trainer.hpp"


using namespace std;
using namespace cv;
using namespace hdf;

int main( int argc, char **argv )
{
    int idx = 0;
    int sChunk = 256;

    bool help = false;
    int nLastTick = -1;

    const char *OutH5Filename = NULL;
    const char *PatchDirPath = NULL;

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
        cout << "ERROR: Invalid " << argv[i] << " option." << endl;
        help = true;
      } else if ( argv[i][0] != '-' )
      {
         if ( ! PatchDirPath )
         {
            PatchDirPath = argv[i];
            continue;
         }
         if ( ! OutH5Filename )
         {
            OutH5Filename = argv[i];
            continue;
         }
      }
    }

    if ( ( ! OutH5Filename) || (! PatchDirPath) )
      help = true;

    if ( help )
    {
        cout << endl;
        cout << "Usage: conv-imgpatches patch_dir_path dst_h5_file" << endl;
        cout << endl;
        exit( 1 );
    }

    ifstream mFile;
    char mFileName[1024] = "";
    sprintf( mFileName, "%s/m50_500000_500000_0.txt", PatchDirPath );
    cout << "Open index file: " << mFileName << endl;

    string line;
    int NumPatches = 0;
    mFile.open( mFileName );

    if( mFile.fail() )
    {
      cout << "ERROR: File m50_500000_500000_0.txt not found." << endl;
      cout << endl;
      exit( 1 );
    }

    // create hdf5 file
    Ptr<HDF5> h5io = open( OutH5Filename );

    // count lines
    int NumLines = 0;
    while ( getline( mFile, line ) )
      NumLines++;

    cout << "Export Indices: #" << NumLines << endl;

    // create indices
    Mat Indices( sChunk, 4, CV_32S );

    // create storage
    int indchunks[2] = { sChunk, 4 };
    h5io->dscreate( NumLines, 4, CV_32S, "Indices", HDF5::H5_NONE, indchunks );

    mFile.clear();
    mFile.seekg( 0, ios::beg );
    // iterate through index file using chunks
    for ( int i = 0; i < NumLines; i += idx )
    {
      for ( idx = 0; idx < sChunk; idx++ )
      {
        if ( ! getline( mFile, line ) )
          break;

        int IdPatch1, IdPatch2;
        int IdPoint1, IdPoint2;

        // parse line
        sscanf( line.c_str(), "%i %i %*s %i %i %*s",
                &IdPatch1, &IdPoint1, &IdPatch2, &IdPoint2 );
        // populate array
        Indices.at<int>(idx,0) = IdPatch1; Indices.at<int>(idx,1) = IdPoint1;
        Indices.at<int>(idx,2) = IdPatch2; Indices.at<int>(idx,3) = IdPoint2;

        // get max id (+1, 0 offset)
        if ( IdPatch1 > NumPatches )
          NumPatches = IdPatch1 + 1;
        if ( IdPatch2 > NumPatches )
          NumPatches = IdPatch2 + 1;
      }
      nLastTick = TermProgress( (double)i / (double)NumLines, nLastTick );

      int offset[2] = {   i, 0 };
      int counts[2] = { idx, 4 };
      h5io->dswrite( Indices, "Indices", offset, counts );
    }
    mFile.close();
    nLastTick = TermProgress( 1.0f, nLastTick );

    Indices.release();

    // iterate through image patches
    cout << "Export Patches: #" << NumPatches << endl;

    Rect Box;
    Mat Image;
    int ImagePatches = 16 * 16;

    int imgdims[] = { ImagePatches, 64, 64 };
    Mat ImgPatch( 3, imgdims, CV_8U );

    // create dataset into HDF5
    int pchdims[3] = { NumPatches, 64, 64 };
    int pchunks[3] = {     sChunk, 64, 64 };
    h5io->dscreate( 3, pchdims, CV_8U, "Patches", 9, pchunks );

    idx = 0;
    for ( int i = 0; i <= (NumPatches / ImagePatches); i++ )
    {
      char PatchFile[1024] = "";
      sprintf( PatchFile, "%s/patches%04i.bmp", PatchDirPath, i );

      Image = imread( PatchFile , CV_LOAD_IMAGE_GRAYSCALE );

      int chunk = 0;
      for (int r = 0; r < 16; r++)
      {
        for (int c = 0; c < 16; c++)
        {
          if ( idx >= NumPatches )
            break;

          // 01 02 03 04 .. 16
          // 17 18 19 20 .. 32
          Box = Rect( Point( c*64, r*64 ), Size( 64, 64 ) );
          memcpy( &ImgPatch.at<uchar>( chunk, 0, 0 ),
                  Image( Box ).clone().data, Image( Box ).total() );
          idx++;
          chunk++;
        }
      }

      // save one block 256 x 64x64 chunk
      int offset[3] = { idx-chunk,  0,  0 };
      int counts[3] = {     chunk, 64, 64 };
      h5io->dswrite( ImgPatch, "Patches", offset, counts );

      nLastTick = TermProgress( (double)i / (double)(NumPatches/256), nLastTick );
    }
    nLastTick = TermProgress( 1.0f, nLastTick );

    // close
    h5io->close();

    Image.release();
    ImgPatch.release();

    cout << "Done." << endl << endl ;

    return 0;
}
