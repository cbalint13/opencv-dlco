# opencv-dlco
---------------------------------------------------------------------------------------

OpenCV DLCO (Descriptor Learning Using Convex Optimisation)


 This is OpenCV implementation of DLCO [1], [2] and [3]. It has booth trainer
and descriptor part, reimplemented from original matlab code. Trainer part
implemented here uses GPU as mandatory requirement. Descriptor part called
VGG was submitted to OpenCV: https://github.com/Itseez/opencv_contrib/pull/486

* The minimum hardware required for training process is to have 16Gb memory,
and CUDA capable GPU card with minimum 4Gb of memory. Matrices involved during
training reach up to 13Gb, and some validation matrices uses up to 2.5G memory
on GPU card for optimised inplace computation. Progress bars indicates realtime
computations and estimated time at any point or part of the training process.
Core storage containers for the very large matrices uses OpenCV's HDF5 backend,
thus these numerical results can be consulted outside the DLCO framework.

* Math part in many places reorders computation from original formulas for being
numericaly stable due to large floating point computations & error accumulations.
Very special attention was given for these aspects during coding DLCO.

* Follow scripts and documentation from the `workspace` folder for getting
through whole training procedure.


Bug reports, suggestions & improvements are welcome !
<cristian dot balint at gmail dot com>

---------------------------------------------------------------------------------------

* Original porject: http://www.robots.ox.ac.uk/~vgg/software/learn_desc/

[1] K. Simonyan, A. Vedaldi, A. Zisserman
Learning Local Feature Descriptors Using Convex Optimisation
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2014

[2] K. Simonyan, A. Vedaldi, A. Zisserman
Descriptor Learning Using Convex Optimisation
European Conference on Computer Vision, 2012

[3] M. Brown, G. Hua, S. Winder
Discriminative Learning of Local Image Descriptors
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2011

---------------------------------------------------------------------------------------

07-Jan-2016

 - Pool-Region filters are learnt over 300 combinations of mu & gamma regularizers.
 - Best selected filters average at 10.64% FPR95 value, lower than authors exposed.

TODO:

 - Projection matrix computation & selection tools for best projection matrix.
 - Export tool that convert and compact learnt matrices into OpenCV's VGG `*.i` C code.
 - Hamming distance binarisation part of the DLCO, add it to OpenCV VGG descriptor too.
