**OpenCV**

* Folder here contains exported C headers files `vgg_generated_XX.i` used by
OpenCV VGG upstream implementation. File sizes are optimized and arrays are
in the most compact possible code. All float values are expressed in hexas and
arrays are enumerated in indexed-sparse layout thus only non zero values are
present.

* Exports are done using [11-opencv-export.sh] script.

* Folder `original` contains exports from author's original matrices.
