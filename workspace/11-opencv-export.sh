#!/bin/bash

# mean: 4.7900  YO:4.48 ND:2.98 LY:6.91 dim #[57/48/49] AUC[0.989307/0.991378/0.985531] yosemite-0.025-0.075-pr#7-0.0020-0.200-pj
# mean: 3.7633  YO:3.18 ND:2.44 LY:5.67 dim #[75/64/67] AUC[0.991713/0.992974/0.988141] liberty-0.035-0.250-pr#7-0.0010-0.100-pj
# mean: 3.1500  YO:2.57 ND:2.03 LY:4.85 dim #[96/80/84] AUC[0.993079/0.994057/0.989595] liberty-0.035-0.250-pr#7-0.0005-0.100-pj
# mean: 2.1933  YO:1.55 ND:1.39 LY:3.64 dim #[141/120/126] AUC[0.995295/0.995743/0.991984] liberty-0.035-0.250-pr#7-0.0001-0.025-pj

../bin/export-opencv -flt filters.h5 \
                     -prg pr-learn/olderbest/yosemite-0.025-0.075-pr.h5 -id 7 \
                     -prj pj-learn/notredame-yosemite-0.025-0.075-pr#7-0.0020-0.200-pj.h5 \
                     opencv/vgg_generated_48.i

../bin/export-opencv -flt filters.h5 \
                     -prg pr-learn/liberty-0.035-0.250-pr.h5 -id 7 \
                     -prj pj-learn/notredame-liberty-0.035-0.250-pr#7-0.0010-0.100-pj.h5 \
                     opencv/vgg_generated_64.i

../bin/export-opencv -flt filters.h5 \
                     -prg pr-learn/liberty-0.035-0.250-pr.h5 -id 7 \
                     -prj pj-learn/notredame-liberty-0.035-0.250-pr#7-0.0005-0.100-pj.h5 \
                     opencv/vgg_generated_80.i

../bin/export-opencv -flt filters.h5 \
                     -prg pr-learn/liberty-0.035-0.250-pr.h5 -id 7 \
                     -prj pj-learn/notredame-liberty-0.035-0.250-pr#7-0.0001-0.025-pj.h5 \
                     opencv/vgg_generated_120.i
