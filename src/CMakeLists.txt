#/*
# *  Copyright (c) 2014  Balint Cristian (cristian dot balint at gmail dot com)
# *
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *
# * * Redistributions of source code must retain the above copyright notice, this
# *   list of conditions and the following disclaimer.
# *
# * * Redistributions in binary form must reproduce the above copyright notice,
# *   this list of conditions and the following disclaimer in the documentation
# *   and/or other materials provided with the distribution.
# *
# * * Neither the name of copyright holder nor the names of its
# *   contributors may be used to endorse or promote products derived from
# *   this software without specific prior written permission.
# *
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
# */
#
#/* CMakeLists.txt */
#/* Project builder */

ADD_LIBRARY(misc OBJECT misc.cpp)
ADD_LIBRARY(vgg OBJECT vgg-desc.cpp)

ADD_EXECUTABLE(conv-impatches $<TARGET_OBJECTS:misc> conv-impatches.cpp)
TARGET_LINK_LIBRARIES(conv-impatches ${OpenCV_LIBS})

ADD_EXECUTABLE(gen-poolregion $<TARGET_OBJECTS:misc> gen-poolregion.cpp )
TARGET_LINK_LIBRARIES(gen-poolregion ${OpenCV_LIBS})

ADD_EXECUTABLE(comp-fulldists $<TARGET_OBJECTS:misc> $<TARGET_OBJECTS:vgg> comp-fulldists.cpp)
TARGET_LINK_LIBRARIES(comp-fulldists ${OpenCV_LIBS})

CUDA_COMPILE(OPCUOPENCV kernelop-opencv.cu)

ADD_EXECUTABLE(pr-learn $<TARGET_OBJECTS:misc> ${OPCUOPENCV} pr-learn.cpp)
TARGET_LINK_LIBRARIES(pr-learn ${OpenCV_LIBS} ${CUDA_CUDART_LIBRARY})

ADD_EXECUTABLE(pr-stats $<TARGET_OBJECTS:misc> $<TARGET_OBJECTS:vgg> pr-stats.cpp)
TARGET_LINK_LIBRARIES(pr-stats ${OpenCV_LIBS})

ADD_EXECUTABLE(comp-uprjdists $<TARGET_OBJECTS:misc> $<TARGET_OBJECTS:vgg> comp-uprjdists.cpp)
TARGET_LINK_LIBRARIES(comp-uprjdists ${OpenCV_LIBS})

ADD_EXECUTABLE(pj-learn $<TARGET_OBJECTS:misc> pj-learn.cpp ${OPCUOPENCV})
TARGET_LINK_LIBRARIES(pj-learn ${OpenCV_LIBS} ${CUDA_CUDART_LIBRARY} ${OpenBLAS_LIB})

ADD_EXECUTABLE(export-opencv $<TARGET_OBJECTS:misc> export-opencv.cpp)
TARGET_LINK_LIBRARIES(export-opencv ${OpenCV_LIBS})

#ADD_EXECUTABLE(pj-stats $<TARGET_OBJECTS:misc> $<TARGET_OBJECTS:vgg> pj-stats.cpp)
#TARGET_LINK_LIBRARIES(pj-stats ${OpenCV_LIBS})
