#/*
# *  Copyright (c) 2012  Balint Cristian (cristian dot balint at gmail dot com)
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

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/trainer/include/)

ADD_EXECUTABLE(conv-impatches conv-impatches.cpp misc.cpp)
TARGET_LINK_LIBRARIES(conv-impatches ${ZLIB_LIBRARIES} ${OpenCV_LIBS})

ADD_EXECUTABLE(gen-poolregion gen-poolregion.cpp misc.cpp)
TARGET_LINK_LIBRARIES(gen-poolregion ${ZLIB_LIBRARIES} ${OpenCV_LIBS})

ADD_EXECUTABLE(comp-fulldists comp-fulldists.cpp misc.cpp vgg-desc.cpp)
TARGET_LINK_LIBRARIES(comp-fulldists ${ZLIB_LIBRARIES} ${OpenCV_LIBS})

CUDA_COMPILE(CUPRLEARN pr-learn.cu)
ADD_EXECUTABLE(pr-learn pr-learn.cpp misc.cpp ${CUPRLEARN})
TARGET_LINK_LIBRARIES(pr-learn ${ZLIB_LIBRARIES} ${OpenCV_LIBS} ${CUDA_CUDART_LIBRARY})

ADD_EXECUTABLE(pr-stats pr-stats.cpp misc.cpp)
TARGET_LINK_LIBRARIES(pr-stats ${ZLIB_LIBRARIES} ${OpenCV_LIBS})

ADD_EXECUTABLE(comp-uprjdists comp-uprjdists.cpp misc.cpp vgg-desc.cpp)
TARGET_LINK_LIBRARIES(comp-uprjdists ${ZLIB_LIBRARIES} ${OpenCV_LIBS})
