# nvcc = /usr/local/cuda/bin/nvcc
# cudalib = /usr/local/cuda/lib64/
# tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/include

# all: tf_nndistance_so.so
# .PHONY : all

# tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
# 	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# tf_nndistance_g.cu.o: tf_nndistance_g.cu
# 	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


nvcc = /usr/local/cuda/bin/nvcc
cudalib = /usr/local/cuda/lib64/
tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/
#TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

all: tf_nndistance_so.so
.PHONY : all

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow)/include -lcudart -L $(cudalib) -L $(tensorflow) -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I $(tensorflow)/include/external/nsync/public/

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow)/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
