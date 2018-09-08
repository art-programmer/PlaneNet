## Training data generation from ScanNet models
Please first compile the C++ code under folder *Renderer/* which is used for rendering 3D planar segmentation results on 2D views. OpenCV, OpenGL and gflags are required.
```bash
cd Renderer
cmake .
make
cc ..
```

After compiling the C++ code successfully, you can run *parse.py* which will generate a folder named *annotation/* under each scan folder. You may need to change the path in the script based on your local file structure. ScanNet v2 also has different file paths with v1.
