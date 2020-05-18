# MandelBrot-Cuda-olc-PixelGameEngine

Inspired by javidx9 video about calculating the Mandelbrot set. 

https://www.youtube.com/watch?v=PBvLs88hvJ8

Code was taken from his Github and then edited 
https://github.com/OneLoneCoder/olcPixelGameEngine/blob/master/Videos/OneLoneCoder_PGE_Mandelbrot.cpp

Have fun exploring the Mandelbrot set smoothly with thousands of iterations.

# Changes 

Removed nearly all Methods since they did not gave a performance advantage and/or will not compile with nvcc on my Linux Machine

Added three Methods with CUDA acceleration.

# Compiling
Assuming you are in the right directory source compiles under Linux with the following command: 

nvcc -O3 -o MandelBrotCudaPixelGameEngine MandelBrotCudaPixelGameEngine.cu -lX11 -lGL -lpthread -lpng -lstdc++fs -std=c++14


I do not know how to compile for windows.

