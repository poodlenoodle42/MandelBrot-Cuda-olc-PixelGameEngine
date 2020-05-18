#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <complex>
#include <cstdlib>
#include <thrust/complex.h>
#include <stdio.h>
__global__ void CreateFractalCUDAKernelNaive(const double x_scale,
	const double y_scale,const double frac_tl_x,const double frac_tl_y,
	const int iterations, int * dev_fractal, const int threadDim ){
	int x = blockIdx.x ; int y = threadIdx.x;
	int n = 0;
	thrust::complex<double> c = thrust::complex<double>(x * x_scale + frac_tl_x,y*y_scale + frac_tl_y);
	thrust::complex<double> z = thrust::complex<double>(0,0);
	
	while(thrust::abs(z) < 2.0 && n < iterations){
		z = (z*z) + c;
		n++;
	}
	dev_fractal[y * threadDim + x] = n;
}

inline __device__ double2 ComplexMul (const	double2 a ,const double2 b){
	double2 result = {a.x * b.x - a.y * b.y , a.x * b.y + b.x * a.y};
	return result;

}

inline __device__ double2 ComplexAdd (const double2 a , const double2 b){
	double2 result = {a.x+b.x,a.y+b.y};
	return result;
}

__global__ void CreateFractalCUDAKernelOptimised(const double x_scale,
	const double y_scale,const double frac_tl_x,const double frac_tl_y,
	const int iterations, int * dev_fractal, const int threadDim ,const int pixelPerThread = 4){
	int x = blockIdx.x*pixelPerThread ; int y = threadIdx.x;
	for(int i = 0;i<pixelPerThread;i++){
		int n = 0;
		double2 c = {x * x_scale + frac_tl_x,y*y_scale + frac_tl_y};
		double2 z = {0.0,0.0};
		while(z.x*z.x + z.y * z.y < 4.0 && n < iterations){
			z =ComplexAdd(ComplexMul(z,z),c);
			n++;
		}
		dev_fractal[y*threadDim + x] = n;
		x++;
	}
}

__global__ void CreateFractalCUDAKernelStepZoom(const double x_scale,
	const double y_scale,const double frac_tl_x,const double frac_tl_y,
	const int iterations, int * dev_fractal, const int threadDim ,const int pixelPerThread,
	const int resScale){
	int x = blockIdx.x*pixelPerThread*resScale ; int y = threadIdx.x*resScale;
	for(int i = 0;i<pixelPerThread;i++){
		int n = 0;
		double2 c = {x * x_scale + frac_tl_x,y*y_scale + frac_tl_y};
		double2 z = {0.0,0.0};
		while(z.x*z.x + z.y * z.y < 4.0 && n < iterations){
			z =ComplexAdd(ComplexMul(z,z),c);
			n++;
		}
		for(int s = 0; s<resScale;s++){
			for(int s2 = 0; s2<resScale;s2++){
				dev_fractal[(y+s2)*threadDim + x+s] = n;
			}
		}
		x+= resScale;
	}
}

class Mandelbrot : public olc::PixelGameEngine
{
public:
	Mandelbrot()
	{
		// Name you application
		sAppName = "Mandelbrot";
	}
    int* pFractal = nullptr;
    int * dev_fractal = nullptr;
	int nMode = 0;
	int nIterations = 128;
public:
	bool OnUserCreate() override
	{
		pFractal = new int[size_t(ScreenWidth()) * size_t(ScreenHeight())];
		cudaMalloc((void**)&dev_fractal, int(ScreenWidth()) * int(ScreenHeight()) * sizeof(int));
		return true;
	}

	bool OnUserDestroy() override{
		cudaFree(dev_fractal);
		delete[] pFractal;
		cudaDeviceReset();
		return true;
	}

	void CreateFractalBasic(const olc::vi2d& pix_tl, const olc::vi2d& pix_br, const olc::vd2d& frac_tl, const olc::vd2d& frac_br, const int iterations){
		double x_scale = (frac_br.x - frac_tl.x) / (double(pix_br.x) - double(pix_tl.x));
		double y_scale = (frac_br.y - frac_tl.y) / (double(pix_br.y) - double(pix_tl.y));
		
		for (int y = pix_tl.y; y < pix_br.y; y++)
		{
			for (int x = pix_tl.x; x < pix_br.x; x++)
			{
				std::complex<double> c(x * x_scale + frac_tl.x, y * y_scale + frac_tl.y);
				std::complex<double> z(0, 0);

				int n = 0;
				while (abs(z) < 2.0 && n < iterations)
				{
					z = (z * z) + c;
					n++;
				}

				pFractal[y * ScreenWidth() + x] = n;
			}
		}
	}

	void CreateFractalCUDA(const olc::vi2d& pix_tl, const olc::vi2d& pix_br, const olc::vd2d& frac_tl, const olc::vd2d& frac_br, const int iterations){
		double x_scale = (frac_br.x - frac_tl.x) / (double(pix_br.x) - double(pix_tl.x));
		double y_scale = (frac_br.y - frac_tl.y) / (double(pix_br.y) - double(pix_tl.y));
		CreateFractalCUDAKernelNaive <<<ScreenWidth(), ScreenHeight()>>>
		(x_scale,y_scale,frac_tl.x,frac_tl.y,iterations,dev_fractal,ScreenWidth());
		//cudaDeviceSynchronize();
		cudaMemcpy(pFractal,dev_fractal, ScreenWidth()*ScreenHeight()*sizeof(int), cudaMemcpyDeviceToHost);

		
	}

	void CreateFractalCUDAOptimised(const olc::vi2d& pix_tl, const olc::vi2d& pix_br, const olc::vd2d& frac_tl, const olc::vd2d& frac_br, const int iterations){
		double x_scale = (frac_br.x - frac_tl.x) / (double(pix_br.x) - double(pix_tl.x));
		double y_scale = (frac_br.y - frac_tl.y) / (double(pix_br.y) - double(pix_tl.y));
		//printf("CreateFractalCUDA called");
		CreateFractalCUDAKernelOptimised <<<ScreenWidth()/10, ScreenHeight()>>>
		(x_scale,y_scale,frac_tl.x,frac_tl.y,iterations,dev_fractal,ScreenWidth(),10);
		//cudaDeviceSynchronize();
		cudaMemcpy(pFractal,dev_fractal, ScreenHeight()*ScreenWidth()*sizeof(int), cudaMemcpyDeviceToHost);
	}

	void CreateFractalCUDAStepZoom(const olc::vi2d& pix_tl, const olc::vi2d& pix_br, const olc::vd2d& frac_tl, const olc::vd2d& frac_br, const int iterations,const int resScale){
		double x_scale = (frac_br.x - frac_tl.x) / (double(pix_br.x) - double(pix_tl.x));
		double y_scale = (frac_br.y - frac_tl.y) / (double(pix_br.y) - double(pix_tl.y));
		CreateFractalCUDAKernelStepZoom <<<ScreenWidth()/5/resScale, ScreenHeight()/resScale>>>
		(x_scale,y_scale,frac_tl.x,frac_tl.y,iterations,dev_fractal,ScreenWidth(),5,resScale);
		//cudaDeviceSynchronize();
		cudaMemcpy(pFractal,dev_fractal, ScreenHeight()*ScreenWidth()*sizeof(int), cudaMemcpyDeviceToHost);
	}
	int resScaleUpdate = 5;
	bool OnUserUpdate(float fElapsedTime) override
	{	
		resScaleUpdate--;
		if(resScaleUpdate < 1){
			resScaleUpdate = 1;
		}
		
        olc::vd2d vMouse = { (double)GetMouseX(), (double)GetMouseY() };

		// Handle Pan & Zoom
		if (GetMouse(2).bPressed)
		{
			vStartPan = vMouse;
		}

		if (GetMouse(2).bHeld)
		{
			vOffset -= (vMouse - vStartPan) / vScale;
			vStartPan = vMouse;
		}

		olc::vd2d vMouseBeforeZoom;
		ScreenToWorld(vMouse, vMouseBeforeZoom);

		if (GetKey(olc::Key::Q).bHeld || GetMouseWheel() > 0) {vScale *= 1.1; resScaleUpdate = 4;}
		if (GetKey(olc::Key::A).bHeld || GetMouseWheel() < 0) {vScale *= 0.9; resScaleUpdate = 4;}
		
		olc::vd2d vMouseAfterZoom;
		ScreenToWorld(vMouse, vMouseAfterZoom);
		vOffset += (vMouseBeforeZoom - vMouseAfterZoom);
		
		olc::vi2d pix_tl = { 0,0 };
		olc::vi2d pix_br = { ScreenWidth(), ScreenHeight() };
		olc::vd2d frac_tl = { -2.0, -1.0 };
		olc::vd2d frac_br = { 1.0, 1.0 };

		ScreenToWorld(pix_tl, frac_tl);
		ScreenToWorld(pix_br, frac_br);

		// Handle User Input
		if (GetKey(olc::K1).bPressed) nMode = 0;
		if (GetKey(olc::K2).bPressed) nMode = 1;
		if (GetKey(olc::K3).bPressed) nMode = 2;
		if (GetKey(olc::K4).bPressed) nMode = 3;
		if (GetKey(olc::UP).bPressed) nIterations += 256;
		if (GetKey(olc::DOWN).bPressed) nIterations -= 256;
		if (nIterations < 64) nIterations = 64;


		// START TIMING
		auto tp1 = std::chrono::high_resolution_clock::now();

		// Do the computation
		switch (nMode)
		{
		case 0: CreateFractalBasic(pix_tl, pix_br, frac_tl, frac_br, nIterations); break;
		case 1: CreateFractalCUDA(pix_tl, pix_br, frac_tl, frac_br, nIterations); break;
		case 2: CreateFractalCUDAOptimised(pix_tl, pix_br, frac_tl, frac_br, nIterations); break;
		case 3: CreateFractalCUDAStepZoom(pix_tl, pix_br, frac_tl, frac_br, nIterations,resScaleUpdate); break;
		}

		// STOP TIMING
		auto tp2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsedTime = tp2 - tp1;


		// Render result to screen
		for (int y = 0; y < ScreenHeight(); y++)
		{
			for (int x = 0; x < ScreenWidth(); x++)
			{
				int i = pFractal[y * ScreenWidth() + x];
				float n = (float)i;
				float a = 0.1f;
				// Thank you @Eriksonn - Wonderful Magic Fractal Oddball Man
				Draw(x, y, olc::PixelF(0.5f * sin(a * n) + 0.5f, 0.5f * sin(a * n + 2.094f) + 0.5f,  0.5f * sin(a * n + 4.188f) + 0.5f));
			}
		}

		// Render UI
		switch (nMode)
		{
		case 0: DrawString(0, 0, "1) Naive Method", olc::WHITE, 3); break;
		case 1: DrawString(0, 0, "2) CUDA Naive Method", olc::WHITE, 3); break;
		case 2: DrawString(0, 0, "3) CUDA Optimised Method", olc::WHITE, 3); break;
		case 3: DrawString(0, 0, "4) CUDA Step Zoom", olc::WHITE, 3); break;
		}

		DrawString(0, 30, "Time Taken: " + std::to_string(elapsedTime.count()) + "s", olc::WHITE, 3);
		DrawString(0, 60, "Iterations: " + std::to_string(nIterations), olc::WHITE, 3);
		return !(GetKey(olc::Key::ESCAPE).bPressed);
	}

	// Pan & Zoom variables
	olc::vd2d vOffset = { 0.0, 0.0 };
	olc::vd2d vStartPan = { 0.0, 0.0 };
	olc::vd2d vScale = { 1280.0 / 2.0, 720.0 };
	

	// Convert coordinates from World Space --> Screen Space
	void WorldToScreen(const olc::vd2d& v, olc::vi2d &n)
	{
		n.x = (int)((v.x - vOffset.x) * vScale.x);
		n.y = (int)((v.y - vOffset.y) * vScale.y);
	}

	// Convert coordinates from Screen Space --> World Space
	void ScreenToWorld(const olc::vi2d& n, olc::vd2d& v)
	{
		v.x = (double)(n.x) / vScale.x + vOffset.x;
		v.y = (double)(n.y) / vScale.y + vOffset.y;
	}
};
int main()
{
	Mandelbrot demo;
	if (demo.Construct(1280, 720, 1, 1))
		demo.Start();
	return 0;
}