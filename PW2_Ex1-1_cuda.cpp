// Execution command /////////////////////////////////////////
//  /usr/local/cuda/bin/nvcc -O3 PW2_Ex1-1_cuda.cu `pkg-config opencv4 --cflags --libs` PW2_Ex1-1_cuda.cpp -o PW2_Ex1-1_cuda
//  ./PW2_Ex1-1_cuda statue.jpg result.png 500 true
//  xdg-open result.png
//////////////////////////////////////////////////////////////

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;
using namespace cv;

void startCUDA ( cv::cuda::GpuMat& src,cv::cuda::GpuMat& dst, int modeN, int blockLines, int blockCols);


int main( int argc, char** argv )
{
  cv::Mat_<Vec3f> h_imaRGB = cv::imread(argv[1]);
  cv::Mat_<Vec3f> h_result  ( h_imaRGB.rows, h_imaRGB.cols/2 ); 

  for (int i=0;i<h_imaRGB.rows;i++)
    for (int j=0;j<h_imaRGB.cols;j++)
      for (int c=0;c<3;c++)
	    h_imaRGB(i,j)[c] /= 255.0; // normalization--> 0-1

  cv::cuda::GpuMat d_imaRGB, d_result; // two space

  d_imaRGB.upload ( h_imaRGB ); // internarly cpp execute cudaMalloc
  d_result.upload ( h_result );
  
  auto begin = chrono::high_resolution_clock::now();

  std::string mode = argv[4];
  int modeN = 0;
  if (mode == "true") { modeN = 0; } 
  else if (mode == "gray") { modeN = 1; }
  else if (mode == "color") { modeN = 2; }
  else if (mode == "halfColor") { modeN = 3; }
  else if (mode == "optimized") { modeN = 4; }

  int blockLines;
  int blockCols;
  int iter = atoi(argv[3]);
  if (argc > 5) {
    blockLines = atoi(argv[5]);
    blockCols = atoi(argv[6]);
  } else {
    blockLines = 32;
    blockCols = 8;
  }
  
  for (int i=0;i<iter;i++)
  {
    startCUDA(d_imaRGB, d_result, modeN, blockLines, blockCols);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;
  
  // cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;
  
  d_result.download(h_result);
  
  for (int i=0;i<h_result.rows;i++)
    for (int j=0;j<h_result.cols;j++)
      for (int c=0;c<3;c++)
	    h_result(i,j)[c] *= 255.0;
  
  imwrite ( argv[2], h_result );

  return 0;
}