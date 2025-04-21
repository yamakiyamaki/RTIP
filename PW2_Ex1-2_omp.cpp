#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>  // for high_resolution_clock


using namespace std;

float Gaussian(float x, float y, float sigma) {
  return std::exp(-(x * x + y * y) / (2.0f * sigma * sigma)) / (2.0f * 3.14159265f * sigma * sigma);
}

void Gaussian_conv(const cv::Mat_<cv::Vec3b>& source, cv::Mat_<cv::Vec3b>& destination, int i, int j, float kernelSize, float sigma) {
  int sizeDiv2 = static_cast<int>(kernelSize / 2.0);
  int halfWidth = source.cols/2;
  cv::Vec3f pixelValue(0.0f, 0.0f, 0.0f);
  cv::Vec3f colorSum(0.0f, 0.0f, 0.0f);
  float weightSum = 0.0f;

  for (int dy = -sizeDiv2; dy <= sizeDiv2; ++dy) {
      for (int dx = -sizeDiv2; dx <= sizeDiv2; ++dx) {
          int y = i + dy;
          int x = j + dx;

          // Taking care about edges and center.
          if (x < 0) {
            pixelValue = source(y, abs(x-1));
          } else if (x > source.cols) {
            pixelValue = source(y, source.cols-x+1);
          } else if (y < 0) {
            pixelValue = source(abs(y-1), x);
          } else if (y > source.rows) {
            pixelValue = source(source.rows-y+1, x);
          } else if (j <= halfWidth){
            if ( halfWidth-x >= 0){
              pixelValue = source(y, x);
            } else {
              pixelValue = source(y, halfWidth-x+1+halfWidth);
            }
          } else if (j > halfWidth){
            if ( x-halfWidth > 0){
              pixelValue = source(y, x);
            } else {
              pixelValue = source(y, halfWidth-x+1+halfWidth);
            }
          } else {
            pixelValue = source(y, x);
          }


          float weight = Gaussian(dx, dy, sigma);
          // weightSum += weight;
          for (int c = 0; c < 3; ++c)
              colorSum[c] += weight * pixelValue[c];
      }
  }

  // cv::Vec3b result;
  // for (int c = 0; c < 3; ++c)
  //     result[c] = cv::saturate_cast<uchar>(colorSum[c] / weightSum);
  // destination(i, j) = result;

  destination(i, j) = colorSum;
}

int main( int argc, char** argv )
{

  cv::Mat_<cv::Vec3b> source = cv::imread ( argv[1], cv::IMREAD_COLOR);
  cv::Mat_<cv::Vec3b> destination ( source.rows, source.cols );

  cv::imshow("Source Image", source );

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 500;

  #pragma omp parallel for
  for (int it=0;it<iter;it++)
    {
      for (int i=0;i<source.rows;i++)
      {
	      for (int j=0;j<source.cols;j++)
        {
          // destination(i,j)[c] = 255.0*cos((255-source(i,j)[c])/255.0);

          ///////////////////////////////////////////////////////////////////
          // You can choose type of anaglyphs by command line
          ///////////////////////////////////////////////////////////////////
          std::string mode = argv[2];
          if (mode == "gauss"){
            if (argc <= 3) {
              cout << "Lack of input argument. Usage: ./execute.sh program image mode kSize sigma" << endl;
            }
            const float kSize = atof(argv[3]);
            const float sigma = atof(argv[4]);
            Gaussian_conv(source, destination, i, j, kSize, sigma);
          }
        } 
      }
    }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cv::imshow("Processed Image", destination );

  
  cout << "kSize " << argv[3] << endl;
  cout << "sigma " << argv[4] << endl;
  cout << "Source cols: " << source.cols << endl;
  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;

  
  
  cv::waitKey();
  return 0;
}
