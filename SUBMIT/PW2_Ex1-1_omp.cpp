#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>  // for high_resolution_clock
#include <omp.h>


using namespace std;

void TrueAnaglyphs(const cv::Mat_<cv::Vec3b>& source, cv::Mat_<cv::Vec3b>& destination, const int i, const int j) //&: reference
{

  // Make sure i+400 doesn't exceed image width
  // if (i + source.cols/2 >= source.cols) return;

  cv::Vec3b pixel_l = source(i, j);
  cv::Vec3b pixel_r = source(i, j + source.cols/2);

  // Convert pixel to float for computation
  cv::Vec3f pixelValue_l, pixelValue_r;
  for (int c = 0; c < 3; ++c) {
      pixelValue_l[c] = static_cast<float>(pixel_l[c]);
      pixelValue_r[c] = static_cast<float>(pixel_r[c]);
  }

  // Luminance conversion weights
  cv::Vec3f m = {0.299f, 0.587f, 0.114f};
  cv::Vec3f anaglyphsValue;

  anaglyphsValue[2] = m[0] * pixelValue_l[2] + m[1] * pixelValue_l[1] + m[2] * pixelValue_l[0]; // Red from left
  anaglyphsValue[1] = 0.0f; // Green 
  anaglyphsValue[0] = m[0] * pixelValue_r[2] + m[1] * pixelValue_r[1] + m[2] * pixelValue_r[0]; // Blue from right
  

  // Final pixel value
  cv::Vec3b resultPixel;
  for (int c = 0; c < 3; ++c) {
    // float val = 255.0f * cos((255.0f - anaglyphsValue[c]) / 255.0f);
    // float val = anaglyphsValue[c];
    // resultPixel[c] = cv::saturate_cast<uchar>(val); // Clamp to [0, 255]
    resultPixel[c] = anaglyphsValue[c]; // Clamp to [0, 255]
  }

  destination(i, j) = resultPixel;
}

void GrayAnaglyphs(const cv::Mat_<cv::Vec3b>& source, cv::Mat_<cv::Vec3b>& destination, const int i, const int j) //&: reference
{
  cv::Vec3b pixel_l = source(i, j);
  cv::Vec3b pixel_r = source(i, j + source.cols/2);

  // Convert pixel to float for computation
  cv::Vec3f pixelValue_l, pixelValue_r;
  for (int c = 0; c < 3; ++c) {
      pixelValue_l[c] = static_cast<float>(pixel_l[c]);
      pixelValue_r[c] = static_cast<float>(pixel_r[c]);
  }

  // Luminance conversion weights
  cv::Vec3f m = {0.299f, 0.587f, 0.114f};
  cv::Vec3f anaglyphsValue;

  //!!!!!Careful! The order is B->G->R
  anaglyphsValue[2] = m[0] * pixelValue_l[2] + m[1] * pixelValue_l[1] + m[2] * pixelValue_l[0]; // Red from left
  anaglyphsValue[1] = m[0] * pixelValue_r[2] + m[1] * pixelValue_r[1] + m[2] * pixelValue_r[0]; // Green 
  anaglyphsValue[0] = m[0] * pixelValue_r[2] + m[1] * pixelValue_r[1] + m[2] * pixelValue_r[0]; // Blue from right

  // Final pixel value
  cv::Vec3b resultPixel;
  for (int c = 0; c < 3; ++c) {
    resultPixel[c] = anaglyphsValue[c]; // Clamp to [0, 255]
  }
  destination(i, j) = resultPixel;
}

void ColorAnaglyphs(const cv::Mat_<cv::Vec3b>& source, cv::Mat_<cv::Vec3b>& destination, const int i, const int j) //&: reference
{

  // Make sure i+400 doesn't exceed image width
  // if (i + source.cols/2 >= source.cols) return;

  cv::Vec3b pixel_l = source(i, j);
  cv::Vec3b pixel_r = source(i, j + source.cols/2);

  // Convert pixel to float for computation
  cv::Vec3f pixelValue_l, pixelValue_r;
  for (int c = 0; c < 3; ++c) {
      pixelValue_l[c] = static_cast<float>(pixel_l[c]);
      pixelValue_r[c] = static_cast<float>(pixel_r[c]);
  }

  // Luminance conversion weights
  cv::Vec3f anaglyphsValue;

  //!!!!!Careful! The order is B->G->R
  anaglyphsValue[2] = 1.0f * pixelValue_l[2] + 0.0f * pixelValue_l[1] + 0.0f * pixelValue_l[0]; // Red from left
  anaglyphsValue[1] = 0.0f * pixelValue_r[2] + 1.0f * pixelValue_r[1] + 0.0f * pixelValue_r[0]; // Green 
  anaglyphsValue[0] = 0.0f * pixelValue_r[2] + 0.0f * pixelValue_r[1] + 1.0f * pixelValue_r[0]; // Blue from right

  // Final pixel value
  cv::Vec3b resultPixel;
  for (int c = 0; c < 3; ++c) {
    // float val = 255.0f * cos((255.0f - anaglyphsValue[c]) / 255.0f);
    // float val = anaglyphsValue[c];
    // resultPixel[c] = cv::saturate_cast<uchar>(val); // Clamp to [0, 255]
    resultPixel[c] = anaglyphsValue[c]; // Clamp to [0, 255]
  }

  destination(i, j) = resultPixel;
}

void HalfColorAnaglyphs(const cv::Mat_<cv::Vec3b>& source, cv::Mat_<cv::Vec3b>& destination, const int i, const int j) //&: reference
{

  // Make sure i+400 doesn't exceed image width
  // if (i + source.cols/2 >= source.cols) return;

  cv::Vec3b pixel_l = source(i, j);
  cv::Vec3b pixel_r = source(i, j + source.cols/2);

  // Convert pixel to float for computation
  cv::Vec3f pixelValue_l, pixelValue_r;
  for (int c = 0; c < 3; ++c) {
      pixelValue_l[c] = static_cast<float>(pixel_l[c]);
      pixelValue_r[c] = static_cast<float>(pixel_r[c]);
  }

  // Luminance conversion weights
  cv::Vec3f m = {0.299f, 0.587f, 0.114f};
  cv::Vec3f anaglyphsValue;

  //!!!!!Careful! The order is B->G->R
  anaglyphsValue[2] = m[0] * pixelValue_l[2] + m[1] * pixelValue_l[1] + m[2] * pixelValue_l[0]; // Red from left
  anaglyphsValue[1] = 0.0f * pixelValue_r[2] + 1.0f * pixelValue_r[1] + 0.0f * pixelValue_r[0]; // Green 
  anaglyphsValue[0] = 0.0f * pixelValue_r[2] + 0.0f * pixelValue_r[1] + 1.0f * pixelValue_r[0]; // Blue from right


  // Final pixel value
  cv::Vec3b resultPixel;
  for (int c = 0; c < 3; ++c) {
    // float val = 255.0f * cos((255.0f - anaglyphsValue[c]) / 255.0f);
    // float val = anaglyphsValue[c];
    // resultPixel[c] = cv::saturate_cast<uchar>(val); // Clamp to [0, 255]
    resultPixel[c] = anaglyphsValue[c]; // Clamp to [0, 255]
  }

  destination(i, j) = resultPixel;
}

void OptimizedAnaglyphs(const cv::Mat_<cv::Vec3b>& source, cv::Mat_<cv::Vec3b>& destination, const int i, const int j) //&: reference
{

  // Make sure i+400 doesn't exceed image width
  // if (i + source.cols/2 >= source.cols) return;

  cv::Vec3b pixel_l = source(i, j);
  cv::Vec3b pixel_r = source(i, j + source.cols/2);

  // Convert pixel to float for computation
  cv::Vec3f pixelValue_l, pixelValue_r;
  for (int c = 0; c < 3; ++c) {
      pixelValue_l[c] = static_cast<float>(pixel_l[c]);
      pixelValue_r[c] = static_cast<float>(pixel_r[c]);
  }

  // Luminance conversion weights
  cv::Vec3f anaglyphsValue;

  //!!!!!Careful! The order is B->G->R
  anaglyphsValue[2] = 0.0f * pixelValue_l[2] + 0.7f * pixelValue_l[1] + 0.3f * pixelValue_l[0]; // Red from left
  anaglyphsValue[1] = 0.0f * pixelValue_r[2] + 1.0f * pixelValue_r[1] + 0.0f * pixelValue_r[0]; // Green 
  anaglyphsValue[0] = 0.0f * pixelValue_r[2] + 0.0f * pixelValue_r[1] + 1.0f * pixelValue_r[0]; // Blue from right

  // Final pixel value
  cv::Vec3b resultPixel;
  for (int c = 0; c < 3; ++c) {
    // float val = 255.0f * cos((255.0f - anaglyphsValue[c]) / 255.0f);
    // float val = anaglyphsValue[c];
    // resultPixel[c] = cv::saturate_cast<uchar>(val); // Clamp to [0, 255]
    resultPixel[c] = anaglyphsValue[c]; // Clamp to [0, 255]
  }

  destination(i, j) = resultPixel;
}

int main( int argc, char** argv )
{

  cv::Mat_<cv::Vec3b> source = cv::imread ( argv[1], cv::IMREAD_COLOR);
  cv::Mat_<cv::Vec3b> destination ( source.rows, source.cols/2 );

  cv::imshow("Source Image", source );

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 100;

  const int num_core = atoi(argv[3]);
  omp_set_num_threads(num_core);
  for (int it=0;it<iter;it++)
    {
      #pragma omp parallel for
      for (int i=0;i<source.rows;i++)
      {
	      for (int j=0;j<source.cols/2;j++)
        {
          // destination(i,j)[c] = 255.0*cos((255-source(i,j)[c])/255.0);

          ///////////////////////////////////////////////////////////////////
          // You can choose type of anaglyphs by command line
          ///////////////////////////////////////////////////////////////////
          std::string mode = argv[2];
          if (mode == "true") {
              TrueAnaglyphs(source, destination, i, j);
          } 
          else if (mode == "gray") {
              GrayAnaglyphs(source, destination, i, j);
          }
          else if (mode == "color") {
              ColorAnaglyphs(source, destination, i, j);
          }
          else if (mode == "halfColor") {
              HalfColorAnaglyphs(source, destination, i, j);
          }
          else if (mode == "optimized") {
              OptimizedAnaglyphs(source, destination, i, j);
          }
        } 
      }
    }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cv::imshow("Processed Image", destination );

  cout << "Source cols: " << source.cols << endl;
  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;

  
  
  cv::waitKey();
  return 0;
}
