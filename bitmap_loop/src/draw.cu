#include <opencv2/opencv.hpp>
#include <iostream>
#include "aux.h"
#include <cuda_runtime.h>

void drawImage(const uchar4* pixelData, unsigned int width, unsigned int height) {

    // Create a cv::Mat to hold the image data
    cv::Mat image(height, width, CV_8UC4);

    // Fill the cv::Mat with pixel data
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            const uchar4& pixel = pixelData[y * width + x];
            image.at<cv::Vec4b>(height-y-1, x) = cv::Vec4b(pixel.y, pixel.z, pixel.x, pixel.w); // OpenCV uses BGR order
        }
    }

    // Create a window to display the image
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", image);
    cv::waitKey(1);
}