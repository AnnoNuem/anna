#ifndef __Video__hpp__
#define __Video__hpp__

#include <opencv2/opencv.hpp>

using namespace cv;

class Video{
	VideoCapture cap;
	Mat frame;

	Video();

	void process();
};



#endif
