#ifndef __Video__hpp__
#define __Video__hpp__

#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

using namespace cv;

class Video{
	VideoCapture cap;
	Mat frame;
	Mat frameSmall;
	public:
	Mat frameBw;
	private:
	Size sizeSmall;


	private:
	int count;

	Video();

	Video(const Video&);

	Video& operator= (const Video);

	public:
	void process();

	static Video& instance()
	{
		static Video _instance;
		return _instance;
	}

	~Video(){};
};



#endif
