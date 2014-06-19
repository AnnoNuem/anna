#include <Video.hpp>
//#include <iostream>

	Video::Video()
	{
		cap.open(0);
		if(!cap.isOpened())
		{
			throw -1;
		}
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 160);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
//		sizeSmall = new Size(12,9);
		count = 0;
	}

	void Video::process()
	{
		count++;
		cap >> frame;
//		MatConstIterator_<Vec3b> it = frame.begin<Vec3b>(), it_end = frame.end<Vec3b>();
//		for(; it != it_end; ++it )
//		{
//		}
		std::stringstream ss;
		ss << "image" << count << ".jpg";
		resize(frame,frameSmall, Size(2,1));
		cvtColor(frameSmall, frameBw, CV_RGB2GRAY);
//		imwrite(ss.str(), frameBw);
	}



