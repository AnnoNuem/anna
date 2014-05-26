#include <Video.hpp>
#include <iostream>

	Video::Video()
	{
		cap.open(0);
		if(!cap.isOpened())
		{
			throw -1;
		}
//		cap.set(CV_CAP_PROP_FRAME_WIDTH, 160);
//		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
	}

	void Video::process()
	{
		cap >> frame;
		MatConstIterator_<Vec3b> it = frame.begin<Vec3b>(), it_end = frame.end<Vec3b>();
		for(; it != it_end; ++it )
		{
		}

	
	}



