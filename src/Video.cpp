#include <Video.hpp>

	Video::Video()
	{
		cap.open(0);
		if(!cap.isOpened())
		{
			cap = 0;
		}
	}

	void Video::process()
	{
		cap >> frame;
		MatConstIterator_<Vec3b> it = frame.begin<Vec3b>(), it_end = frame.end<Vec3b>();
		for(; it != it_end; ++it )
		{
		}

	
	}



