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
	
	}



