#ifndef __Ultraschall__hpp__
#define __Ultraschall__hpp__

#include <wiringPi.h>
#include <iostream>
//#include <time.h>
#include <chrono>
#include <unistd.h>

class Ultraschall{

	private:
	const static int UltraschallTriggerPin = 13;
	const static int UltraschallSignalPin = 14;
//	static Ultraschall u;
	std::chrono::high_resolution_clock::time_point signalStartTime;
	std::chrono::high_resolution_clock::time_point signalEndTime;
	double difTime;

//	clock_t signalStartTime;
//	clock_t signalEndTime;

	bool signalActive;
//	bool finished;

	void signalStart(void);
	void signalEnd(void);
	void edgeChange(void);

	static void edgeChangeWrapper(void);

	public:
	Ultraschall();
//	Ultraschall(const Ultraschall&);
//	Ultraschall& operator= (const Ultraschall&)
//	{return this;}

	static Ultraschall& instance()
	{
		static Ultraschall _instance;
		return _instance;
	}

	~Ultraschall();
	
	unsigned int getDistance(void);

};

#endif //__Ultraschall__hpp__

