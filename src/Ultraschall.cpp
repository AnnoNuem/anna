#include "Ultraschall.hpp"
Ultraschall u;
using namespace std;

void Ultraschall::signalStart(void)
{
	u.signalStartTime = chrono::high_resolution_clock::now();
	u.signalActive = true;
}

void Ultraschall::signalEnd(void)
{
	u.signalEndTime = chrono::high_resolution_clock::now();
	u.difTime = chrono::duration_cast<chrono::microseconds>(u.signalEndTime - u.signalStartTime).count() ;
	u.signalActive = false;
}

void Ultraschall::edgeChange(void)
{
	if(u.signalActive)
	{
		u.signalEnd();
	}
	else
	{
		u.signalStart();
	}
}

Ultraschall::Ultraschall()
{
	wiringPiSetup();
	pinMode(UltraschallTriggerPin, OUTPUT);
	pinMode(UltraschallSignalPin, INPUT);
	digitalWrite(UltraschallTriggerPin, LOW);
	wiringPiISR(UltraschallSignalPin, INT_EDGE_BOTH, *edgeChangeWrapper);


	u = *this;
}


Ultraschall::~Ultraschall()
{
	digitalWrite(UltraschallTriggerPin, LOW);
}

void Ultraschall::edgeChangeWrapper(void)
{
	u.edgeChange();
}
	
unsigned int Ultraschall::getDistance(void)
{
	digitalWrite(UltraschallTriggerPin, HIGH);
	u.signalActive = false;
	//delayMicroseconds(10);
	usleep(10000);
	digitalWrite(UltraschallTriggerPin, LOW);

	usleep(30000);
	/**if (u.difTime > 23200)
	*{
		return 400;
	}
	if (u.difTime <= 116)
	{
		return 0;
	}
**/
	return u.difTime/29;
}



