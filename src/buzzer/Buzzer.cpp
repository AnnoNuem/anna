#include "Buzzer.hpp"
Buzzer b;
using namespace std;

struct timespec duration;

Buzzer::Buzzer()
{
	wiringPiSetup();
	softToneCreate(BuzzerPin);

       	duration.tv_sec = 0;
	duration.tv_nsec = 0;  
	

	b = *this;
}


Buzzer::~Buzzer()
{
	softToneWrite(BuzzerPin, 0);
	pthread_join( thread1, NULL);
}

void Buzzer::beep(void)
{
	Buzzer::beep(2111,1000);
}

void Buzzer::beep(int durationMilSec)
{
	Buzzer::beep(2111,durationMilSec);
}

void Buzzer::beep(int frequency, int durationMilSec)
{
	softToneWrite(BuzzerPin, frequency);
	duration.tv_sec = (int)(durationMilSec / 1000);	
	duration.tv_nsec = (durationMilSec % 1000) * 1000;
	pthread_create( &thread1, NULL, wait , (void*) NULL);
}

void Buzzer::stop(void)
{
	softToneWrite(BuzzerPin, 0);
}

void *Buzzer::wait(void *ptr)
{
	nanosleep(&duration, NULL);
	stop();
}

	



