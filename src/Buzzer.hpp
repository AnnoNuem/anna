#ifndef ___Buzzer__hpp___
#define ___Buzzer__hpp___

#include <wiringPi.h>
#include <iostream>
#include <unistd.h>
#include <softTone.h>
#include <time.h>

class Buzzer{

	private:
	const static int BuzzerPin = 16;

	pthread_t thread1;

	static void stop(void);
	static void *wait(void *ptr);

	public:
	Buzzer();

	

	static Buzzer& instance()
	{
		static Buzzer _instance;
		return _instance;
	}

	~Buzzer();
	
	void beep(void);

	void beep(int);

	void beep(int, int);
};

#endif //___Buzzer__hpp___

