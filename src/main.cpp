#include "Motorsteuerung.hpp"
#include "Ultraschall.hpp"
#include "Video.hpp"
#include <curses.h>
#include <iostream>
#include <math.h>

int main(int argc, const char* argv[]){
	int ch;

	const char keyForward = 'w';
	const char keyBackward = 's';
	const char keyLeft = 'a';
	const char keyRight = 'd';
	const char keyHandbrake = ' ';
	const char keyChangeDirection = 'r';
	const char keyIgnoreUltraschall = 'f';

	unsigned int distance;

	// braitenberg parameter
	const char keyBiasUp = 'u';
	const char keyBiasDown = 'h';
	const char keyMultUp = 'i';
	const char keyMultDown = 'j';
	double hueLeft;
	double hueRight;
	double hueLeftOld;
	double hueRightOld;
	uchar *frameData;
	double bias = -98;
	double biasDelta = 1;
	double multiplikator = 1;
	double multiplikatorDelta = 0.1;
	double delta = 0.5;



	bool ignoreUltraschall = false;

	initscr();
	noecho();
	nonl();
	timeout(0);
	keypad(stdscr, TRUE);
	
	Motorsteuerung& m = Motorsteuerung::instance();
	Ultraschall& u = Ultraschall::instance();
	Video* v = 0;
	try
	{
		v = &Video::instance();
	}
	catch (int e)
	{
		if (e == -1)
		{
			endwin();
			std::cout << "Unable to open Video Source.\n";
			return -1;
		}
	}

	
	do{
		distance = u.getDistance();
		clear();
		printw("Distance is: %d \n", distance);
		v->process();

		// braitenberg computations
		frameData = (uchar*) (v->frameBw.data);
		
		
		hueLeft = ((double)frameData[0]+bias) * multiplikator;
		hueRight = ((double)frameData[1]+bias) * multiplikator;
		

		hueLeft = (hueLeft + hueLeftOld) / 2;
		hueRight = (hueRight + hueRightOld) / 2;

		if (hueRight < 0)
		{
			hueRight = 0;
		}
		if (hueRight > 100)
		{
			hueRight = 100;
		}

		if (hueLeft < 0)
		{
			hueLeft= 0;
		}
		if (hueLeft > 100)
		{
			hueLeft = 100;
		}

		printw("HueLeft is: %f and HueRight is: %f \n", hueLeft, hueRight);
		printw("Bias is: %f and multiplikator is: %f", bias, multiplikator);
		if (distance < 60 && !ignoreUltraschall)
		{
			m.stop();
		}
		else
		{
			ch = getch();
			switch(ch)
			{
				case keyLeft: m.left(); break;
				case keyRight: m.right(); break;
				case keyForward: m.faster(); break;
				case keyBackward: m.slower(); break;
				case keyHandbrake: m.stop(); break;
				case keyChangeDirection: m.changeDirection(); break;
				case keyIgnoreUltraschall: ignoreUltraschall = !ignoreUltraschall; break;
				case keyBiasUp: bias += biasDelta; break;
				case keyBiasDown: bias -= biasDelta; break;
				case keyMultUp: multiplikator += multiplikatorDelta; break;
				case keyMultDown: multiplikator -= multiplikatorDelta; break;
			}
			m.setLeftSpeed(hueLeft);
			m.setRightSpeed(hueRight);
		}
		hueRightOld = hueRight;
		hueLeftOld = hueLeft;
	}while( ch != 'x');
	endwin();
	return 0;
}
