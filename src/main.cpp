#include "Motorsteuerung.hpp"
#include "Ultraschall.hpp"
#include "Video.hpp"
#include <curses.h>
#include <iostream>

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
	uchar hueLeft;
	uchar hueRight;
	uchar *frameData;
	uchar bias = 60;
	uchar multiplikator = 1;
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
		////////////////
		frameData = (uchar*) (v->frameBw.data);
		
		/**
		for(int i = 0; i < v->frameBw.cols; i++){
			printw(" \n");
			for(int j = 0; j < v->frameBw.rows; j++){
				printw(" %d", frameData[j + i]);
			}
		}
		**/
		///////////////
		hueLeft = (frameData[0]-bias) * multiplikator;
		hueRight = (frameData[1]-bias) * multiplikator;
		printw("HueLeft is: %d and HueRight is: %d", hueLeft, hueRight);
		if (distance < 60 && !ignoreUltraschall)
		{
			m.stop();
		}
		ch = getch();
		switch(ch){
			case keyLeft: m.left(); break;
			case keyRight: m.right(); break;
			case keyForward: m.faster(); break;
			case keyBackward: m.slower(); break;
			case keyHandbrake: m.stop(); break;
			case keyChangeDirection: m.changeDirection(); break;
			case keyIgnoreUltraschall: ignoreUltraschall = !ignoreUltraschall; break;
		}
		m.setLeftSpeed(hueRight);
		m.setRightSpeed(hueLeft);
	}while( ch != 'x');
	endwin();
	return 0;
}
