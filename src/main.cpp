#include "Motorsteuerung.hpp"
#include "Ultraschall.hpp"
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
	bool ignoreUltraschall = false;

	initscr();
	noecho();
	nonl();
	timeout(0);
	keypad(stdscr, TRUE);
	
	Motorsteuerung& m = Motorsteuerung::instance();
	Ultraschall& u = Ultraschall::instance();
	
	do{
		distance = u.getDistance();
		clear();
		printw("Distance is: %d", distance);
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
	}while( ch != 'x');
	endwin();
	return 0;
}
