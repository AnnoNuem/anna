#include "Motorsteuerung.hpp"
#include "Ultraschall.hpp"
#include "Buzzer.hpp"
#include <linux/joystick.h>
#include <curses.h>
#include <iostream>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#define NAME_LENGTH 128

int leftSpeed;
int rightSpeed;
int rightDirection = 1;
int leftDirection = 1;
float dif;
float speed;
float ls;
float rs;
bool stop = false;
bool ignoreStop = false;

// joystick variables
int fd;
unsigned char axes = 2;
unsigned char buttons = 2;
int version = 0x000800;
char name[NAME_LENGTH] = "Unknown";
struct js_event js;

const int leftShoulderButton = 4;
const int rightShoulderButton = 5;
const int leftShoulderAxis = 2;
const int rightShoulderAxis = 5;
const int xAxis = 1;
const int yAxis = 0;
const int buttonA = 0;
const int buttonB = 1;
const int buttonX = 8;
const int buttonStart = 7;
	
Motorsteuerung& m = Motorsteuerung::instance();
Ultraschall& us = Ultraschall::instance();
Buzzer& bz = Buzzer::instance();
	
PI_THREAD(buzz)
{
	bz.beep(2111,300);
	usleep(300000);
	bz.beep(1046,300);
}

PI_THREAD(readJoystick)
{
	while(true){
		if (read(fd, &js, sizeof(struct js_event)) != sizeof(struct js_event)) {
			perror("\njstest: error reading");
			exit (1);
		}

	
		switch(js.type){
		case JS_EVENT_AXIS:
			switch (js.number){
				case(leftShoulderAxis):
					rightSpeed = (js.value + 32767) *100 / 65534;
					break;
				case(rightShoulderAxis):
					leftSpeed = (js.value + 32767) * 100 / 65534;
					break;
				/**
				case(xAxis):
					dif = js.value / 32767;

					break;
				case(yAxis):
					speed = js.value / -32767;
					break;
				**/
			}
			break;
		case JS_EVENT_BUTTON:
			switch(js.number){
				case (leftShoulderButton):
					rightDirection = rightDirection * -1;
					break;
				case (rightShoulderButton):
					leftDirection = leftDirection * -1;
					break;
				case (buttonA):
					ignoreStop = !ignoreStop;
					break;
				case (buttonB):
					if (js.value == 1){
						piThreadCreate(buzz);
					}
					break;
				case (buttonX):
					system("shutdown -h now");
					exit(0);
					break;
				case (buttonStart):
					exit(0);
					break;
			}
			break;
		}

		if(!stop || ignoreStop)
		{
			m.setRightSpeed(rightSpeed * rightDirection);
			m.setLeftSpeed(leftSpeed * leftDirection);
			/**
			if (dif != 0)
			{
				ls = speed / ( dif + 1);
				rs = speed / ( (1/dif) + 1);
			}else
			{
				ls = speed;
				rs = speed;
			}
			m.setLeftSpeed((int)ls * 100);
			m.setRightSpeed((int)rs * 100);

			**/
		}
		else
		{
			m.setRightSpeed(0);
			m.setLeftSpeed(0);
		}
	}
}


int main(int argc, const char* argv[]){
	int ch;
	
	// input keys
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
	
	// joystick init
	if ((fd = open("/dev/input/js0", O_RDONLY)) < 0) {
		exit(1);
	}

	ioctl(fd, JSIOCGVERSION, &version);
	ioctl(fd, JSIOCGAXES, &axes);
	ioctl(fd, JSIOCGBUTTONS, &buttons);
	ioctl(fd, JSIOCGNAME(NAME_LENGTH), name);

	printw("Joystick (%s) has %d axes and %d buttons. Driver version is %d.%d.%d.\n",
		name, axes, buttons, version >> 16, (version >> 8) & 0xff, version & 0xff);

	int c = piThreadCreate (readJoystick);

	do{
		distance = us.getDistance();
		//distance = 61;
		if (distance < 60 )
		{
			stop = true;
		} 
		else 
		{
			stop = false;
		}


		//ch = getch();
		//clear();
		//printw("leftspeed: %d, rightspeed: %d", leftSpeed, rightSpeed);

		//switch(ch){
		//	case keyLeft: m.left(); break;
		//	case keyRight: m.right(); break;
		//	case keyForward: m.faster(); break;
		//	case keyBackward: m.slower(); break;
		//	case keyHandbrake: m.stop(); break;
		//	case keyChangeDirection: m.changeDirection(); break;
		//	case keyIgnoreUltraschall: ignoreUltraschall = !ignoreUltraschall; break;
		//}
	}while( ch != 'x');
	clear();
	endwin();
	return 0;
}
