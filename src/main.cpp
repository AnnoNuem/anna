#include "Motorsteuerung.hpp"
#include "Ultraschall.hpp"
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

int main(int argc, const char* argv[]){
	int ch;

	int leftSpeed;
	int rightSpeed;

	// joystick variables
	int fd;
	unsigned char axes = 2;
	unsigned char buttons = 2;
	int version = 0x000800;
	char name[NAME_LENGTH] = "Unknown";
	struct js_event js;

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
	
	Motorsteuerung& m = Motorsteuerung::instance();
	Ultraschall& u = Ultraschall::instance();
	
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

	do{
		//distance = u.getDistance();
		//distance = 61;
		//if (distance < 60 && !ignoreUltraschall)
		//{
		//	m.stop();
		//}
		//ch = getch();
		if (read(fd, &js, sizeof(struct js_event)) != sizeof(struct js_event)) {
			perror("\njstest: error reading");
			exit (1);
		}

		
		if (js.type == JS_EVENT_AXIS){
			if (js.number == 5){
				leftSpeed = (js.value + 32767) * 100 / 65534;
				m.setLeftSpeed(leftSpeed);
			}
			else if (js.number == 2){
				rightSpeed = (js.value + 32767) *100 / 65534;
				m.setRightSpeed(rightSpeed);
			}
		}

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
	endwin();
	return 0;
}
