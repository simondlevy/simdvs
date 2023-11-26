all: capture

capture: capture.o
	g++ -o capture capture.o -L/usr/lib/x86_64-linux-gnu/ -lopencv_core -lopencv_highgui -lopencv_videoio

capture.o: capture.cpp
	g++ -c -I/usr/include/opencv4/ capture.cpp 

clean:
	rm -f capture *.o
