# Copyright (C) 2023 Simon D. Levy
# 
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

LIBDIR = /usr/lib/x86_64-linux-gnu
INCDIR = /usr/include/opencv4

all: capture

capture: capture.o
	# g++ -o capture capture.o -L$(LIBDIR) -lopencv_core -lopencv_highgui -lopencv_videoio
	g++ -o capture capture.o -L$(LIBDIR) -lopencv_core -lopencv_highgui -lopencv_videoio

capture.o: capture.cpp
	g++ -c -I$(INCDIR) capture.cpp 

run: capture
	./capture

clean:
	rm -f capture *.o
