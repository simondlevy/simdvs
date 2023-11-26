/*
Demo simdvs with camera

Copyright (C) 2023 Simon D. Levy

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 51
Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include<opencv2/opencv.hpp>
#include<iostream>
#include "simdvs.hpp"

using namespace std;
using namespace cv;

int main() 
{
    VideoCapture cap(0);

    static SimDvs dvs;

    while (true) { 

        Mat image;

        cap >> image;

        auto events = dvs.getEvents(image);

        if (!dvs.display(image, events)) {
            break;
        }

        /*
           imshow("Events", img);

           if (waitKey(1) == 27) {  // ESC
           break;
           }*/
    }

    cap.release();

    return 0;
}
