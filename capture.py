#!/usr/bin/python3
'''
A Python class to simulate a Dynamic Vision Sensor through first-differencing
of current and previous image from an ordinary camera.

Run it as a standalone script to see the difference between two images
in the images/ directory.

Copyright (C) 2023 Simon D. Levy, Armando Mendez-Anastasio

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
'''

import cv2

from simdvs import SimDvs

dvs = SimDvs(threshold=4, display_scaleup=1)

cap = cv2.VideoCapture(0)

while True:

    _, image = cap.read()

    dvs.getEvents(image)

    if cv2.waitKey(1) == 27:
        break
