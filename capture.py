#!/usr/bin/python3
'''
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
'''

import cv2
import numpy as np

from simdvs import SimDvs

# dvs = SimDvs(threshold=4, display_scale=1, resolution=(128,128))
dvs = SimDvs(threshold=4)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    _, image = cap.read()

    events = dvs.getEvents(image)

    if not dvs.display(image, events):

        break
