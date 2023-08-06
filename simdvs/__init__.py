'''
A Python class to simulate a Dynamic Vision Sensor through first-differencing
of current and previous image from an ordinary camera.

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

import numpy as np
import cv2


class SimDvs:

    def __init__(
            self,
            threshold=0,
            display_scaleup=0,
            quit_key=27,
            colorize=True):

        self.threshold = threshold
        self.display_scaleup = display_scaleup
        self.quit_key = quit_key
        self.colorize = colorize

        self.image_prev = None

    def getEvents(self, image):
        '''
        Returns current event image, or None if user quits display
        '''

        graycurr = self._color2gray(image)

        events = np.zeros(graycurr.shape, dtype=np.int8)

        if self.image_prev is not None:

            grayprev = self._color2gray(self.image_prev)
            diffimg = graycurr - grayprev

            events[diffimg > +self.threshold] = +1
            events[diffimg < -self.threshold] = -1

            if self.display_scaleup > 0:

                rows, cols = events.shape

                eventimg = np.zeros((rows, cols, 3))

                if self.colorize:
                    eventimg[events == +1, 1] = 255
                    eventimg[events == -1, 2] = 255
                else:
                    eventimg[events !=0, :] = 255

                self.annotate(eventimg)

                rows, cols = events.shape

                bigimg = np.zeros((rows, 2*cols, 3)).astype(np.uint8)

                bigimg[:, :cols, :] = image
                bigimg[:, cols:(2*cols), :] = eventimg

                cv2.imshow('Events',
                           cv2.resize(bigimg,
                                      (self.display_scaleup *
                                       bigimg.shape[1],
                                       self.display_scaleup *
                                       bigimg.shape[0])))

                if cv2.waitKey(1) == self.quit_key:

                    events = None

        self.image_prev = image

        return events

    def annotate(self, eventimg):
        '''
        Override this method to annotate the event image.
        '''

        pass

    def _color2gray(self, img):

        # Convert to grayscale and downsample for subtraction
        return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) >> 1).astype(np.int8)
