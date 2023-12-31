/*
A ++ class to simulate a Dynamic Vision Sensor through first-differencing
of current and previous image from an ordinary camera.

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

#pragma once

#include<opencv2/opencv.hpp>

using namespace cv;

class SimDvs {

    public:

        SimDvs(void)
        {
        }

        void * getEvents(const Mat & image)
        {
            return NULL;
        }

        bool display(const Mat image, const void * events)
        {
            imshow("Events", image);

            return waitKey(1) != 27; // ESC
        }

};

/*
class _Event:
    '''Mirrors event structure in dvs-filter repo
    '''

    def __init__(self, timestamp, x, y):

        self.timestamp = timestamp
        self.x = x
        self.y = y


class SimDvs:

    def __init__( self, resolution=None, threshold=0):

        self.resolution = resolution
        self.threshold = threshold

        self.image_prev = None
        self.start_time = time()

    def getEvents(self, image):
        '''
        Returns current event image, or None if user quits display
        '''

        # Convert the full-size image to grayscale
        graycurr = self._color2gray(image)

        # Create an empty events array of the same size as the grayscale image
        events = np.zeros(graycurr.shape, dtype=np.int8)

        # Once we have a current and previous image, we can compute events
        if self.image_prev is not None:

            # Make a first-difference image between the current and previous
            # grayscale images
            grayprev = self._color2gray(self.image_prev)
            diffimg = graycurr - grayprev

            # Simulate positive and negative events w.r.t. a threshold
            events[diffimg > +self.threshold] = +1
            events[diffimg < -self.threshold] = -1

        # Track the previous image for first-differencing
        self.image_prev = image

        # If sensor resolution was indicated, resize the event image now
        if self.resolution is not None:
            events = cv2.resize(events.astype('float32'), (128, 128))

        return events

    def annotate(self, ceventimg):
        '''
        Override this method to annotate the event image.
        '''
        pass

    def display(self, image, events, scaleup=1, quit_key=27, colorize=True):

        # Make a color image from the event image
        rows, cols = events.shape
        ceventimg = self._colorize(events, colorize)

        # Make two-column image to display the original and events
        rows, cols = events.shape
        wideimg = np.zeros((rows, 2 * cols, 3)).astype(np.uint8)

        # Fill the first column with the original (resized if # indicated)
        wideimg[:, :cols, :] = (
                image if self.resolution is None 
                else cv2.resize(image, self.resolution))

        # Fill the second column with the events image
        wideimg[:, cols:(2*cols), :] = ceventimg

        # Scale up the two-column image
        bigimg = cv2.resize(
                wideimg, 
                (scaleup * wideimg.shape[1], scaleup * wideimg.shape[0]))

        # Support annotating the scaled-up image in a subclass
        self.annotate(bigimg[:, scaleup*cols:(2*scaleup*cols), :])

        # Display the scaled-up image
        cv2.imshow('Events', bigimg)
        # Check whether the user hit the quit key
        if cv2.waitKey(1) == quit_key:
            return False

        return True

    def filter(self, events, noise_filter):

        filtered = np.zeros(events.shape)

        nz = np.where(events)

        for x, y in zip(nz[0], nz[1]):

            e = _Event(int((time() - self.start_time) * 1e6), x, y)

            if noise_filter.check(e):

                filtered[x, y] = events[x, y]

        return filtered

    def _colorize(self, events, colorize):

        rows, cols = events.shape
        ceventimg = np.zeros((rows, cols, 3))

        # If color was indicated, display positive events as green,
        # negative as red
        if colorize:

            ceventimg[events == +1, 1] = 255
            ceventimg[events == -1, 2] = 255

        # Otherwise, display all events as white
        else:
            ceventimg[events != 0, :] = 255

        return ceventimg

    def _color2gray(self, img):

        # Convert to grayscale and downsample for subtraction
        return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) >> 1).astype(np.int8)

*/
