import ipcv
import numpy

def filter_notchpass(im, notchCenter, notchRadius, order=1, filterShape=ipcv.IPCV_IDEAL):

   '''
title::
      filter_notchpass

   description::
      This method creates a notchpass filter to be applied to the 
      centered fourier transform of the input image.
      It calls the notchreject method in ipcv and subtracts the resulting 
      filter from 1, thereby only letting through the notch frequencies. 

   attributes::
      im
         Input image of tpye numpy nd array, used to get the dimensions for
         the frequency filter.
      notchCenter
        List of tuples with coordinates (u,v) corresponding to frequencies
        in the fourier transform to be rejected. 
      notchRadius
        The size of the notch to be created at each (u,v) coordinate. 
      order
         Integer value that influences the shape of the butterworth filter. 
         The higher the order, the more the butterworth filter resembles an 
         ideal filter. 
      filterShape
         Options for the shape of the filter, specified in the constants.py 
         file in the ipcv directory. The different shapes will attenuate 
         the higher frequencies differently. 
            IDEAL
               Ideal shaped filter STRICTLY allows ALL frequencies within the 
               specified notch sizes and locations. Binary mask. 
            BUTTERWORTH
               Gaussian-like shaped filter that can be tuned based on the order
               parameter. 
            GAUSSIAN
               Gaussian-shaped filter. 
   returns::
      notchpass filter - numpy array with the same dimensions as the input image.

   author::
      Victoria Scholl

 
   '''
   notchPassFilter = 1 - ipcv.filter_notchreject(im, notchCenter, notchRadius, order, filterShape)

   return notchPassFilter.astype(numpy.float64)



if __name__ == '__main__':

   import cv2
   import ipcv
   import numpy
   import matplotlib.pyplot
   import matplotlib.cm
   import mpl_toolkits.mplot3d

   filename = '/cis/faculty/cnspci/public_html/courses/common/images/lenna_color.tif'
   im = cv2.imread(filename)

   frequencyFilter = ipcv.filter_notchpass(im,
                                         16,
                                         filterShape=ipcv.IPCV_IDEAL)
   frequencyFilter = ipcv.filter_notchpass(im,
                                         16,
                                         order=1,
                                         filterShape=ipcv.IPCV_BUTTERWORTH)
   frequencyFilter = ipcv.filter_notchpass(im,
                                         16,
                                         filterShape=ipcv.IPCV_GAUSSIAN)

   # Create a 3D plot and image visualization of the frequency domain filter
   rows = im.shape[0]
   columns = im.shape[1]
   u = numpy.arange(-columns/2, columns/2, 1)
   v = numpy.arange(-rows/2, rows/2, 1)
   u, v = numpy.meshgrid(u, v)

   figure = matplotlib.pyplot.figure('Frequency Domain Filter', (14, 6))
   p = figure.add_subplot(1, 2, 1, projection='3d')
   p.set_xlabel('u')
   p.set_xlim3d(-columns/2, columns/2)
   p.set_ylabel('v')
   p.set_ylim3d(-rows/2, rows/2)
   p.set_zlabel('Weight')
   p.set_zlim3d(0, 1)
   p.plot_surface(u, v, frequencyFilter)
   i = figure.add_subplot(1, 2, 2)
   i.imshow(frequencyFilter, cmap=matplotlib.cm.Greys_r)
   matplotlib.pyplot.show()
