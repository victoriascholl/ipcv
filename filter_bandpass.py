import ipcv
import numpy

def filter_bandpass(im, radialCenter, bandwidth, order=1, filterShape=ipcv.IPCV_IDEAL):

   '''

   title::
      filter_bandpass

   description::
      This method creates a bandpass filter to be applied to the 
      centered fourier transform of the input image.
      It calls the bandreject method in ipcv and subtracts the resulting 
      filter from 1, thereby only letting a band of frequencies from the
      input image pass through. 

   attributes::
      im
         Input image of tpye numpy nd array, used to get the dimensions for
         the frequency filter.
      radialCenter
        Distance (integer) from the center of the fourier transform at which
        the frequency range begins.
      bandwidth
        Thickness of the band of frequencies to be passed, measured
        outward from the radialCenter distance.  
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
               range to be passed. Binary mask. 
            BUTTERWORTH
               Gaussian-like shaped filter that can be tuned based on the order
               parameter. 
            GAUSSIAN
               Gaussian-shaped filter. 
   returns::
      bandpass filter - numpy array with the same dimensions as the input image.
        It looks like a donut! Allows frequencies to pass within the specified
        range and attenuates outsiders. 

   author::
      Victoria Scholl

   '''

   bandPassFilter = 1 - ipcv.filter_bandreject(im, radialCenter, bandwidth, order, filterShape)
   return bandPassFilter.astype(numpy.float64)



if __name__ == '__main__':

   import cv2
   import ipcv
   import numpy
   import matplotlib.pyplot
   import matplotlib.cm
   import mpl_toolkits.mplot3d

   filename = '/cis/faculty/cnspci/public_html/courses/common/images/lenna_color.tif'
   im = cv2.imread(filename)

   frequencyFilter = ipcv.filter_bandpass(im,
                                         16,
                                         2,
                                         filterShape=ipcv.IPCV_IDEAL)
   frequencyFilter = ipcv.filter_bandpass(im,
                                         32,
                                         1,
                                         order=2,
                                         filterShape=ipcv.IPCV_BUTTERWORTH)
   #frequencyFilter = ipcv.filter_bandpass(im,
   #                                      32,
   #                                      15,
   #                                      filterShape=ipcv.IPCV_GAUSSIAN)

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
