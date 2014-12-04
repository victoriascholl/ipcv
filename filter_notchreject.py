import ipcv
import numpy

def filter_notchreject(im, notchCenter, notchRadius, order=1, filterShape=ipcv.IPCV_IDEAL):

   '''
   title::
      filter_notchreject

   description::
      This method creates a frequency filter that rejects
      specific frequencies supplied by the user.
      User must specify the coordinates of the frequencies to be blocked 
      (in the form of a list of tuples (u,v)), each with a corresponding
      radius value (in the list, notchRadius)
      A mask is created to preserve the other frequencies in the image.
      First, the dimensions of the input image are determined.
      Then an array of ones is created with the same dimensions to serve 
      as the notchreject filter. 
      The dist method is used to return a distance value at each pixel
      location (measured from the corner of the image). This dist filter
      is then rolled twice in both the x and y dimensions to center the 
      values about the center of the array. 
      The specified filter shape is then created based on the use's input. 
      The filter is returned as type numpy.float64. The filter is ready to be 
      multiplied with a centered fourier transform of the input image.

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
         the frequencies differently. 
            IDEAL
               Ideal shaped filter STRICTLY blocks ALL frequencies within the
               specified notch sizes and locations. Binary mask. 
            BUTTERWORTH
               Gaussian-like shaped filter that can be tuned based on the order
               parameter. 
            GAUSSIAN
               Gaussian-shaped filter. 
   returns::
      notchreject filter - numpy array with the same dimensions as the input image
      that will preserve all frequencies outside of the specified notches.

   author::
      Victoria Scholl
   '''

   # get image dimensions, which dictate the filter dimensions
   imRows, imColumns, imBands, dataType = ipcv.dimensions(im)
   notchRejectFilter = numpy.ones([imRows,imColumns])

   # use ipcv.dist to generate an array with distances from the corner pixel.
   # roll twice to center the array (distances measured from center) 
   distFilter = ipcv.dist( ( imRows, imColumns ) )
   distFilter = numpy.roll(numpy.roll(distFilter, imRows/2, axis=0), imColumns/2, axis=1)

   for center in range(len(notchCenter)):
      # D1 is the distance from the notch
      # D2 is the distance from the notch's conjugate 
      D1 = numpy.roll(numpy.roll(distFilter,imRows/2 - notchCenter[center][0],axis=1),imColumns/2-notchCenter[center][1],axis=0)
      D2 = numpy.roll(numpy.roll(distFilter,imRows/2 + notchCenter[center][0],axis=1),imColumns/2+notchCenter[center][1],axis=0)
     

      if filterShape == ipcv.IPCV_IDEAL:
         notchRejectFilter[ D1 <= notchRadius[center] ] = 0
         notchRejectFilter[ D2 <= notchRadius[center] ] = 0

      elif filterShape == ipcv.IPCV_BUTTERWORTH:  # butterworth equation
         
         # To avoid a div by 0 error
         productOfDs = D1 * D2
         productOfDs[productOfDs == 0] == 1e-10


         nextNotchRejectFilter = 1 / ( 1 + (( notchRadius[center]**2 ) / (productOfDs)) ** order) 
         notchRejectFilter = notchRejectFilter * nextNotchRejectFilter

      else: # Third filter type option is Gaussian
         nextNotchRejectFilter = 1 - numpy.exp( -0.5 * (D1 * D2) / ( notchRadius[center]**2) )
         notchRejectFilter = notchRejectFilter * nextNotchRejectFilter

   notchRejectFilter = numpy.roll(numpy.roll(notchRejectFilter, imRows/2, axis=0), imColumns/2, axis=1)
   
   return notchRejectFilter.astype(numpy.float64)



if __name__ == '__main__':

   import cv2
   import ipcv
   import numpy
   import matplotlib.pyplot
   import matplotlib.cm
   import mpl_toolkits.mplot3d

   filename = '/cis/faculty/cnspci/public_html/courses/common/images/lenna_color.tif'
   filename = 'lena_periodic.jpg'
   im = cv2.imread(filename)

   notchCenter = [(32,0)]
   notchRadius = [3]
   frequencyFilter = ipcv.filter_notchreject(im,
                                         notchCenter,
                                         notchRadius,
                                         filterShape=ipcv.IPCV_IDEAL)
   frequencyFilter = ipcv.filter_notchreject(im,
                                         notchCenter,
                                         notchRadius,
                                         order=1,
                                         filterShape=ipcv.IPCV_BUTTERWORTH)
   #frequencyFilter = ipcv.filter_notchreject(im,
   #                                      notchCenter,
   #                                      notchRadius,
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
