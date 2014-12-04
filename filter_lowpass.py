import ipcv
import numpy

def filter_lowpass(im, cutoffFrequency, order=1, filterShape=ipcv.IPCV_IDEAL):

   '''
   title::
      filter_lowpass

   description::
      This method creates a lowpass filter to be applied to the 
      centered fourier transform of the input image.
      First, the dimensions of the input image are determined.
      Then an array of zeros is created with the same dimensions to serve 
      as the lowpass filter. 
      The dist method is used to return a distance value at each pixel
      location (measured from the corner of the image). This dist filter
      is then rolled twice in both the x and y dimensions to center the 
      values about the center of the array. 
      The specified filter shape is then created based on the use's input. 
      The filter is returned as type numpy.float64; ready to be applied to
      the input image using the frequency_filter.py method. 

   attributes::
      im
         Input image of tpye numpy nd array, used to get the dimensions for
         the frequency filter.
      cutoffFrequency
         Frequency of type integer above which to attentuate higher frequencies
         (frequencies lower than the cutoff value will be preserved). 
      order
         Integer value that influences the shape of the butterworth filter. 
         The higher the order, the more the butterworth filter resembles an 
         ideal filter. 
      filterShape
         Options for the shape of the filter, specified in the constants.py 
         file in the ipcv directory. The different shapes will attenuate 
         the higher frequencies differently. 
            IDEAL
               Ideal shaped filter STRICTLY blocks ALL frequencies greater than
               the cutoffFrequency. Binary mask. 
            BUTTERWORTH
               Gaussian-like shaped filter that can be tuned based on the order
               parameter. 
            GAUSSIAN
               Gaussian-shaped filter. 
   returns::
      lowpass filter - numpy array with the same dimensions as the input image

   author::
      Victoria Scholl

   '''

   # get image dimensions, which dictate the filter dimensions
   imRows, imColumns, imBands, dataType = ipcv.dimensions(im)
   lowPassFilter = numpy.zeros([imRows,imColumns])

   # use ipcv.dist to generate an array with distances from the corner pixel.
   # roll twice to center the array (distances measured from center) 
   distFilter = ipcv.dist( ( imRows, imColumns ) )
   distFilter = numpy.roll(numpy.roll(distFilter, imRows/2, axis=0), 
                           imColumns/2, axis=1)


   if filterShape == ipcv.IPCV_IDEAL:
      # threshold dist array. distances < cutoff freq set to 1, else to 0. 
      lowPassFilter[ distFilter <= cutoffFrequency ] = 1
      lowPassFilter[ distFilter > cutoffFrequency ] = 0

   elif filterShape == ipcv.IPCV_BUTTERWORTH:  # butterworth equation
      lowPassFilter = 1 / ( 1 + (( distFilter/cutoffFrequency)**(2*order)))

   else: # Third filter type option is Gaussian
      lowPassFilter = numpy.exp(-1*(distFilter**2)/(2.0*cutoffFrequency**2))

   return lowPassFilter.astype(numpy.float64)



if __name__ == '__main__':

   import cv2
   import ipcv
   import numpy
   import matplotlib.pyplot
   import matplotlib.cm
   import mpl_toolkits.mplot3d

   filename = '/cis/faculty/cnspci/public_html/courses/common/images/lenna_color.tif'
   im = cv2.imread(filename)

   frequencyFilter = ipcv.filter_lowpass(im,
                                         16,
                                         filterShape=ipcv.IPCV_IDEAL)
   frequencyFilter = ipcv.filter_lowpass(im,
                                         16,
                                         order=1,
                                         filterShape=ipcv.IPCV_BUTTERWORTH)
   frequencyFilter = ipcv.filter_lowpass(im,
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
