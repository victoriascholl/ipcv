import ipcv
import numpy

def filter_bandreject(im, radialCenter, bandwidth, order=1, filterShape=ipcv.IPCV_IDEAL):

   '''
  title::
      filter_bandreject

   description::
      This method creates a bandreject filter to be applied to the 
      centered fourier transform of the input image.
      A mask is created to preserve all frequencies except
      for those restricted to a range defined by the radial center 
      plus the bandwidth. 
      First, the dimensions of the input image are determined.
      Then an array of zeros is created with the same dimensions to serve 
      as the bandreject filter. 
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
      radialCenter
        Distance (integer) from the center of the fourier transform at which
        the frequency range begins
      bandwidth
        Thickness of the band of frequencies to be rejected, measured
        outward from the radialCenter distance.  
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
               specified range. Binary mask. 
            BUTTERWORTH
               Gaussian-like shaped filter that can be tuned based on the order
               parameter. 
            GAUSSIAN
               Gaussian-shaped filter. 
   returns::
      bandreject filter - numpy array with the same dimensions as the input image
      that will preserve all frequencies outside of the specified range. 

   author::
      Victoria Scholl
   '''

   # get image dimensions, which dictate the filter dimensions
   imRows, imColumns, imBands, dataType = ipcv.dimensions(im)
   bandRejectFilter = numpy.zeros([imRows,imColumns])

   # use ipcv.dist to generate an array with distances from the corner pixel.
   # roll twice to center the array (distances measured from center) 
   distFilter = ipcv.dist( ( imRows, imColumns ) )
   distFilter = numpy.roll(numpy.roll(distFilter, imRows/2, axis=0), imColumns/2, axis=1)

   if filterShape == ipcv.IPCV_IDEAL:
      # threshold dist array. distances outside (radialCenter, radialCenter
      # plus bandwidth) set to 1, else 0. 
      bandRejectFilter[ distFilter < radialCenter ] = 1
      bandRejectFilter[ distFilter > radialCenter + bandwidth ] = 1

   elif filterShape == ipcv.IPCV_BUTTERWORTH:  # butterworth equation
      bandRejectFilter = 1 / ( 1 + (( distFilter * bandwidth / ( distFilter ** 2 - radialCenter **2)   ) ** 2 * order)) 

   else: # Third filter type option is Gaussian
      bandRejectFilter = 1 - numpy.exp( -0.5 * ( ( distFilter ** 2 - radialCenter **2 ) / (distFilter * bandwidth)) ** 2 )

   
   return bandRejectFilter.astype(numpy.float64)



if __name__ == '__main__':

   import cv2
   import ipcv
   import numpy
   import matplotlib.pyplot
   import matplotlib.cm
   import mpl_toolkits.mplot3d

   filename = '/cis/faculty/cnspci/public_html/courses/common/images/lenna_color.tif'
   im = cv2.imread(filename)

   frequencyFilter = ipcv.filter_bandreject(im,
                                         32,
                                         15,
                                         filterShape=ipcv.IPCV_IDEAL)
   frequencyFilter = ipcv.filter_bandreject(im,
                                         32,
                                         15,
                                         order=1,
                                         filterShape=ipcv.IPCV_BUTTERWORTH)
   frequencyFilter = ipcv.filter_bandreject(im,
                                         32,
                                         15,
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
