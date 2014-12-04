import ipcv
import numpy

def frequency_filter(im, frequencyFilter, delta=0):
   '''
   title::
      frequency_filter

   description::
      This method applies a frequency filter to the Fourier transform
      of the specified input image. 
      First, the image dimensions are determined. The 2D inverse Fourier
      Transform is taken of the input image, which is then shifted to
      center the zero-frequency component to the center of the array.
      The transform is then multiplied by the frequency filter.
      The result is shifted back and the 2D inverse Fourier transform is 
      taken. These steps are performed for each band in the input image.
      The filtered image is returned as type numpy.complex128. 

   attributes::
      im
         Input source image, of type numpy array
      frequencyFilter
         Mask to be applied to the input image to filter out some
         frequencies and preserve others
      delta
         Bias value added to the filtered image with a default value of 0.          

   returns::
      Filtered image array of type numpy.complex128

   author::
      Victoria Scholl
   '''

   # Find the image dimensions
   imRows, imColumns, imBands, dataType = ipcv.dimensions(im) 
   filteredImage = numpy.zeros((imRows,imColumns,imBands))
   
   for band in range(imBands): 
      imTransform = numpy.fft.fft2(im[:,:,band])
      transformCentered = numpy.fft.fftshift(imTransform)
      filteredTransform = transformCentered * frequencyFilter
      filteredUncentered = numpy.fft.ifftshift(filteredTransform)
      filteredImage[:,:,band] = numpy.fft.ifft2(filteredUncentered)
   
   filteredImage += delta

   return filteredImage.astype(numpy.complex128)

if __name__ == '__main__':

   import cv2
   import ipcv
   import numpy
   import time

   #filename = '/cis/faculty/cnspci/public_html/courses/common/images/lenna_color.tif'
   #filename = 'lena_periodic.jpg'
   filename = '/cis/faculty/cnspci/public_html/courses/common/images/giza.jpg'
   im = cv2.imread(filename)

   frequencyFilter = ipcv.filter_lowpass(im, 
                                         32,2, 
                                         filterShape=ipcv.IPCV_GAUSSIAN)


   #notchCenter = [(32,0)]
   #notchRadius = [3]
   #frequencyFilter = ipcv.filter_notchreject(im,
   #                                          notchCenter,
   #                                          notchRadius,
   #                                          order=2,
   #                                          filterShape=ipcv.IPCV_GAUSSIAN)
   
   startTime = time.clock()
   offset = 0
   filteredImage = ipcv.frequency_filter(im, frequencyFilter, delta=offset)
   filteredImage = numpy.abs(filteredImage)
   filteredImage = filteredImage.astype(dtype=numpy.uint8)
   
   # Clipping the filtered image assuming 8-bit 
   filteredImage[ filteredImage > 255] = 255
   elapsedTime = time.clock() - startTime
   print 'Elapsed time (frequency_filter) = %s [s]' % elapsedTime

   cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
   cv2.imshow(filename, im)
   cv2.imshow(filename, ipcv.histogram_enhancement(im))

   filterName = 'Filtered (' + filename + ')'
   cv2.namedWindow(filterName, cv2.WINDOW_AUTOSIZE)
   cv2.imshow(filterName, filteredImage)
   cv2.imshow(filterName, ipcv.histogram_enhancement(filteredImage))

   ipcv.flush()
