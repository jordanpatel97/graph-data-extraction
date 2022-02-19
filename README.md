# Extracting data points from graphs in scientific research papers

Computer vision project where both bitmap and vectorised images are analysed to locate graphs and then digitise the data points.

Involves the use of object detection (EfficientDet) and image segementation.

![Brochure_Image](https://user-images.githubusercontent.com/74137905/154816233-4f2b8d7a-16d8-4f12-81dc-bd4af784533f.png)


## Project Summary:

“Create a tool capable of extracting data points from graphs in PDF research papers.”

The project was split into 2 sections. 1 focussing on manipulating vector images and the other on bitmap images. This 2 tool approach was required as graphs can be embedded in PDFs in both formats.
The project was limited to scatter and line graphs only.

A 6 step approach was adopted to create both tools:

	1.	Identify graphs in PDFs 
	⁃	Bitmap -> Used an EfficientDet D0 object detection algorithm to identify graphs from PNG files of individual PDF pages.
	⁃	Vector -> Converted entire PDF pages to SVG format using Inkscape. Required a large pre-processing step to be able to process and effectively use the outputted SVG. Did not specifically identify graphs, just used step 2 and assumed if axes exist, a graph must exist. 
	2.	Identify graph axes
	⁃	Bitmap -> Used OpenCV’s Hough line transform and Canny edge detection to find all straight lines in graph. Then use logic based on intersection points and line orientations to find graph axes.
	⁃	Vector -> Used simple logic looking for intersecting horizontal and vertical paths defined in the SVG file. 
	3.	Identify graph properties (e.g. axis markers and labels)
	⁃	Bitmap -> Use optical character recognition (Tesseract OCR)
	⁃	Vector -> Convert SVG to PNG and use OCR
	4.	Identify graph lines to extract
	⁃	Bitmap -> Use OpenCV’s contour detection and masking features to isolate line colours. Use EfficientDet D0 object detection algorithm to identify legends with graph and use OCR to read legend labels.
	⁃	Vector -> Simply look for all paths contained within the graph axes (No line-colour segmentation or legend interpretation).
	5.	Create an image to graph coordinate mapping
	⁃	Bitmap -> Use the axis marker text and position found in Step 3 to create a linear pixel to graph coordinate mapping.
	⁃	Vector -> Use the axis marker text and position found in Step 3 to create a linear SVG to graph coordinate mapping.
	6.	Export data points to CSV
	⁃	Bitmap -> Use an averaging window to reduce noise in outputted data points. Then export to CSV with legend label in file name and x and y axes labels as column headers in the CSV
	⁃	Vector -> Export data points to CSV with x and y axes labels as column headers in the CSV

The bitmap tool was 5x faster than the vector tool and had fewer operational failures.

Manipulating bitmap images was concluded as the best method of the 2. Mainly due to the ability to use common object detection algorithms with bitmap images which allowed both graphs embedded in PDFs in both bitmap and vector form to be extracted.
