import fitz
import os
import glob
from graphDetector import graphDetector
import numpy as np
from PyPDF2 import PdfFileWriter,PdfFileReader
from pdf2image import convert_from_path


# FUNCTION TO extract graph PNGs from input PDF files
def pdf2graphs(inputFolderPath):
    '''Function to identify, isolate and extract graphs from PDFs and save 
    as PNG files. 
    
    Input:
    
    Filepath of folder containing all PDF's to be analysed.
    
    Output:
    
    PNG files of extracted graphs saved into the input folder path'''


    # Initialise OD model
    gDetector =  graphDetector('/inference_graph/saved_model') 

    for pdf_path in glob.glob(inputFolderPath+'/*.pdf'):
        doc = fitz.open(pdf_path)
        pageCount = doc.pageCount
        for pageNum in range(0,pageCount):
            page = doc.loadPage(pageNum) 
            pix = page.getPixmap()
            pdfName = os.path.split(pdf_path)
            output = inputFolderPath + '/' + pdfName[1][:-4] + '_Pg' + str(pageNum).zfill(4) + '.png' # Output stored in given folder with name e.g. [pdfname]_pg0001.png
            # Write png of full pdf page
            pix.writePNG(output)
            
            # Run OD on produced png
            output_dict = gDetector.runModel(output)
            idx = np.where(output_dict['detection_scores']>0.5) # Get index of all detected legends (i.e probabiliy>50%)
            boundingBoxes = output_dict['detection_boxes'][idx] # Get bounding box
            pageSize = page.MediaBox # Get size of pages png 
            for i, boundingBox in enumerate(boundingBoxes):
                y_min = round(boundingBox[0] * pageSize[3]) # Convert from relative coordinates to absolute
                x_min = round(boundingBox[1] * pageSize[2])
                y_max = round(boundingBox[2] * pageSize[3])
                x_max = round(boundingBox[3] * pageSize[2])
                cropWidth = x_max - x_min
                cropHeight = y_max - y_min
                # Crop PDF around the identified image
                pdf_file = PdfFileReader(open(pdf_path,"rb")) # TODO: Messy. Try to use just pymupdf or just pypdf2
                page_write = pdf_file.getPage(pageNum)
                page_write.mediaBox.lowerRight = (round(x_max+(cropWidth*0.05)), round(pageSize[3]) - round(y_min-(cropHeight*0.05)))
                page_write.mediaBox.lowerLeft = (round(x_min-(cropWidth*0.05)), round(pageSize[3]) - round(y_min-(cropHeight*0.05)))
                page_write.mediaBox.upperRight = (round(x_max+(cropWidth*0.05)), round(pageSize[3]) - round(y_max+(cropHeight*0.05)))
                page_write.mediaBox.upperLeft = (round(x_min-(cropWidth*0.05)), round(pageSize[3]) - round(y_max+(cropHeight*0.05)))
                
                pdfWriter = PdfFileWriter()
                pdfWriter.addPage(page_write)
                with open(pdf_path[:-4] + 'temp.pdf', "wb") as outputStream:
                    pdfWriter.write(outputStream)

                # Save new cropped PDF as high res PNG
                tempPages = convert_from_path(pdf_path[:-4] + 'temp.pdf', dpi=600)
                for tempPage in tempPages:
                    tempPage.save(output[:-4] + '_graph' + str(i).zfill(2) + '.png', 'png')

                # Delete temp pdf
                os.remove(pdf_path[:-4] + 'temp.pdf')

                # Reset Cropbox for next iteration
                page_write.mediaBox.lowerRight = (pageSize[2], 0)
                page_write.mediaBox.lowerLeft = (0, 0)
                page_write.mediaBox.upperRight = (pageSize[2], pageSize[3])
                page_write.mediaBox.upperLeft = (0, pageSize[3])
            
            # Delete full pdf page PNG
            os.remove(output)


# Run Function
pdf2graphs('INPUT FILEPATH OF FOLDER HERE')
