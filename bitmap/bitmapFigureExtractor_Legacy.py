# Extracts bitmap images, but not all images (maybe because some graphs are in vector format?)

import fitz #Import PyMuPDF
import io
from PIL import Image


#Function to extract bitmap figures from a PDF (works for ALL bitmap filetypes)
#Input: filepath to the PDF of interest (as a string)
#       filepath to folder to save extracted bitmaps into (as a string)        
def convertPDFToBitmap(file, saveFolder):

    # open PDF
    pages = fitz.open(file)

    # iterate over PDF pages
    for page_index in range(len(pages)):
        # get the page
        page = pages[page_index]
        #Get all bitmap items from the page
        items = page.getImageList(full=True)
        #Get bounding box of each item
        for item in items:
            bbox = page.getImageBbox(item)
            print(f"Bounding Box(s) found on page {page_index} = {bbox}")

        image_list = page.getImageList()
        # Disp number of images found in this page
        if image_list:
            print(f"Found {len(image_list)} figure(s) on page {page_index}")
        else:
            print("[!] No images found on page", page_index)
        for image_index, img in enumerate(page.getImageList(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pages.extractImage(xref) #Returns a dict of image data eg size etc
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            # save it to image
            image.save(open(f"{saveFolder}/image{page_index+1}_{image_index}.{image_ext}", "wb"))


#ToDo: add a function to input a folder and iterate through all PDF's in that folder.

#Run the function
convertPDFToBitmap("/Users/Jordan/Documents/Imperial/ME4/FYP/Data_Extraction_Project/Test_PDFs/VibrationsAll.pdf", "/Users/Jordan/Documents/Imperial/ME4/FYP/Data_Extraction_Project/Python/bitmap/")
