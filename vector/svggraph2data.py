import xml.etree.ElementTree as ET
ET.register_namespace("","http://www.w3.org/2000/svg")#,"http://creativecommons.org/ns#","http://purl.org/dc/elements/1.1/","http://www.w3.org/1999/02/22-rdf-syntax-ns#")
import re
import sys
import os
import subprocess
import cairosvg
import pytesseract
from PIL import Image

from pytesseract import Output
from cv2 import cv2
from PyPDF2 import PdfFileWriter,PdfFileReader,PdfFileMerger
from pdf2image import convert_from_path
from statistics import mode, median

from math import log10, floor
from matplotlib import pyplot as plt
import pandas as pd

from timeit import default_timer as timer


####################################################
# 4 FILEPATH CHANGES NEEDED (Line 63, 167, 943)
####################################################



class svggraph2data:

        # Global Variables
        INPUT_FILEPATH = None
        OUTPUT_SVG_FILEPATH = None
        TREE = None #Element Tree
        ROOT = None #Element Tree Root
        AXES = None #Path ID's of Axes.
        AXESBB = None #Axes Bounding Boxes
        TEMP_PDF_Height = None # Height of temp pdf used for ocr
        TEMP_PDF_WIDTH = None # Width of temp pdf used for ocr
        EXTRACTION_TYPE = None # Final stage. How to process output. user inputted

        #TODO: Convert all try catchs to if numeric......
        # =====================================================================
        # PREPROCESSING
        # ===================================================================== 
        def createTree(self,filepath):
                tree = ET.parse(filepath)
                return tree

        # Function to read svg and convert all paths to absolute
        # Outputs: None. Stores updated paths in the ET tree
        def applyReltoAbs(self,tree):
                root = tree.getroot()

                # Iterates through every single path in SVG
                for p in root.iter('{http://www.w3.org/2000/svg}path'):
                        dPath = p.get('d')
                        #print(group.attrib['d'])
                        # Only convert to abs if relative m exists
                        if 'm' in dPath:
                                # Call the node js script #TODO: make file location more dynamic
                                js = subprocess.Popen(['/usr/local/bin/node', '', dPath], stdout=subprocess.PIPE)
                                out = js.stdout.read()
                                # TODO: handle errors from JS (error = p.stderr.read())
                                absPath = out.decode('utf-8') # Removes the b and /n
                                # Add spaces back into path
                                absPath = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", absPath)
                                # Removes blank space around negative signs
                                tempString = ''
                                for i,char in enumerate(absPath):
                                        if '-' in absPath[i]:
                                                tempString = tempString + ' -'
                                                continue
                                        else:
                                                tempString = tempString + char
                                absPath = tempString[1:]
                                # Round numbers in absPath to avoid erroneous cases
                                pathList = svggraph2data.parsePath(self,absPath)
                                for i,item in enumerate(pathList):f
                                        try:
                                                num = float(item)
                                                pathList[i] = round(num, 3)
                                        except:
                                                pathList[i] = item
                                absPath = svggraph2data.pathList2Path(self,pathList)
                                # Modify the path in the XML
                                p.attrib['d'] = absPath
                                
        def applyTranslationJS(self,tree):
                root = tree.getroot()

                # Placeholder
                innerTrans = []
                innerTransID = []
                parsedInnerTransID = []
                # Populate placeholder list
                for group in root.iter('{http://www.w3.org/2000/svg}g'):
                        innerTranslations = group.findall('{http://www.w3.org/2000/svg}g')
                        for trans in innerTranslations:
                                innerTransID.append(trans.attrib.get('id'))
                                innerTrans.append(trans.attrib.get('transform'))
                for id in innerTransID:
                        parsedInnerTransID.append(svggraph2data.parseChildID(self,str(id))) 
                # Remove the inner groups (inner groups are small values at the end of list)
                for i in range(len(parsedInnerTransID)-1):
                        if parsedInnerTransID[i] > parsedInnerTransID[i+1]:
                                parsedInnerTransID = parsedInnerTransID[:i]
                                break
                # Unique 'g' values sorted from low to high
                parsedInnerTransID.sort()
                parsedInnerTransID = list(dict.fromkeys(parsedInnerTransID))
                
                allTrans = []
                allTransID = []
                # Populate All list
                for group in root.iter():
                        # All
                        allTrans.append(group.get('transform'))
                        allTransID.append(group.get('id'))
                #arrayAll = list(zip(allTransID,allTrans)) # Creates 2d array 
                
                # Initialise list of transforms to apply to each element
                transformList = []

                # Iterates through every single path in SVG
                for group in root.iter():
                        id = group.get('id')

                        # Cycle through all children
                        for child in group:
                                path = child.get('d')
                                
                                if path is None:
                                        continue
                                # Find global matrix transform
                                matrixIdx = []
                                for i in range(len(allTrans)):
                                        if allTrans[i] is not None:
                                                if 'matrix' in allTrans[i]:
                                                        matrixIdx.append(allTransID[i])
                        
                                # Parse the ID
                                parsedID = svggraph2data.parseChildID(self,id)
                                # Skip any matrix transform
                                if parsedID is not None and id not in matrixIdx:
                                        for i in range(len(parsedInnerTransID)-1):
                                                # IF id is embedded
                                                if parsedID >= parsedInnerTransID[i] and parsedID < parsedInnerTransID[i+1]:
                                                        # Gives index of current id
                                                        idx = allTransID.index(id)
                                                        # If id is a top element, empty transform list and perform single transform
                                                        if parsedID == parsedInnerTransID[i]:
                                                                transformList = []
                                                                transformList.append(allTrans[idx])
                                                        # Else if is an inner transform, just append
                                                        else:
                                                                # Get transform based off of index and populate transformlist
                                                                transformList.append(allTrans[idx])
                                                                

                                        # Only run if transformList is not empty
                                        if len(transformList) != 0:
                                                # Apply the transforms in transformList to the path
                                                for transform in transformList:
                                                        if transform is not None:
                                                                js = subprocess.Popen(['/usr/local/bin/node', '', path, str(transform)], stdout=subprocess.PIPE)
                                                                out = js.stdout.read()
                                                                # TODO: handle errors from JS (error = p.stderr.read())
                                                                absPath = out.decode('utf-8') # Removes the b and /n
                                                                # Add spaces back into path
                                                                absPath = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", absPath)
                                                                # Removes blank space around negative signs
                                                                tempString = ''
                                                                for i,char in enumerate(absPath):
                                                                        if '-' in absPath[i]:
                                                                                tempString = tempString + ' -'
                                                                                continue
                                                                        else:
                                                                                tempString = tempString + char
                                                                absPath = tempString[1:]

                                                                # Cycle through all paths and compare pathid to group id to set path 'd' to new value
                                                                for p in root.iter('{http://www.w3.org/2000/svg}path'):
                                                                        pID = str(p.get('id'))
                                                                        try:
                                                                                int(pID[4:])
                                                                        except:
                                                                                continue
                                                                        if (int(pID[4:]) == int(parsedID) + 2):  # Assumes path is always +2 from group. CHECK THIS
                                                                                p.attrib['d'] = absPath
                                                                                continue
                                                        
                                                                # Update path variable for next iteration
                                                                path = absPath
                                                                # Remove translation
                                                                try:
                                                                        del group.attrib['transform']
                                                                except:
                                                                        continue
                # Remove clip paths from new SVG
                svggraph2data.removeClipPath(self,root)

        def removeClipPath(self,root):
                for group in root.iter('{http://www.w3.org/2000/svg}g'):
                        if group.get('clip-path') is not None:
                                del group.attrib['clip-path']

        def parseChildID(self,id):
                if 'g' in id:
                        parsedID = id.split('g')
                        return int(parsedID[1])
                return None

        def parseTranslation(self,translation):
                # Get x translation
                # If x and y
                translation = str(translation)
                if ',' in translation:
                        xstart = 'translate('
                        xend = ','
                        s = translation
                        translationx = s[s.find(xstart)+len(xstart):s.rfind(xend)]
                # If just x
                else:
                        xstart = 'translate('
                        xend = ')'
                        s = translation
                        translationx = s[s.find(xstart)+len(xstart):s.rfind(xend)]

                # Get y translation
                tranlsationy = 0
                if ',' in translation:
                        ystart = ','
                        yend = ')'
                        tranlsationy = s[s.find(ystart)+len(ystart):s.rfind(yend)]
                
                return translationx,tranlsationy

        # Creates list from a path string
        def parsePath(self,path):
                path = str(path)
                pathList = re.split(r'[,\s]\s*', path) # Split path by space and , delimiters
                # Deal with e values
                if 'e' in pathList:
                        i = pathList.index('e')
                        pathList[i-1 : i+2] = [''.join(pathList[i-1 : i+2])] 
                return pathList

        # Used to convert from list of paths to string of paths                        
        def pathList2Path(self,pathList):
                path = ''
                for items in pathList:
                        path = path + str(items) + " "
                return path

        # Checks if value is a float (returns true if is)
        def isfloat(self,value):
                try:
                        float(value)
                        return True
                except ValueError:
                        return False

        # ====================================================================
        # AXES FINDING
        # ====================================================================

        # Function to remove duplicates from 2 lists based on only 1 list
        def remove_duplicates(self, lst,lstID):
                res = []
                resID = []
                for i,x in enumerate(lst):
                        if x not in res:
                                res.append(x)
                                resID.append(lstID[i])
                return res, resID

        # Function to find horizontal and vertical lines of a random form 
        # Check length of all vertical lines vs horizontal lines in path and if ratio is large, path is x / y axis
        def findComplexAxes(self, dPaths, pathIDs):
                horizontalLines = []
                horizontalLineIDs =[]
                verticalLines = []
                verticalLineIDs =[]
                pgHeight = float(self.ROOT.attrib['height'])
                pgWidth = float(self.ROOT.attrib['width'])

                for i,dPath in enumerate(dPaths):
                        # Ignore all extremely large paths
                        if len(dPath) > 100:
                                continue
                        parsedPath = self.parsePath(dPath)
                        currentMx = 0 # Most recent M value we come across
                        currentMy = 0
                        lenH = 0 # Largest horizontal line in path
                        lenV = 0 # Largest vertical line in path
                        for j in range(len(parsedPath)-1):
                                if 'M' in parsedPath[j]:
                                        currentMx = float(parsedPath[j+1])
                                        currentMy = float(parsedPath[j+2])
                                # Get max length of horizontal line in path
                                if 'H' in parsedPath[j]:
                                        # If new horizontal line is largest so far, update lenH
                                        if (float(parsedPath[j+1]) - currentMx) > lenH:
                                                lenH = float(parsedPath[j+1]) - currentMx
                                # Get max length of vertical line in path
                                if 'V' in parsedPath[j]:
                                        # If new vertical line is largest so far, update lenV
                                        if (float(parsedPath[j+1]) - currentMy) > lenV:
                                                lenV = float(parsedPath[j+1]) - currentMy
                        
                        # Check ratio of vertical length to horiztonal. If > 100 class as y axis, if < 0.001 class as x axis.
                        # Check that the axes width / height requirements are satsified        
                        if (lenH > (pgWidth/7)) or (lenV > (pgHeight/15)):
                                # Deal with all simple cases when only H OR V in path
                                if lenV == 0:
                                        horizontalLines.append(dPaths[i])
                                        horizontalLineIDs.append(pathIDs[i])
                                        continue
                                if lenH == 0:
                                        verticalLines.append(dPaths[i])
                                        verticalLineIDs.append(pathIDs[i])
                                        continue
                                # Deal with complex case with both H and V
                                else:
                                        if (lenV/lenH) > 100:
                                                verticalLines.append(dPaths[i])
                                                verticalLineIDs.append(pathIDs[i])
                                        elif (lenH/lenV) > 100:
                                                horizontalLines.append(dPaths[i])
                                                horizontalLineIDs.append(pathIDs[i])
                
                # Remove any duplicates from axes list
                verticalLines, verticalLineIDs = self.remove_duplicates(verticalLines,verticalLineIDs)       
                horizontalLines, horizontalLineIDs = self.remove_duplicates(horizontalLines,horizontalLineIDs)  

                return horizontalLines, horizontalLineIDs, verticalLines, verticalLineIDs

        # Group by interestion point (axes groups)
        # TODO: Only works for expressions with 1 M
        def findItersects(self, horizontalLines, horizontalLineIDs, verticalLines, verticalLineIDs):
                axes = [] # 2d list of paths [horiz ID, verti ID]
                horizontalAxesVals = [] # COntains actual x axis path
                verticalAxesVals = [] # Contains actual y axis path
                for i,vLine in enumerate(verticalLines):
                        parsedvPath = self.parsePath(str(vLine))
                        idxV = parsedvPath.index('M')
                        vOrigin = parsedvPath[idxV+1:idxV+3]
                        for j,hLine in enumerate(horizontalLines):
                                parsedhPath = self.parsePath(str(hLine))
                                idxH = parsedhPath.index('M')
                                hOrigin = parsedhPath[idxH+1:idxH+3]

                                # Check that the verical lines and horizontal line origins are very close 1% difference allowed
                                if abs((float(vOrigin[0])/float(hOrigin[0]))-1) < 0.01 and abs((float(vOrigin[1])/float(hOrigin[1]))-1) < 0.01:
                                        axes.append([horizontalLineIDs[j], verticalLineIDs[i]])
                                        horizontalAxesVals.append(horizontalLines[j])
                                        verticalAxesVals.append(verticalLines[i])
                return axes, horizontalAxesVals, verticalAxesVals

        def findBoundingBox(self, axes, horizontalAxesVals, verticalAxesVals):
                axesBB = [] #[['startx', 'endx'], ['starty', 'endy']]
                # Temp lists
                axesBBx = []
                axesBBy = []

                # Cycle through horizontal lines
                for dPath in horizontalAxesVals:
                        parsedPath = self.parsePath(dPath)
                        currentMx = 0 # Most recent M value we come across
                        currentMy = 0
                        lenH = 0 # Largest horizontal line in path
                        lenV = 0 # Largest vertical line in path
                        for j in range(len(parsedPath)-1):
                                if 'M' in parsedPath[j]:
                                        currentMx = float(parsedPath[j+1])
                                        currentMy = float(parsedPath[j+2])
                                # Get max length of horizontal line in path
                                if 'H' in parsedPath[j]:
                                        # If new horizontal line is largest so far, update lenH and add start x and end x positions to list
                                        if (float(parsedPath[j+1]) - currentMx) > lenH:
                                                lenH = float(parsedPath[j+1]) - currentMx
                                                axesBBx.append([currentMx, parsedPath[j+1]])

                # Cylce through Vertcial lines
                for dPath in verticalAxesVals:
                        parsedPath = self.parsePath(dPath)
                        currentMx = 0 # Most recent M value we come across
                        currentMy = 0
                        lenH = 0 # Largest horizontal line in path
                        lenV = 0 # Largest vertical line in path
                        for j in range(len(parsedPath)-1):
                                if 'M' in parsedPath[j]:
                                        currentMx = float(parsedPath[j+1])
                                        currentMy = float(parsedPath[j+2])
                                # Get max length of vertical line in path
                                if 'V' in parsedPath[j]:
                                        # If new vertical line is largest so far, update lenV and add start y and end y positions to list
                                        if (float(parsedPath[j+1]) - currentMy) > lenV:
                                                lenV = float(parsedPath[j+1]) - currentMy
                                                axesBBy.append([currentMy, parsedPath[j+1]])

                # Combine the AxesBBx and AxesBBy
                axesBB = list(zip(axesBBx, axesBBy))

                return axesBB  

        def runAxesFinder(self):
                pathIDs = []
                dPaths = []
                for p in self.ROOT.iter('{http://www.w3.org/2000/svg}path'):
                        strPath = str(p.get('d'))
                        strPath = strPath.strip('\n') # Remove new line command
                        strPath = strPath.strip() # Remove whitespace before and after string
                        pathIDs.append(str(p.get('id')))
                        dPaths.append(strPath)

                horizontalLines, horizontalLineIDs, verticalLines, verticalLineIDs = self.findComplexAxes(dPaths, pathIDs)

                # Find axes combinations by grouping horixontal and vertical lines that iteresect
                axes, horizontalAxesVals, verticalAxesVals = self.findItersects(horizontalLines, horizontalLineIDs, verticalLines, verticalLineIDs)
                axesBB = self.findBoundingBox(axes, horizontalAxesVals, verticalAxesVals)
                print(str(len(axes))+' Axes found')
                return axes, axesBB

        # =========================================================================
        # FIND AXIS LABELS
        # =========================================================================

        # Converts output SVG to PDF
        def convertSVG(self):
                cairosvg.svg2pdf(url=self.OUTPUT_SVG_FILEPATH, write_to='temp.pdf')

        def cropPDF(self):
                # Open PDF
                pdf_file = PdfFileReader(open("temp.pdf","rb"))
                page = pdf_file.getPage(0)
                self.TEMP_PDF_HEIGHT = float(page.mediaBox.getHeight())
                self.TEMP_PDF_WIDTH = float(page.mediaBox.getWidth())

                # Holds the crop locations for each axes wrt entire page
                macroCrop = [] # ['axis', 'lowerleft','lowerright', 'upperleft', 'upperright']
                
                # For each axis, create a PDF containing the axes markers
                for i,axis in enumerate(self.AXESBB):
                        x0 = float(axis[0][0]) # Left coordinate of graph
                        x1 = float(axis[0][1]) # Right coordinate of graph
                        y0 = float(axis[1][0]) # Bottom coordinate of graph
                        y1 = float(axis[1][1]) # Top coordinate of graph
                        # X-axis
                        # Set the x-axis crop box wrt size of axis
                        xMacroLowerRight = (x1+((x1-x0)*0.05), y0-((y1-y0)*0.2))
                        xMacroLowerLeft = (x0-((x1-x0)*0.05), y0-((y1-y0)*0.2))
                        xMacroUpperRight = (x1+((x1-x0)*0.05), y0)
                        xMacroUpperLeft = (x0-((x1-x0)*0.05), y0)
                        macroCrop.append(['x', xMacroLowerLeft, xMacroLowerRight, xMacroUpperLeft, xMacroUpperRight])
                        page.mediaBox.lowerRight = xMacroLowerRight
                        page.mediaBox.lowerLeft = xMacroLowerLeft
                        page.mediaBox.upperRight = xMacroUpperRight
                        page.mediaBox.upperLeft = xMacroUpperLeft

                        # Write temp PDF and convert to png
                        outputx = PdfFileWriter()
                        outputx.addPage(page)
                        with open("x_graph%s.pdf" % i, "wb") as outputStream:
                                outputx.write(outputStream)
                        self.pdf2png('x_graph%s.pdf' % i, 'graph_%s_x.png' % "{0:0=2d}".format(i))
                        # Delete temp PDF when finished
                        os.remove("x_graph%s.pdf" % i)

                        # Y-axis
                        # Set the y-axis crop box wrt to size of axis
                        yMacroLowerRight = (x0-((x1-x0)*0.05), y0)
                        yMacroLowerLeft = (x0-((x1-x0)*0.2), y0-((y1-y0)*0.05))
                        yMacroUpperRight = (x0, y1+((y1-y0)*0.05))
                        yMacroUpperLeft = (x0-((x1-x0)*0.2), y1+((y1-y0)*0.05))
                        macroCrop.append(['y', yMacroLowerLeft, yMacroLowerRight, yMacroUpperLeft, yMacroUpperRight])
                        page.mediaBox.lowerRight = yMacroLowerRight
                        page.mediaBox.lowerLeft = yMacroLowerLeft
                        page.mediaBox.upperRight = yMacroUpperRight
                        page.mediaBox.upperLeft = yMacroUpperLeft

                        # Write temp PDF and convert to png
                        outputy = PdfFileWriter()
                        outputy.addPage(page)
                        with open("y_graph%s.pdf" % i, "wb") as outputStream:
                                outputy.write(outputStream)
                        self.pdf2png('y_graph%s.pdf' % i, 'graph_%s_y.png' % "{0:0=2d}".format(i))
                        # Delete temp PDF when finished
                        os.remove("y_graph%s.pdf" % i)
                
                return macroCrop

        def pdf2png(self, inFile, outFile):
                pages = convert_from_path(inFile, dpi=600)
                # Only expecting 1 page always
                for page in pages:
                        page.save(outFile, 'png')

        # Main function to get the axis labels. Returns OCR output
        def ocrPNG(self):
                # Config settings for tesseract OCR
                vertConfig = r'--psm 3, -c tessedit_char_whitelist=0123456789.-'
                horiConfig = r'--psm 6, -c tessedit_char_whitelist=0123456789.-'

                # Store all OCR outputs
                outOCR = []

                # Get all PNG files from directory of format ("graph_00_x.png")
                pngFiles = [] # stores all PNG files in dir
                for file in os.listdir():
                        if re.search("^(graph_)[0-9][0-9](_)(x|y)(.png)$", file):
                                pngFiles.append(file)
                # Sort to ensure consistent order
                pngFiles = sorted(pngFiles)

                for png in pngFiles:                
                        img = cv2.imread(png)
                        d = pytesseract.image_to_data(img, output_type=Output.DICT, config=horiConfig)

                        # # Use the below code to show OCR detection
                        # n_boxes = len(d['level'])
                        # for i in range(n_boxes):
                        #        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                        #        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # cv2.imshow('img', img)
                        # cv2.waitKey(0)

                        # Add identifier for x or y axis
                        if 'y' in png:
                                d['axis'] = 'y'
                        elif 'x' in png:
                                d['axis'] = 'x'
                        # Add file name to dict
                        d['pngFile'] = png

                        print(d['text'])
                        outOCR.append(d)

                return outOCR

        def mapping(self, outOCR, macroCrop):

                xMarkerTextList = []
                xMarkerPositionList = []
                yMarkerTextList = []
                yMarkerPositionList = []
                # For each x/y axis
                for j,d in enumerate(outOCR):
                        xMarkerTextList.append([])
                        xMarkerPositionList.append([])
                        yMarkerTextList.append([])
                        yMarkerPositionList.append([])
                        # For each marker in the axis
                        for i in range(len(d['text'])):
                                # Only consider numeric values
                                if self.isfloat(d['text'][i]) == True:
                                        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                                        # Find centre of marker (global centre -> micro coordinate + macro)
                                        # Find mapping between axis specific PNG and large PDF
                                        macroWidth = float(macroCrop[j][2][0]) - float(macroCrop[j][1][0]) # x lowerRight - lowerLeft 
                                        macroHeight = float(macroCrop[j][3][1]) - float(macroCrop[j][1][1]) # y UpperLeft - LowerLeft  ------PDF-> 0 at the top of page
                                        img = Image.open(str(d['pngFile']))
                                        # Width
                                        pngWidth, pngHeight = img.size
                                        pdfWidthMapping = macroWidth/pngWidth
                                        #Height
                                        pdfHeightMapping = macroHeight/pngHeight

                                        if 'x' in d['axis'] and 'x' in macroCrop[j][0]:
                                                xMarkerPositionList[j].append((float(x)*pdfWidthMapping)+((float(w)*pdfWidthMapping)/2) + float(macroCrop[j][1][0]))
                                                xMarkerTextList[j].append(d['text'][i])
                                        elif 'y' in d['axis'] and 'y' in macroCrop[j][0]:
                                                yMarkerPositionList[j].append((float(y)*pdfHeightMapping)+((float(h)*pdfHeightMapping)/2)+ (self.TEMP_PDF_HEIGHT - float(macroCrop[j][3][1])))
                                                yMarkerTextList[j].append(d['text'][i])
                
                # Remove blanks from lists of PDF points
                xMarkerTextList = [x for x in xMarkerTextList if x]
                xMarkerPositionList = [x for x in xMarkerPositionList if x]
                yMarkerTextList = [y for y in yMarkerTextList if y]
                yMarkerPositionList = [y for y in yMarkerPositionList if y]

                # Map PDF points to SVG coordinates
                # Linear mapping with no consideration for origin offset (newx = map * x)
                tempxPosList = []
                tempyPosList = []
                # PDF2SVGHeightMap = 1
                # PDF2SVGWidthMap = 1
                for i,xMarkers in enumerate(xMarkerPositionList):
                        tempxPosList.append([])
                        for xMarker in xMarkers:
                                tempxPosList[i].append(float(xMarker))
                for i,yMarkers in enumerate(yMarkerPositionList):
                        tempyPosList.append([])
                        for yMarker in yMarkers:
                                tempyPosList[i].append(float(self.TEMP_PDF_HEIGHT) - (float(yMarker))) # Convert from (0,0) top left to (0,0) bottom left
                xMarkerPositionList = tempxPosList
                yMarkerPositionList = tempyPosList

                # Map SVG cooridnates to Graph coordinates in a linear manner (gradient only here)
                xMapGradient, xMapIntercept, yMapGradient, yMapIntercept = self.createMap(xMarkerPositionList, xMarkerTextList, yMarkerPositionList, yMarkerTextList)
                
                return xMapGradient, xMapIntercept, yMapGradient, yMapIntercept 

        # Function to get the graph to SVG coordinate mapping
        # Checks for outliers
        # Returns a list of [a and b for x, a and b for y](size number of graphs)
        # TODO: UPDATE OUTLIER DETECTION USING BITMAP VERSION
        def createMap(self, xMarkerPositionList, xMarkerTextList, yMarkerPositionList, yMarkerTextList):
                # Returned lists
                xMapGradient = []
                xMapIntercept = []
                yMapGradient = []
                yMapIntercept = []

                #####################################################################
                # Determine x mapping
                #####################################################################
                macroMode = [] #Stores step size for each axis
                # Determine x mapping via mode outlier detection
                # Cycle through each graph
                for k, xTextMarkers in enumerate(xMarkerTextList):
                        xMarkerPosition = xMarkerPositionList[k]
                        macroMode.append([])
                        # Mode step sizes for each 'pivot'
                        microModeStep = []

                        #######################################################
                        # Calculate gradient of mapping (m part of linear eq)
                        for i,xTextMarker in enumerate(xTextMarkers):
                                microStep = []
                                for j in range(i+1,len(xTextMarkers)):
                                        microStep.append((float(xTextMarkers[j]) - float(xTextMarker))/(j-i))
                                # Find mode of step array
                                try:
                                        microModeStep.append(mode(microStep))
                                # Errors when no 'clear' mode
                                except ValueError:
                                        microModeStep.append(0)
                        # Calculate mode of all pivot passes. If no clear mode, set to 0 and assume unreadable
                        try:
                                macroMode[k].append(mode(microModeStep))
                        except:
                                macroMode[k].append(0)
                        # Check the steps != 0 -> Make sure we have confidence in OCR results
                        if macroMode[k][0] == 0:
                                print('ERROR: Graph %s x-axis cannot be interpretted correctly' % k)
                                xMapGradient.append(0)
                                xMapIntercept.append(0)
                                continue
                        else:
                                xMapGradient.append(self.calcMapGradient(microModeStep, xTextMarkers, xMarkerPosition))

                        ###################################################
                        # Calculate Intercept mapping (+c bit of linear eq)
                        step = macroMode[k][0]
                        if step == 0: # Check that axis is not undefined
                                xMapIntercept.append(0)
                        else:
                                # Easy method first. If 0 exists in marker list and is not erroneous, find it.
                                if '0' in xTextMarkers:
                                        indices = [i for i, x in enumerate(xTextMarkers) if x == '0']
                                        for idx in indices:
                                                if microModeStep[idx] != 0:
                                                        zeroPosSVG = xMarkerPosition[xTextMarkers.index('0')]
                                                        c = float(zeroPosSVG*xMapGradient[k]) # (xgraph = mxsvg + c) so c needs to be in graph coordinates
                                                        xMapIntercept.append(-1*c)
                                                # else:
                                                # xMapIntercept.append(0)  
                                else:
                                        # Harder method. Extrapolate to find 0 position
                                        # TODO: Test negative case
                                        idx = [i for i, x in enumerate(microModeStep) if x == step]
                                        refVal1 = max(float(xTextMarkers[idx[0]]), float(xTextMarkers[idx[1]])) # Get max value of 1st 2 none outlier vals
                                        refVal2 = min(float(xTextMarkers[idx[0]]), float(xTextMarkers[idx[1]]))  # Get min value of 1st 2 none outlier vals
                                        refValDiff = refVal1 - refVal2
                                        refValPosDiff = xMarkerPosition[xTextMarkers.index(refVal1)] - xMarkerPosition[xTextMarkers.index(refVal2)] # Position difference between max and min refVal
                                        stepPosDiff = refValPosDiff / (refValDiff / step) # Position difference of 1 step
                                        zeroPosSVG = xMarkerPosition[xTextMarkers.index(refVal1)] - ((refVal1/step)*stepPosDiff)
                                        c = float(zeroPosSVG*xMapGradient[k])
                                        xMapIntercept.append(-1*c)


                #####################################################################
                # Determine y mapping
                #####################################################################
                macroMode = [] #Stores step size for each axis
                # Cycle through each graph
                for k, yTextMarkers in enumerate(yMarkerTextList):
                        yMarkerPosition = yMarkerPositionList[k]
                        macroMode.append([])
                        # Mode step sizes for each 'pivot'
                        microModeStep = []
                        for i,yTextMarker in enumerate(yTextMarkers):
                                microStep = []
                                for j in range(i+1,len(yTextMarkers)):
                                        microStep.append((float(yTextMarkers[j]) - float(yTextMarker))/(j-i))
                                # Find mode of step array
                                try:
                                        microModeStep.append(mode(microStep))
                                # Errors when no 'clear' mode
                                except ValueError:
                                        microModeStep.append(0)

                        # Calculate mode of all pivot passes. If no clear mode, set to 0 and assume unreadable
                        try:
                                macroMode[k].append(mode(microModeStep))
                        except:
                                macroMode[k].append(0)
                        # Check the steps != 0 -> Make sure we have confidence in OCR results
                        if macroMode[k][0] == 0:
                                print('ERROR: Graph %s y-axis cannot be interpretted correctly' % k)
                                yMapGradient.append(0)
                                yMapIntercept.append(0)
                                continue
                        else:
                                yMapGradient.append(self.calcMapGradient(microModeStep, yTextMarkers, yMarkerPosition))

                        
                        ###################################################
                        # Calculate Intercept mapping (+c bit of linear eq)
                        step = macroMode[k][0]
                        if step == 0: # Check that axis is not undefined
                                yMapIntercept.append(0)
                        else:
                                # Easy method first. If 0 exists in marker list and is not erroneous, find it.
                                if '0' in yTextMarkers:
                                        indices = [i for i, y in enumerate(yTextMarkers) if y == '0']
                                        for idx in indices:
                                                if microModeStep[idx] != 0: # If 0 is not outlier or at the end of list (since cant check last item for outlier)
                                                        zeroPosSVG = yMarkerPosition[yTextMarkers.index('0')]
                                                        c = float(zeroPosSVG*yMapGradient[k]) # (ygraph = mysvg + c) so c needs to be in graph coordinates
                                                        yMapIntercept.append(-1*c)
                                                # else:
                                                #         yMapIntercept.append(0)
                                else:
                                        # Harder method. Extrapolate to find 0 position
                                        # TODO: Test negative case
                                        yTextMarkers = [float(n) for n in yTextMarkers] # Convert to float
                                        idx = [i for i, y in enumerate(microModeStep) if y == step]
                                        refVal1 = max(float(yTextMarkers[idx[0]]), float(yTextMarkers[idx[1]])) # Get max value of 1st 2 none outlier vals
                                        refVal2 = min(float(yTextMarkers[idx[0]]), float(yTextMarkers[idx[1]]))  # Get min value of 1st 2 none outlier vals
                                        refValDiff = refVal1 - refVal2
                                        refValPosDiff = yMarkerPosition[yTextMarkers.index(refVal1)] - yMarkerPosition[yTextMarkers.index(refVal2)] # Position difference between max and min refVal
                                        stepPosDiff = refValPosDiff / (refValDiff / step) # Position difference of 1 step
                                        zeroPosSVG = yMarkerPosition[yTextMarkers.index(refVal1)] - ((refVal1/step)*stepPosDiff)
                                        c = float(zeroPosSVG*yMapGradient[k])
                                        yMapIntercept.append(-1*c)
                
                return xMapGradient, xMapIntercept, yMapGradient, yMapIntercept 

        # calculates map
        # Returns mapGraident
        def calcMapGradient(self, microModeStep, textMarkers, markerPosition):
                mapGradient = None
                for i in range(len(textMarkers)-1):
                        for j in range(i+1,len(textMarkers)-1):
                                # If marker values are not erroneous
                                if microModeStep[i] != 0 and microModeStep[j] != 0 :
                                        mapGradient =  (float(textMarkers[j]) - float(textMarkers[i]))/(float(markerPosition[j]) - float(markerPosition[i]))
                                        return mapGradient # Break as soon as 1st non-erroneous value is found     
                return mapGradient # Return none is fails to calc gradient

        # Rounds inpit to 3 sig figs
        def round_sig(self, x, sig=3):
                return round(x, sig-int(floor(log10(abs(x))))-1)

        # Function to apply mapping to all relevent svg coordinates
        def applyMapping(self, xMapGradient, xMapIntercept, yMapGradient, yMapIntercept):
                # Return lists of x and y coordinates (wrt graph)
                x = []
                y = []
                # List of every x, y pair
                xFull = []
                yFull = []

                # Cycle through all graphs
                xmin = [] # Initialise bounding box lists
                xmax = []
                ymin = []
                ymax = []
                for axis in self.AXESBB:
                        # Get bounding box
                        xmin.append(float(axis[0][0])) # Left coordinate of graph
                        xmax.append(float(axis[0][1])) # Right coordinate of graph
                        ymin.append(float(axis[1][0])) # Bottom coordinate of graph
                        ymax.append(float(axis[1][1])) # Top coordinate of graph

                # Cycle through all paths
                # Get full list of x y coordinates in SVG files
                for p in self.ROOT.iter('{http://www.w3.org/2000/svg}path'):
                        dPath = p.get('d')
                        pathList = self.parsePath(dPath)
                        currentIdentifier = None # Type of path (M,C,V,H etc)

                        # Helper variables
                        count = 0

                        for k,item in enumerate(pathList):
                                # Check if string. If not, append to xfull / yfull
                                if self.isfloat(item):
                                        count = count + 1
                                        # x coord if count odd. y if count even
                                        if (count % 2) == 0 and currentIdentifier != 'H' and currentIdentifier != 'V':
                                                yFull.append(float(item))
                                                xFull.append(float(pathList[k-1]))
                                else:
                                        currentIdentifier = item
                                        count = 0

                # Remove duplicates from xFull and yFull
                combFull = list(zip(xFull, yFull))
                combFull = list(set(combFull))
                xFull, yFull = zip(*combFull)

                # Final converted coordinates
                xOut = []
                xMedian = []
                yOut = []
                yMedian = []
                # Check if items in xFull and yFull are in a BB
                # Apply mapping
                for i in range(len(xmin)):
                        xOut.append([])
                        xMedian.append([]) # List with no duplicates or outliers
                        yOut.append([])
                        yMedian.append([])
                        if xMapGradient[i] != 0 and xMapIntercept[i] != 0 and yMapGradient[i] != 0 and yMapIntercept[i] != 0: # Check for erroneous axis
                                for j in range(len(xFull)):
                                        # x,y is in a BB
                                        if xFull[j] > xmin[i] and xFull[j] < xmax[i] and yFull[j] > ymin[i] and yFull[j] < ymax[i]:
                                                # Apply Mapping
                                                xOut[i].append((xFull[j]*xMapGradient[i])+xMapIntercept[i])
                                                yOut[i].append((yFull[j]*yMapGradient[i])+yMapIntercept[i])
                                # plt.plot(xOut[i], yOut[i], 'o')
                                # plt.show()
                                # Remove duplicate coordinates to 3 sig figs
                                # Round all to 3 sig figs
                                for k in range(len(xOut[i])):
                                        xOut[i][k] = self.round_sig(xOut[i][k])
                                        yOut[i][k] = self.round_sig(yOut[i][k])
                                # Remove duplicates
                                combOut = list(zip(xOut[i], yOut[i]))
                                combOut = list(set(combOut))
                                xOut[i], yOut[i] = zip(*combOut)
                                #######################################
                                # MEDIAN OPTION
                                # Get median y value for equal x values (Remove outliers and thinkness of line)
                                xSet = set(xOut[i])
                                for uniqueX in xSet:
                                        indexes = [n for n,x in enumerate(xOut[i]) if x==uniqueX]
                                        tempyVals = [yOut[i][idx] for idx in indexes]
                                        newy = median(tempyVals)
                                        xMedian[i].append(uniqueX)
                                        yMedian[i].append(newy)
                                ########################################
                                # Convert to CSV with user required post pro
                                outputCSVPath = str(self.INPUT_FILEPATH[:-8] + '_graph_%s.csv' % i)
                                if self.EXTRACTION_TYPE == 'median':
                                        self.export2CSV(xMedian[i], yMedian[i], outputCSVPath)
                                elif self.EXTRACTION_TYPE == 'raw':
                                        self.export2CSV(xOut[i], yOut[i], outputCSVPath)
                                

                print('complete')

        def export2CSV(self, x, y, outputFilepath):
                # pandas can convert a list of lists to a dataframe.
                # each list is a row thus after constructing the dataframe
                # transpose is applied to get to the user's desired output. 
                df = pd.DataFrame([x,y])
                df = df.transpose() 
                df.columns = ['x', 'y']
                # Sort by x values
                df = df.sort_values(by=['x'])
                df.to_csv(outputFilepath, sep=',', header=['x', 'y'], index=False)

        # Delete PNG files created
        def cleanDir(self):
                directory = os.getcwd() # get current directory
                files_in_directory = os.listdir(directory)
                filtered_files = [file for file in files_in_directory if file.endswith(".png")]
                for file in filtered_files:
                        path_to_file = os.path.join(directory, file)
                        os.remove(path_to_file)
                                                
                                        


        def __init__ (self, filepath, extractionType='raw'):
                # Declare input and output filepaths 
                entireStart = timer()
                self.EXTRACTION_TYPE = extractionType
                self.INPUT_FILEPATH = filepath
                self.OUTPUT_SVG_FILEPATH = str(os.path.dirname(os.path.abspath(filepath))) + '/out_' + self.INPUT_FILEPATH[-15:]

                # PREPROCESSING
                self.TREE = svggraph2data.createTree(self,filepath)
                print('created Tree')
                start = timer()
                svggraph2data.applyReltoAbs(self,self.TREE)
                end = timer()
                print('converted all paths to absolute in: ' + str(end - start))
                start = timer()
                svggraph2data.applyTranslationJS(self, self.TREE)
                end = timer()
                print('transformed all coordinates in: ' + str(end-start))
                self.TREE.write(self.OUTPUT_SVG_FILEPATH)
                print('New svg created')

                # AXIS FINDING
                start = timer()
                newSVGFilepath = filepath[:-15] + 'out_' + filepath[-15:]
                self.TREE = self.createTree(newSVGFilepath)# Update tree to new SVG
                self.ROOT = self.TREE.getroot()
                self.AXES, self.AXESBB = self.runAxesFinder()
                end = timer()
                print('Completed Axis finding function in: ' + str(end - start))

                # AXIS LABELS
                start = timer()
                self.convertSVG() # Converts SVG to temp.pdf
                macroCrop = self.cropPDF() # Crop PDF to get axes.png
                outOCR = self.ocrPNG() # OCR axes.png
                end = timer()
                print('Completed axis label determination function in: ' + str(end - start))

                # MAPPING
                start = timer()
                xMapGradient, xMapIntercept, yMapGradient, yMapIntercept = self.mapping(outOCR, macroCrop)
                self.applyMapping(xMapGradient, xMapIntercept, yMapGradient, yMapIntercept)
                end = timer()
                print('Completed mapping in: ' + str(end - start))

                # Clean up directory
                self.cleanDir()

                entireEnd = timer()
                print('Finished in: ' + str(entireEnd - entireStart) + ' seconds')

svggraph2data('pg_0007.pdf.svg', 'raw')
