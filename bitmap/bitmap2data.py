from os.path import abspath, splitext
from PIL.Image import ROTATE_90
from cv2 import cv2 as cv
import numpy as np
import pandas as pd
from math import log10, floor
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
from statistics import mode
import os
from legendDetector import legendDetector
import glob
from timeit import default_timer as timer


###########################################
# 2 INPUTS REQUIRED - SEE BOTTOM OF SCRIPT (Line 920, 924)
###########################################

# Reads image, greysclaes it and finds edges
def preprocess(filepath):
    # Read in image
    img = cv.imread(filepath)

    # Greyscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Run canny edge detection
    edges = cv.Canny(gray,125,175,apertureSize = 3)
    #cv.imshow('edges',edges)
    #cv.waitKey(0)

    return img, gray, edges

# Applies the hough line transform.
# Outputs a list of found lines
def findStraightLines(gray, edges):
    # Set min line length to half of smallest side of image
    minLineLength = min(edges.shape) // 1.5 # Get min size of image
    thresholdVal = min(edges.shape[0], edges.shape[1]) // 2

    # Apply hough line transform
    lines = cv.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=thresholdVal, lines=np.array([]), minLineLength=minLineLength,maxLineGap=25)
    
    # Display detected lines
    if lines is None:
        print('Error')
    else:
        gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # Convert back to BGR to add coloured lines
        a,b,c = lines.shape
        for i in range(a):
            cv.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 1, cv.LINE_AA)
        #cv.imshow('hough lines', gray)
        #cv.waitKey(0)
    
    return lines


def findLineOrientation(lines):
    horizontalLines = []
    verticalLines = []
    for line in lines:
        startx = line[0][0]
        starty = line[0][1]
        endx = line[0][2]
        endy = line[0][3]
        
        # check aspect ratio of line and add line to corresponding list
        if abs((startx/endx)-1) < 0.01:
            verticalLines.append(line)
        elif abs((starty/endy)-1) < 0.01:
            horizontalLines.append(line)  
    
    return horizontalLines, verticalLines


# Checks if tuple is close to other tuples (< 10 pixels counts as close)
def isClose(ind, all, limit=10):
    for item in all:
        if abs(item[0] - ind[0]) < limit and abs(item[1] - ind[1]) < limit:
            return True
    return False

# Finds intersects between lines
def findIntersects(horizontalLines, verticalLines):
    # Gets line equation ax + by = c
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    # Checks for intersection
    # Returns intersection point if exist
    def intersection(L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        else:
            return False

    intersects = [] # List of intersects
    intersectingLines = [] #  horizontal and vertical lines intersecting ([axis][start x, start y, end x, end y])
    for hLine in horizontalLines:
        for vLine in verticalLines:
            L1 = line([hLine[0][0], hLine[0][1]], [hLine[0][2], hLine[0][3]])
            L2 = line([vLine[0][0], vLine[0][1]], [vLine[0][2], vLine[0][3]])
            R = intersection(L1, L2)
            if R:
                #print(R)
                if not isClose(R, intersects):
                    intersects.append(R)
                    intersectingLines.append([[hLine[0][0], hLine[0][1], hLine[0][2], hLine[0][3]], [vLine[0][0], vLine[0][1], vLine[0][2], vLine[0][3]]])
            else:
                print('No intersects')

    return intersectingLines, intersects

# Crop images around intersecting lines (i.e. potential axes)
# Output cropped images for OCR input
def cropImg(img, intersectingLines):
    xCrop = [] # List of cropped 'x axes'
    yCrop = [] # List of cropped 'y axes'
    for i, linePair in enumerate(intersectingLines):
        # Check aspect ratio
        # x-axis
        xStartCrop = min(linePair[0][0], linePair[0][2])
        xEndCrop = max(linePair[0][2], linePair[0][0])
        xHeightCrop = abs(linePair[1][1] - linePair[1][3])//10 # crop to 10% of height of vert line
        cropBuffer = 20 # TODO: Feed this into the getAxes() micro scale to macro scale conversion (currently hardcoded)
        xCrop.append(img[max(linePair[0][1], 0):min(linePair[0][1]+xHeightCrop, img.shape[0]), max(xStartCrop-cropBuffer, 0):min(xEndCrop+cropBuffer, img.shape[1])]) # TODO: Currently assumes markers are below / left of axis.
        #cv.imshow('X Cropped Image', xCrop[i])
        #cv.waitKey(0)

        # y-axis
        yStartCrop = min(linePair[1][1], linePair[1][3])
        yEndCrop = max(linePair[1][3], linePair[1][1])
        yWidthCrop = abs(linePair[0][0] - linePair[0][2])//10 # crop to 10% of width of hori line
        yCrop.append(img[max(yStartCrop-cropBuffer, 0):min(yEndCrop+cropBuffer, img.shape[0]), max(linePair[1][0]-yWidthCrop, 0):min(linePair[1][0], img.shape[1])])
        #cv.imshow('Y Cropped Image', yCrop[i])
        #cv.waitKey(0)
    return xCrop, yCrop


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError: 
        return False

# Main function to get image OCR'd. 
# Input image to be OCR'd and type to determine if axis label or marker recognition
# Returns cleaned OCR output
def ocrImg(img, type):
    textList = []
    textPositionList = [] # Stores centre point of text (x,y) in micro reference frame

    # Config settings for tesseract OCR. If type==markers only want digits. else everything
    if type == 'marker':
        config = r'--psm 6 -c tessedit_char_whitelist=0123456789.-'
    else:
        config = r'--psm 6'

    # Perform OCR
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
    # # UNCOMMENT BELOW FOR VISUAL DEBUG!!!
    # # Use the below code to show OCR detection
    # n_boxes = len(d['level'])
    # for i in range(n_boxes):
    #        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv.imshow('img', img)
    # cv.waitKey(0)
    # print(d['text'])

    # Clean up output
    # Remove blanks from lists of PDF points
    for i in range(len(d['text'])):
        textList.append(d['text'][i])
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #textPositionList.append([round(x+0.5*w), round(y+0.5*h)])
        textPositionList.append([x,y,w,h])

    # Remove empty values
    idx = [i for i,x in enumerate(textList) if x]
    textList = [textList[index] for index in idx]
    textPositionList = [textPositionList[index] for index in idx]
    #textList = [x for x in textList if x]
    return textList, textPositionList
            
# Checks if crop pairs are axes via ocr output. if both x and y pairs have meaningful OCR outputs -> they are axes
# Outputs OCR text output and lines defining the x and y of axes 
def getAxes(xCrop, yCrop, intersectingLines, img):
    '''Function to check if interesecting lines have readable 
        (via OCR) axis markers. Uses full image to get x and y axis labels.
         Returns a list of axes marker text values, text positions
            bounding boxes for the axes and x y labels'''

    axesBB = [] # Identified axes bounding box (x and y lines defined)
    axesMarkers = [] # Axes Text tuple ([x],[y])
    axesMarkersPos = [] # Axes text position turple (in global reference frame)
    yLabels = [] # List of Y-axis lables (list of strings)
    xLabels = [] # List of X-axis lables (list of strings)
    for i in range(len(xCrop)):
        xOCRText, xOCRPosition = ocrImg(xCrop[i], type='marker')
        yOCRText, yOCRPosition = ocrImg(yCrop[i], type='marker')
        # If more than 2 markers on both axes, count as axes set
        if len(xOCRText) > 2 and len(yOCRText) > 2:
            ################################## FIND AXIS MARKERS ####################################
            # convert OCR position to global reference frame (only converts required coordinate (i.e. x axis will only convert x position))
            cropBuffer = 20 # TODO: Remove hardcoding
            xStartCrop = max(min(intersectingLines[i][0][0], intersectingLines[i][0][2]) - cropBuffer, 0) # 0 at left. 
            yStartCrop = max(min(intersectingLines[i][1][1], intersectingLines[i][1][3]) - cropBuffer, 0) # 0 at the top

            # Perform iterative crop to maximise padding (and thus OCR accuracy)
            rightIdentified = 0 # Initialise y-axis variables
            leftIdentified = 0
            totalLeftChange = 0 # Variable that tracks the change in the left coord of yCrop
            # Y-axis (x squeeze) #############################################################
            for iterNo in range(5): # Limit of 5 iterations
                # Get 'updated' crop coordinates
                leftCoord = [] # Temp lists
                rightCoord = []
                for yOCRPos in yOCRPosition:
                    leftCoord.append(yOCRPos[0])
                    rightCoord.append(yOCRPos[0] + yOCRPos[2]) # x + width
                leftIdentified = min(leftCoord)
                rightIdentified = max(rightCoord)

                if yCrop[i].shape[1] > rightIdentified - leftIdentified: # Test if current cropped image is larger than OCR'd limits
                    print('Squeezing X-coordinates around y-axis markers')
                    totalLeftChange = totalLeftChange + leftIdentified # Update total change tracker
                    newyCrop = yCrop[i][:, leftIdentified:rightIdentified]
                    # cv.imshow('new y crop', newyCrop)
                    # cv.imshow('old y crop', yCrop[i])
                    # cv.waitKey(0)
                    yCrop[i] = newyCrop # Update yCrop

                    # Perform OCR on new cropped image
                    yOCRText, yOCRPosition = ocrImg(yCrop[i], type='marker')
                else:
                    break # Exit statement

            # TODO:THE X_AXIS SQUEEZE IS NOT NORMALLY NEEDED. LEAVE OUT FOR NOW UNTIL MORE STABLE
            # X-axis (y squeeze) #############################################################
            bottomIdentified = 0 # Initialise x-axis variables
            topIdentified = 0 
            totalBottomChange = 0 # Variable that tracks the change in the bottom coord of xCrop
            for iterNo in range(5): # Limit of 5 iterations
                # Get 'updated' crop coordinates
                topCoord = [] # Temp lists
                bottomCoord = []
                for xOCRPos in xOCRPosition:
                    topCoord.append(xOCRPos[1])
                    bottomCoord.append(xOCRPos[1] + xOCRPos[3]) # y + height
                topIdentified = min(topCoord)
                bottomIdentified = max(bottomCoord)

                if xCrop[i].shape[0] > bottomIdentified - topIdentified: # Test if current cropped image is larger than OCR'd limits
                    print('Squeezing Y-coordinates around X-axis markers')
                    totalBottomChange = totalBottomChange + (xCrop[i].shape[0] - bottomIdentified) # Update total change tracker
                    newxCrop = xCrop[i][topIdentified:bottomIdentified, :]
                    # cv.imshow('new x crop', newxCrop)
                    # cv.imshow('old x crop', xCrop[i])
                    # cv.waitKey(0)
                    xCrop[i] = newxCrop # Update xCrop

                    # Perform OCR on new cropped image
                    xOCRText, xOCRPosition = ocrImg(xCrop[i], type='marker')
                else:
                    break # Exit statement



            # centre of OCR position = round(x+0.5*w) OR round(y+0.5*h)
            # cropped region + OCR location in crop reference frame = global reference frame
            for j,pos in enumerate(xOCRPosition):
                xOCRPosition[j][0]= xStartCrop + round(pos[0]+0.5*pos[2]) 
            for k,pos in enumerate(yOCRPosition):
                yOCRPosition[k][1] = yStartCrop + round(pos[1]+0.5*pos[3])  

            axesMarkers.append([xOCRText,yOCRText])
            axesMarkersPos.append([xOCRPosition, yOCRPosition]) # convert position to global reference frame
            axesBB.append(intersectingLines[i])
            print('Axis found!')


            ################################## FIND AXIS LABELS ####################################
            # Find Y axis label location
            yAxisStarty = yStartCrop
            yAxisEndy = max(intersectingLines[i][1][3], intersectingLines[i][1][1])
            yAxisStartx = 0 # TODO:Assumes left hand axis only
            yWidthCrop = abs(intersectingLines[i][0][0] - intersectingLines[i][0][2])//10 # From crop2Axes func
            yAxisEndx = max(intersectingLines[i][1][0]-yWidthCrop, 0) + totalLeftChange  # Original yCrop left + change in yCrop left from iterative squeeze
            yLabelCrop = img[yAxisStarty:yAxisEndy, yAxisStartx:yAxisEndx]
            yLabelCrop = cv.rotate(yLabelCrop, rotateCode=cv.ROTATE_90_CLOCKWISE) # Rotate the image so text is horizontal TODO: Don't assume vertical y label
            # cv.imshow('Y Label Cropped', yLabelCrop)
            # cv.waitKey(0)
            # OCR Y axis label
            ylabelList, yLabelPos = ocrImg(yLabelCrop, type='label')
            yLabels.append(' '.join(ylabelList)) # Join items in list by space

            # Find X axis label location
            xAxisStartx = xStartCrop
            xAxisEndx = max(intersectingLines[i][0][2], intersectingLines[i][0][0])
            xHeightCrop = abs(intersectingLines[i][1][1] - intersectingLines[i][1][3])//10 # From crop2Axes func
            xAxisStarty = min(intersectingLines[i][0][1]+xHeightCrop, img.shape[0]) - totalBottomChange
            xAxisEndy = img.shape[0] # TODO: Assumes bottom of image is end of graph space
            xLabelCrop = img[xAxisStarty:xAxisEndy, xAxisStartx:xAxisEndx]
            # cv.imshow('X Label Cropped', xLabelCrop)
            # cv.waitKey(0)
            # OCR X axis label
            xlabelList, xLabelPos = ocrImg(xLabelCrop, type='label')
            xLabels.append(' '.join(xlabelList)) # Join items in list by space

            return axesMarkers, axesMarkersPos, axesBB, xLabels, yLabels
    return axesMarkers, axesMarkersPos, axesBB, xLabels, yLabels # Return empty values if nothing found

# Rounds input to sig figures
def round_sig(x, sig=3):
    try:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    except:
        return x



def createMap(markerTextList, markerPositionList):
    '''Function to create Linear mapping between pixel location and graph coordinates.
    Inputs list of Marker positions and text values.
    Detects outliers in x and y marker text values
    Returns 2 lists of gradients and intercepts'''

    # Returned lists
    mapGradient = []
    mapIntercept = []

    #####################################################################
    # Determine Mapping
    #####################################################################
    macroMode = [] #Stores step size for each axis
    # Determine x mapping via mode outlier detection
    # Cycle through each graph
    for k, textMarkers in enumerate(markerTextList):
        textMarkers = [float(text) for text in textMarkers] # Convert all markers to floats
        markerPosition = markerPositionList[k]
        macroMode.append([])
        # Mode step sizes for each 'pivot'
        microModeStep = []
        #######################################################
        # Calculate gradient of mapping (m part of linear eq)
        for i,textMarker in enumerate(textMarkers):
            microStep = []
            for j in range(i+1,len(textMarkers)):
                microStep.append(round_sig((float(textMarkers[j]) - float(textMarker))/(j-i), sig=8)) # Round to 8 sig figs to fix bug
            # Find mode of step array
            try:
                if len(microStep) > 1:
                    microModeStep.append(mode(microStep))
                else:
                    microModeStep.append(0)
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
            # Ignore all 0's and see if there's another mode
            zeroIgnored = [num for num in microModeStep if num!=0]
            # Calculate mode of all non zero pivot passes. If success, calc gradient
            try:
                macroMode[k][0] = mode(zeroIgnored)
                mapGradient.append(calcMapGradient(microModeStep, textMarkers, markerPosition))
            except:
                # If still no clear mode, error.
                macroMode[k].append(0)
                print('ERROR: Graph %s axis cannot be interpretted correctly' % k)
                mapGradient.append(0)
                mapIntercept.append(0)
                continue
        else:
            mapGradient.append(calcMapGradient(microModeStep, textMarkers, markerPosition))
        ###################################################
        # Calculate Intercept mapping (+c bit of linear eq)
        step = macroMode[k][0]
        if step == 0: # Check that axis is not undefined
            mapIntercept.append(0)
        else:
            # Easy method first. If 0 exists in marker list and is not erroneous, find it.
            indices = [i for i, x in enumerate(textMarkers) if x == float(0)]
            if len(indices) != 0:
                for idx in indices:
                    if float(0) in textMarkers and microModeStep[idx] != 0:
                        zeroPosPix = markerPosition[idx]
                        c = float(zeroPosPix*mapGradient[k]) # (xgraph = mxpixel + c) so c needs to be in graph coordinates
                        mapIntercept.append(-1*c) 
                        break
                    else:
                        if idx == indices[-1]: # if last idx, go to harder method #TODO: MAKE THE BELOW HARDER METHOD INTO A FUNCTION TO AVOID REPEATING
                            idx = [i for i, x in enumerate(microModeStep) if x == step]
                            if len(idx) > 1:
                                refVal1 = max(float(textMarkers[idx[0]]), float(textMarkers[idx[1]])) # Get max value of 1st 2 none outlier vals
                                refVal2 = min(float(textMarkers[idx[0]]), float(textMarkers[idx[1]]))  # Get min value of 1st 2 none outlier vals
                                refValDiff = refVal1 - refVal2
                                refValPosDiff = markerPosition[textMarkers.index(refVal1)] - markerPosition[textMarkers.index(refVal2)] # Position difference between max and min refVal
                                stepPosDiff = refValPosDiff / (refValDiff / step) # Position difference of 1 step
                                zeroPosPix = markerPosition[textMarkers.index(refVal1)] - ((refVal1/step)*stepPosDiff)
                                c = float(zeroPosPix*mapGradient[k])
                                mapIntercept.append(-1*c)
                                break
                            else: # Can't interpret
                                mapIntercept.append(0) 
            else:
                # Harder method. Extrapolate to find 0 position
                idx = [i for i, x in enumerate(microModeStep) if x == step]
                if len(idx) > 1:
                    refVal1 = max(float(textMarkers[idx[0]]), float(textMarkers[idx[1]])) # Get max value of 1st 2 none outlier vals
                    refVal2 = min(float(textMarkers[idx[0]]), float(textMarkers[idx[1]]))  # Get min value of 1st 2 none outlier vals
                else: # Hope the next label is ok and not an outlier (can't be sure though)
                    refVal1 = max(float(textMarkers[idx[0]]), float(textMarkers[idx[0]+1])) # Get max value of 1st 2 none outlier vals
                    refVal2 = min(float(textMarkers[idx[0]]), float(textMarkers[idx[0]]+1))  # Get min value of 1st 2 none outlier vals

                refValDiff = refVal1 - refVal2
                refValPosDiff = markerPosition[textMarkers.index(refVal1)] - markerPosition[textMarkers.index(refVal2)] # Position difference between max and min refVal
                stepPosDiff = refValPosDiff / (refValDiff / step) # Position difference of 1 step
                zeroPosPix = markerPosition[textMarkers.index(refVal1)] - ((refVal1/step)*stepPosDiff)
                c = float(zeroPosPix*mapGradient[k])
                mapIntercept.append(-1*c)
                break

    return mapGradient, mapIntercept


# calculates map
# Returns mapGraident
# Returns 0 if only 1 non erroneous value / any other error
def calcMapGradient(microModeStep, textMarkers, markerPosition):
    mapGradient = 0
    for i in range(len(textMarkers)-1):
        for j in range(i+1,len(textMarkers)-1):
            # If marker values are not erroneous
            if (microModeStep[i] != 0 and microModeStep[j] != 0):
                mapGradient =  (float(textMarkers[j]) - float(textMarkers[i]))/(float(markerPosition[j]) - float(markerPosition[i]))
                return mapGradient # Break as soon as 1st non-erroneous value is found     
    # Worst case scenorio
    if mapGradient == 0:
        if microModeStep[0] != 0: # If 1st is not 0, then use 1st and 2nd marker to calculate gradient
            mapGradient =  (float(textMarkers[i+1]) - float(textMarkers[i]))/(float(markerPosition[i+1]) - float(markerPosition[i]))
    return mapGradient # Return none is fails to calc gradient


def findInnerAxes(img,axesBB,intersects):
    '''Function to crop around graphs and extract inner part. Inputs raw image, axes bb, x/ycrop (+updated)
    Outputs a img of original size, but with whitespace outside of graph area'''
    # Find inner axes for inner contour identfiication
    # Crop into each axis
    innerAxes = [] # List of inner regions for graphs
    for i in range(len(axesBB)):
        
        x, y = zip(*intersects)
        intersect = [min(x), max(y)] # Bottom Left intersect TODO: Remove this assumption of 'normal LH' axis

        pixPadding = 10 # Pad the inner area by 10 pixels each side (removes noisy markers etc)
        # y0 = int(min(min(axesBB[i][1][1], axesBB[i][1][3]), intersect[1])) + pixPadding
        # y1 = int(min(max(axesBB[i][1][1], axesBB[i][1][3]), intersect[1])) - pixPadding
        # x0 = int(max(min(axesBB[i][0][0],axesBB[i][0][2]),intersect[0])) + pixPadding
        # x1 = int(max(max(axesBB[i][0][0],axesBB[i][0][2]),intersect[0])) - pixPadding

        # if abs(y1 - y0) < 100 or abs(x1 - x0) < 100: # If above logic fails, go back to standard axis detection system
        y0 = int(min(axesBB[i][1][1], axesBB[i][1][3])) + pixPadding
        y1 = int(max(axesBB[i][1][1], axesBB[i][1][3])) - pixPadding
        x0 = int(min(axesBB[i][0][0],axesBB[i][0][2])) + pixPadding
        x1 = int(max(axesBB[i][0][0],axesBB[i][0][2])) - pixPadding

        # Colour all outer regions white
        tempImg = img.copy()
        tempImg[0:y0, :] = [255,255,255] # Colour region above graph white 
        tempImg[y1:tempImg.shape[0], :] = [255,255,255] # Colour region below graph white
        tempImg[:, 0:x0] = [255,255,255] # Colour region left of graph white
        tempImg[:, x1:tempImg.shape[1]] = [255,255,255]# Colour region right of graph white 

        innerAxes.append(tempImg) 
        # plt.imshow(innerAxes[i])
        # plt.show()
    return innerAxes

def findColours(img):
    '''Function to find all unique colours in an image. 
    Groups similar colours. 
    Inputs an image, returns 2 lists. 1 of unique colour tuples (lower and upper H val limits) and other of black line colurs'''
    
    # Convert image to HSV
    img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    uniqueColours = np.unique(img, axis=0)
    # Get all H values
    hVals = uniqueColours[:,:,0]
    vVals = uniqueColours[:,:,2]
    vVals = vVals.ravel()
    hVals = hVals.ravel() #Flatten
    idx = np.where(vVals < 245) # Remove whiteish colours
    hVals = hVals[idx]
    vVals = vVals[idx]
    idx = np.where(vVals > 20) # Remove blackish colours
    hVals = hVals[idx]
    hVals = hVals[hVals > 1]
    # Plot histogram of H value distributions (chose 15 bins as seems reasonable)
    (hn, hbins, patches) = plt.hist(hVals, bins=15) # USEFUL PLOT FOR REPORT 
    #plt.show()
    plt.clf() # Clear plot
    # Get list of coloured and blackend peaks
    hMaxColourFreq = max(hn)
    hColourBins = [] # List of tuple h values (lower and upper limit) which are dominent in plot
    for i in range(len(hn)):
        if hn[i] > 0.4*hMaxColourFreq: # Assume any colour with a freq of > 40% of max freq is key
            hColourBins.append([hbins[i], hbins[i+1]])
    print('Identified %s colours in graph' % len(hColourBins))
    return hColourBins

def applyMaskandFindContour(lower, upper, imgHSV):
    '''Function to apply a mask to an image in the range of lower and upper (np arrays 3x1).
    Applies the findContours function.
    Input image must be in HSV colourspace.
    Outputs numpy arrays of x and y pixel coordinates of contours of the specified colour in the input image.
    Returns empty np arrays if failed'''

    xArr = np.array([])
    yArr = np.array([])
    mask = cv.inRange(imgHSV, lower, upper)
    kernal = np.ones((5, 5), "uint8") 
    mask = cv.dilate(mask, kernal) # Clean up mask / make lines bigger
    #res = cv.bitwise_and(imgHSV, imgHSV, mask = mask)
    #cv.imshow('masked', res)
    # cv.waitKey(0)
    # Find contours from each coloured graph line (masked graph)
    contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    if len(contours) == 0: # Stop if no contours. Return empty array
        return xArr, yArr
    # # Get list of contour lengths -> helps determine line type
    contourLengths = []
    for contour in contours:
        contourLengths.append(contour.shape[0])
    # mainContours = contours
    # for i in range(len(contourLengths)):
    #     if contourLengths[i] < 0.1*max(contourLengths):
    #         mainContours = np.delete(contours,i) # Delete the contour if it is of insignificant size
    # contours = mainContours
    if max(contourLengths) < 100:
        print('DASHED LINE DETECTED')
    
        
    # Flatten contours into an array of [x,y] pixels
    contourCat = np.concatenate([arr for arr in contours], axis=0)
    pixArr = np.reshape(contourCat, (contourCat.shape[0], 2)) # Create a 2D array of x y pixelcoordinates
    # Apply mappings
    xArr = pixArr[:,0]
    yArr = pixArr[:,1]
    return xArr, yArr

def export2CSV(x, y, inputImageFilepath, xLabel, yLabel, graphNo, colourNo):
    '''Converts numpy arrays to pd, applies labels and exports to CSV
    Input: x coordinate array | y coordinate array | input image filepath | x axis label | y axis label | graph number | colour number
    Output: Outputs a CSV file in the nput image directory'''
     # Convert each colour output from each graph to Pandas df with appropriate labels
    df = pd.DataFrame({xLabel: x, yLabel: y}, columns=[xLabel, yLabel])
    # Post Process outputted data points
    df = postProcessData(df, xLabel, yLabel)
    # Write dataframe to CSV
    outputFilepath = os.path.splitext(inputImageFilepath)[0] + '_output_' + str(colourNo) + '.csv'  # First number = graph no. 2nd = colour number
    df.to_csv(outputFilepath,index=False)

def postProcessData(df, xlabel, ylabel, intervalSize=20):
    ''' Post processes the extracted data points. Uses a moving kernel of size = (max-min)/20 by default (20x20 grid)

    Input:
    - df Data frame with 2 columns, x and y
    - xlabel (column identifiers)
    - ylabel (column identifiers)

    Output:
    - Truncated Data frame
    '''

    # Sort both columns
    df.sort_values([xlabel, ylabel], ascending=[True, True])
    x = df[xlabel]
    y = df[ylabel]
    xMin = x.min()
    xMax = x.max()
    yMin = y.min()
    yMax = y.max()

    xKernel = np.linspace(xMin, xMax, intervalSize)
    yKernel = np.linspace(yMin, yMax, intervalSize)

    # Define new dataframe with post processed results
    new_df = pd.DataFrame(columns=[xlabel, ylabel])
    for j in range(len(yKernel)-1):
        for i in range(len(xKernel)-1):
            kernel = df[(df[xlabel] >= xKernel[i]) & (df[xlabel] < xKernel[i+1]) & (df[ylabel] >= yKernel[j]) & (df[ylabel] < yKernel[j+1])]
            newSeries = kernel.mean()
            new_df = new_df.append(newSeries, ignore_index=True)

    # Remove NaN's
    new_df = new_df.dropna()
    new_df = new_df.reset_index(drop=True)
    return new_df


def legendFinder(img, inputImage):
    '''Runs the legend detection model.

    Input: 
    - Img (unedited cv image)
    - inputImage (filepath to input image)

    Output:
    - legendImgs (list of np images of identified legends)
    - noLegendImg (inout image minus all identfied legends)'''
    legendImgs = [] # List containing extracted legends
    imgRawForLegend = img.copy()
    imgRawForGraph = img.copy()
    output_dict = legDetector.runModel(inputImage)
    idx = np.where(output_dict['detection_scores']>0.5) # Get index of all detected legends (i.e probabiliy>50%)
    legendBBs = output_dict['detection_boxes'][idx] # Get bounding box
    # Check if a legend exists
    if legendBBs.size != 0:
        # Mask every identified legend
        mask = np.zeros(imgRawForLegend.shape[:2], dtype='uint8')
        for l,legendBB in enumerate(legendBBs):
            y_min = round(legendBB[0] * imgRawForLegend.shape[0]) # Convert from relative coordinates to absolute
            x_min = round(legendBB[1] * imgRawForLegend.shape[1])
            y_max = round(legendBB[2] * imgRawForLegend.shape[0])
            x_max = round(legendBB[3] * imgRawForLegend.shape[1])
            cv.rectangle(mask, (x_min, y_min), (x_max, y_max), (255,255,255),thickness=-1)
            newLegendImg= imgRawForLegend[y_min:y_max, x_min:x_max]
            legendImgs.append(newLegendImg)
            #cv.imshow('Legend', legendImgs[l])
            #cv.waitKey(0)

        cv.rectangle(imgRawForGraph, (x_min, y_min), (x_max, y_max), (255,255,255),thickness=-1) #'Block' off legend with white rectangle
        noLegendImg = imgRawForGraph
        #noLegendImg = cv.bitwise_not(img, img, mask=mask)
        #cv.imshow('Removed Legend', noLegendImg)
        #cv.waitKey(0)
    else:
        noLegendImg = img

    return legendImgs, noLegendImg

def interpretLegend(legendList,hColours=[], upperBlack=[], lowerBlack=[]):
    '''Runs legend through OCR
    
    Input: 
    
    legendList - Python list of legends (np arrays)
    hColours - Python list of colours identified in graph (output of findColours()). Leave blank if BLACK case
    upperBlack - Upper limit of black colour to identify. Leave black if only interested in colours
    lowerBlack - Lower limit of black colour to identify. Leave black if only interested in colours

    
    Output:
    
    legendLabels - List of legend labels text, where the index corresponds to the colour in hColours'''

    # Initialise list of legend text entries (in order of hColours)
    legendLabels = []
    for legend in legendList:
        # Initialise list of coordinates defining legend line position
        minx = []
        miny = []
        maxx = []
        maxy = []

        # Convert colourspace to HSV
        legend = cv.cvtColor(legend,cv.COLOR_BGR2HSV)

        if hColours != []:
            for hColour in enumerate(hColours[0]):
                lower = np.array([round(hColour[1][0]), 0, 0], np.uint8) 
                upper = np.array([round(hColour[1][1]), 255, 255], np.uint8) 
                mask = cv.inRange(legend, lower, upper)
                kernal = np.ones((5, 5), "uint8") 
                mask = cv.dilate(mask, kernal) # Clean up mask / make lines bigger
                #res = cv.bitwise_and(imgHSV, imgHSV, mask = mask)

                contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # Find coordinates of the line
                # Flatten contours into an array of [x,y] pixels
                contourCat = np.concatenate([arr for arr in contours], axis=0)
                pixArr = np.reshape(contourCat, (contourCat.shape[0], 2)) # Create a 2D array of x y pixelcoordinates
                xArr = pixArr[:,0]
                yArr = pixArr[:,1]

                minx.append(min(xArr))
                maxx.append(max(xArr))
                miny.append(min(yArr))
                maxy.append(max(yArr))

        elif upperBlack != []:
            lower = lowerBlack
            upper = upperBlack
            mask = cv.inRange(legend, lower, upper)
            kernal = np.ones((5, 5), "uint8") 
            mask = cv.dilate(mask, kernal) # Clean up mask / make lines bigger
            #res = cv.bitwise_and(imgHSV, imgHSV, mask = mask)
            # Perfom canny and hough to get lines
            dst = cv.Canny(legend, 50, 200, None, 3)
            lines = cv.HoughLinesP(dst,1,np.pi/180,50,None,50,10)
            #contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # Find coordinates of the line
            # Flatten contours into an array of [x,y] pixels
            contourCat = np.concatenate([arr for arr in contours], axis=0)
            pixArr = np.reshape(contourCat, (contourCat.shape[0], 2)) # Create a 2D array of x y pixelcoordinates
            xArr = pixArr[:,0]
            yArr = pixArr[:,1]
            minx.append(min(xArr))
            maxx.append(max(xArr))
            miny.append(min(yArr))
            maxy.append(max(yArr))

        
        # Only look right of line
        for i in range(len(minx)):
            startOCRx = maxx[i]
            startOCRy = round(miny[i] - 20) # go up 20 pixels
            endOCRy = round(maxy[i] + 20) 
            upperxLimit = [x for x in minx if x > (maxx[i]+10)]
            if len(upperxLimit) > 0: # If a legend entry is detected to the right, end ocr at the start of that, else end at edge of image
                endOCRx = round(min(upperxLimit, legend.shape[1])) # Either go to next colour item, or the edge of legend
            else:
                endOCRx = round(legend.shape[1])
            # Crop legend ready for OCR
            croppedLegend = legend[startOCRy:endOCRy, startOCRx:endOCRx]
            textList, textPosition = ocrImg(croppedLegend, 'text')
            if len(textList) != 0:
                legendLabels.append('_'.join(textList)) # Join items in list by underscore
                # Look for invalid filepath charcters and replace with _
                for k, char in enumerate(legendLabels[i]):
                    if char != "/" or char != "\\" or char != '>' or char != '<' or char != '|' or char != ':' or char != '*' or char != '?':
                        legendLabels[i] += char
                    else:
                        legendLabels[i] += '_'
            else:
                legendLabels.append('UNKNOWN') # Return None if not found

    return legendLabels




#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################


def runBitmap2Data(inputImage):
    '''Main method to run the program.
    
    Input:
    
    Filepath to image to analyse
    
    Output:
    
    CSV file with extracted coordinates'''

    start = timer()
    img, gray, edges = preprocess(inputImage)

    # Find all straight lines in images ############################################################################
    lines = findStraightLines(gray, edges)

    # Categorise straight lines ############################################################################
    horizontalLines, verticalLines = findLineOrientation(lines)

    # Find if horizontal and vertical lines interesct ############################################################################
    intersectingLines, intersects = findIntersects(horizontalLines, verticalLines)
    
    end = timer()
    print('Found candidate axes in: ' + str(end - start))

    
    start = timer()
    # Crop around intersecting lines ############################################################################
    xCrop, yCrop = cropImg(img, intersectingLines)


    # OCR cropped images to check if they are axes. If axes present, OCR them. ############################################################################
    axesMarkers, axesMarkersPos, axesBB, xLabels, yLabels = getAxes(xCrop, yCrop, intersectingLines, img)
    print('MARKERS:')
    print(axesMarkers)
    print('X-LABELS')
    print(xLabels)
    print('Y-LABELS')
    print(yLabels)
    # Update x/ycrop to actual x/y axes
    xCropUpdated, yCropUpdated = cropImg(img, axesBB)

    end = timer()
    print('OCR Axes complete in: ' + str(end - start))

    start = timer()
    # Create mapping between pixel location and graph coordinates ############################################################################
    # Split axesMarkers into x and y TODO: Make this nested list manipulation less messy
    xAxesMarkers, xAxesMarkersPos, yAxesMarkers, yAxesMarkersPos = [],[],[],[]
    for i in range(len(axesMarkers)): # For number of graphs
        xAxesMarkersPos.append([])
        yAxesMarkersPos.append([])
        xAxesMarkers.append(axesMarkers[i][0])
        for j in range(len(axesMarkersPos[i][0])): # For number of markers
            xAxesMarkersPos[i].append(axesMarkersPos[i][0][j][0])
        yAxesMarkers.append(axesMarkers[i][1])
        for k in range(len(axesMarkersPos[i][1])):
            yAxesMarkersPos[i].append(axesMarkersPos[i][1][k][1])

    # Create mapping ############################################################################
    xMapGradient, xMapIntercept = createMap(xAxesMarkers, xAxesMarkersPos)
    yMapGradient, yMapIntercept = createMap(yAxesMarkers, yAxesMarkersPos)
    print('MAPPING:')
    print(xMapGradient, xMapIntercept, yMapGradient, yMapIntercept)
    # Error handler
    if len(xMapGradient) == 0 or len(xMapIntercept) == 0 or len(yMapGradient) == 0 or len(yMapIntercept) == 0:
        print('ERROR: UNABLE TO COMPUTE MAPPING')
        return # Exit function
    for i in range(len(xMapGradient)):
        if xMapGradient[i] == 0 or xMapIntercept[i] == 0 or yMapGradient[i] == 0 or yMapIntercept[i] == 0:
            print('ERROR: UNABLE TO COMPUTE MAPPING FOR GRAPH NO %s' %i)
            return # Exit function

    end = timer()
    print('Mapping created in: ' + str(end - start))

    # Find Contours ############################################################################
    start = timer()

    # Find the legend, extract and remove from graph area
    legendImgs, noLegendImg = legendFinder(img, inputImage)

    # White out outer regions of graph so all non white space has actual data in
    innerAxes = findInnerAxes(noLegendImg,axesBB,intersects) #innerAxes = findInnerAxes(img,axesBB,intersects)

    # Find all colours in inneraxes
    hColours = [] # List of unique colours in each axes 
    for innerAxis in innerAxes:
        hColours.append(findColours(innerAxis))

    # Interpret / read legend 
    legendLabels = interpretLegend(legendImgs, hColours=hColours)

    end = timer()
    print('Colours and legends analysed in: ' + str(end - start))

    start = timer()
    # Mask each colour
    # Convert images to HSV and apply masks and mappings
    xCoordinates = [] # List of np arrays of graph x coordinates
    yCoordinates = [] # List of np array of graph y coordinates
    for i,innerAxis in enumerate(innerAxes):
        xCoordinates.append([]) # Create new lists for each graph in image
        yCoordinates.append([])
        ########## COLOUR BASED SEGMENTATION METHOD ###########################################
        imgHSV = cv.cvtColor(innerAxis,cv.COLOR_BGR2HSV)
        # Apply mask for each colour range
        for j, hColour in enumerate(hColours[i]):
            lower = np.array([round(hColour[0]), 0, 0], np.uint8) 
            upper = np.array([round(hColour[1]), 255, 255], np.uint8)
            startCont = timer() 
            xArr, yArr = applyMaskandFindContour(lower,upper,imgHSV)
            endCont = timer()
            print('Lines to extract identified in: ' + str(endCont - startCont))
            xCoordinates[i].append((xArr * xMapGradient[i]) + xMapIntercept[i])
            yCoordinates[i].append((yArr * yMapGradient[i]) + yMapIntercept[i])
            # Export to CSV
            try:
                export2CSV(xCoordinates[i][j], yCoordinates[i][j], inputImage, xLabels[i], yLabels[i], i, legendLabels[j])
            except:
                export2CSV(xCoordinates[i][j], yCoordinates[i][j], inputImage, xLabels[i], yLabels[i], i, j) # If no legend found
            #plt.scatter(xCoordinates[i][j], yCoordinates[i][j])
            #plt.show()

        ##########  BLACK SEGMENTATION METHOD ###########################################
        # Hard code black line defintion
        lowerBlack = np.array([0, 0, 0], np.uint8) 
        upperBlack = np.array([179, 160, 20], np.uint8) # HSV MAX = (179,255,255)
        xArr, yArr = applyMaskandFindContour(lowerBlack,upperBlack,imgHSV)
        #legendLabels = interpretLegend(legendImgs, lowerBlack=lowerBlack, upperBlack=upperBlack) #TODO: ADD LEGEND INTERPRETATION FOR BLACK LINES
        if xArr.size != 0 or yArr.size != 0:
            xCoordinates[i].append((xArr * xMapGradient[i]) + xMapIntercept[i])
            yCoordinates[i].append((yArr * yMapGradient[i]) + yMapIntercept[i])
            # Export to CSV
            export2CSV(xCoordinates[i][-1], yCoordinates[i][-1], inputImage, xLabels[i], yLabels[i], i, len(xCoordinates[i])-1) # TODO: Change colour naming convention to match legend
            #plt.scatter(xCoordinates[i][-1], yCoordinates[i][-1])
            #plt.show()
        else:
            print('No Black contours found in image')


    # plt.imshow(img)
    # plt.show()
    end = timer()
    print('Mapping applied and export in: ' + str(end - start))
    print('----------------FINISHED----------------')



# TODO: FIX DOTTED LINES OUTPUT FROM FIND CONTOURS & IDENTIFY DIFFERENT LINE TYPES (USE NUMBER OF CONTOURS)
# TODO: IDENTIFY GRIDLINES (USING HOUGHLINE TRANSFORM AND BASIC INTERCEPT LOGIC OR a sweeping kernal to identify colour intensity in all graph regions (~const intensity = grid colour) and remove this using a mask)
                    

# Initialise the legend detector class and front loads OD model
legDetector =  legendDetector('/Users/Jordan/Documents/Imperial/ME4/FYP/Data_Extraction_Project/Python/bitmap/legend_detection/edet_d0/inference_graph/saved_model') #INPUT PATH TO LEGEND DETECTION 'saved_model' FOLDER

# Run the model
iterationNumber = 0
for image_path in glob.glob('/Users/Jordan/Documents/Imperial/ME4/FYP/Data_Extraction_Project/Python/Bitmap_TestCases/Testing/*.png'): #INSERT FOLDER CONTAINING PNGs OF GRAPHS
    iterationNumber = iterationNumber+1
    try:
        startFull = timer()
        print('PROCESSING:' + str(image_path))
        runBitmap2Data(image_path)
        endFull = timer()
        print('FULL PROCESS DONE IN: ' + str(endFull - startFull))
        print('Iteration Number = ' + str(iterationNumber))
        print('----------------------------------------')
    except:
        print('ERROR PROCESSING - UNHANDLED EXCEPTION')
        print('----------------------------------------')