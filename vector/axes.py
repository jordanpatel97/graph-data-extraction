from re import compile, match
import preprocess

# Initialise preprocess class
pre = preprocess.preprocess('/Users/Jordan/Documents/Imperial/ME4/FYP/Data_Extraction_Project/Python/TestCases/pg_0012.pdf.svg')
root = pre.tree.getroot()

# Function to remove duplicates from 2 lists based on only 1 list
def remove_duplicates(lst,lstID):
    res = []
    resID = []
    for i,x in enumerate(lst):
        if x not in res:
            res.append(x)
            resID.append(lstID[i])
    return res, resID

    
def findSimpleHorizontalAxes(dPaths, pathIDs):
        # Find all horizontal lines (Only M . . H .)
        regexp = compile(r'[M]\s[+-]?[0-9]*[.]?[0-9]+\s[+-]?[0-9]*[.]?[0-9]+\s[H]\s[+-]?[0-9]*[.]?[0-9]+$')
        horizontalLines = []
        horizontalLineIDs =[]
        for i,dPath in enumerate(dPaths):
                if regexp.match(dPath):
                        horizontalLines.append(dPath)
                        horizontalLineIDs.append(pathIDs[i])
        # Remove duplicates
        horizontalLines, horizontalLineIDs = remove_duplicates(horizontalLines,horizontalLineIDs)
        # Remove h lines smaller than 1/6 of page width
        pgWidth = float(root.attrib['width'])
        newHLines = [] # Temp arrays to populate during length check
        newHLinesID = []
        for i,line in enumerate(horizontalLines):
                parsedPath = pre.parsePath(str(line))
                H = parsedPath[4]
                Mx = parsedPath[1]
                HLength = abs(float(H)-float(Mx))
                # add lines less than 1/6 of pgwidth
                if HLength > pgWidth/7:
                        newHLines.append(horizontalLines[i])
                        newHLinesID.append(horizontalLineIDs[i])
        horizontalLineIDs = newHLinesID
        horizontalLines = newHLines
        del newHLines, newHLinesID
        return horizontalLines, horizontalLineIDs

# Function to find horizontal and vertical lines of a random form 
# Check length of all vertical lines vs horizontal lines in path and if ratio is large, path is x / y axis
def findComplexAxes(dPaths, pathIDs):
        horizontalLines = []
        horizontalLineIDs =[]
        verticalLines = []
        verticalLineIDs =[]
        pgHeight = float(root.attrib['height'])
        pgWidth = float(root.attrib['width'])

        for i,dPath in enumerate(dPaths):
                # Ignore all extremely large paths
                if len(dPath) > 100:
                        continue
                parsedPath = pre.parsePath(dPath)
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
        verticalLines, verticalLineIDs = remove_duplicates(verticalLines,verticalLineIDs)       
        horizontalLines, horizontalLineIDs = remove_duplicates(horizontalLines,horizontalLineIDs)  

        return horizontalLines, horizontalLineIDs, verticalLines, verticalLineIDs

def findSimpleVerticalAxes(dPaths, pathIDs):
        # Find all vertical lines (M . . V .)
        regexp = compile(r'[M]\s[+-]?[0-9]*[.]?[0-9]+\s[+-]?[0-9]*[.]?[0-9]+\s[V]\s[+-]?[0-9]*[.]?[0-9]+$')
        verticalLines = []
        verticalLineIDs =[]
        for i,dPath in enumerate(dPaths):
                if regexp.match(dPath):
                        verticalLines.append(dPath)
                        verticalLineIDs.append(pathIDs[i])
        # Remove duplicates
        verticalLines, verticalLineIDs = remove_duplicates(verticalLines,verticalLineIDs)
        # Remove h lines smaller than 1/6 of page width
        pgHeight = float(root.attrib['height'])
        newVLines = [] # Temp arrays to populate during length check
        newVLinesID = []
        for i,line in enumerate(verticalLines):
                parsedPath = pre.parsePath(str(line))
                V = parsedPath[4]
                My = parsedPath[2]
                VLength = abs(float(V)-float(My))
                # add lines less than 1/6 of pgwidth
                if VLength > pgHeight/15:
                        newVLines.append(verticalLines[i])
                        newVLinesID.append(verticalLineIDs[i])
        verticalLineIDs = newVLinesID
        verticalLines = newVLines
        del newVLines, newVLinesID
        return verticalLines, verticalLineIDs

# Group by interestion point (axes groups)
# TODO: Only works for expressions with 1 M
def findItersects(horizontalLines, horizontalLineIDs, verticalLines, verticalLineIDs):
        axes = [] # 2d list of paths [horiz ID, verti ID]
        horizontalAxesVals = [] # COntains actual x axis path
        verticalAxesVals = [] # Contains actual y axis path
        for i,vLine in enumerate(verticalLines):
                parsedvPath = pre.parsePath(str(vLine))
                idxV = parsedvPath.index('M')
                vOrigin = parsedvPath[idxV+1:idxV+3]
                for j,hLine in enumerate(horizontalLines):
                        parsedhPath = pre.parsePath(str(hLine))
                        idxH = parsedhPath.index('M')
                        hOrigin = parsedhPath[idxH+1:idxH+3]

                        # Check that the verical lines and horizontal line origins are very close 1% difference allowed
                        if abs((float(vOrigin[0])/float(hOrigin[0]))-1) < 0.01 and abs((float(vOrigin[1])/float(hOrigin[1]))-1) < 0.01:
                                axes.append([horizontalLineIDs[j], verticalLineIDs[i]])
                                horizontalAxesVals.append(horizontalLines[j])
                                verticalAxesVals.append(verticalLines[i])

        return axes, horizontalAxesVals, verticalAxesVals

def findBoundingBox(axes, horizontalAxesVals, verticalAxesVals):
        axesBB = [] #[['startx', 'endx'], ['starty', 'endy']]
        # Temp lists
        axesBBx = []
        axesBBy = []

        # Cycle through horizontal lines
        for dPath in horizontalAxesVals:
                parsedPath = pre.parsePath(dPath)
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
                parsedPath = pre.parsePath(dPath)
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
        


# Get all paths and associated IDs
def run():
        pathIDs = []
        dPaths = []
        for p in root.iter('{http://www.w3.org/2000/svg}path'):
                strPath = str(p.get('d'))
                strPath = strPath.strip('\n') # Remove new line command
                strPath = strPath.strip() # Remove whitespace before and after string
                pathIDs.append(str(p.get('id')))
                dPaths.append(strPath)

        horizontalLines, horizontalLineIDs, verticalLines, verticalLineIDs = findComplexAxes(dPaths, pathIDs)
        # Find horizontal lines of form M . . H .
        #horizontalLines, horizontalLineIDs = findSimpleHorizontalAxes(dPaths, pathIDs)
        # Find horizontal lines of any form with main compenent being H
        
        # Find vertical lines of form M . . H .
        #verticalLines, verticalLineIDs = findSimpleVerticalAxes(dPaths, pathIDs)
        # Find vertical lines of any form with main compenent being V

        # Find axes combinations by grouping horixontal and vertical lines that iteresect
        axes, horizontalAxesVals, verticalAxesVals = findItersects(horizontalLines, horizontalLineIDs, verticalLines, verticalLineIDs)
        axesBB = findBoundingBox(axes, horizontalAxesVals, verticalAxesVals)

        print(len(axes))
        
run()




