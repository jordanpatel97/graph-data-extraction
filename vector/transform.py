import xml.etree.ElementTree as ET
import re

def applyTransforms(path):
        tree = ET.parse(path)
        root = tree.getroot()

        # Iterate over all groups
        for group in root.iter('{http://www.w3.org/2000/svg}g'):
                translation = group.get('transform')
                if group.get('id') == 'g272':
                        print(group.get('transform')) #This is the tranlation
                        # Ensures only translations are processed -- ignore matrix transform at the top of the tree
                        if 'translate' in str(translation):
                                # Cycle through all children to be transformed
                                for child in group:
                                        path = child.get('d')
                                        #print(child.get('d')) #This is the path
                                        if path is not None:
                                                transformed2Absolute(translation, path)

def transformed2Absolute(translation, path):
        # Parse both strings (see parsePath method in svg2data)
        # Add translation onto M part of path

        # Parse tranlsation
        translationx, translationy = parseTranslation(translation)
        print(translationx, translationy)

        # Parse path
        pathList = parsePath(path)

        print(pathList)

        # Apply the translation to the path
        applyTranslation(translationx, translationy, pathList)



def parseTranslation(translation):
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

def parsePath(path):
        
        path = str(path)
        pathList = re.split(r'[,\s]\s*', path) # Split path by space and , delimiters
        return pathList

def applyTranslation(translationx, translationy, pathList):
        #Convert all paths to absolute
        pathList = convertReltoAbs(pathList)

        # If letter is capital, add tranlsation onto all components
        translationx = float(translationx)
        translationy = float(translationy)

        currentCommand = str(pathList[0])
        newPathList = []
        numCount = 2 # Counter to see how far along into each command we are
        isx = True
        for item in pathList:
                # Check if abosolute or relative
                if currentCommand.isupper():
                        # Make sure we are perfoming addition to a numeric element
                        try:
                                num = float(item)
                                if (isx == True) and ('V' not in currentCommand):
                                        newPathList.append(str(num+translationx))
                                elif (isx == True) and ('V' in currentCommand):
                                        newPathList.append(num+translationy)
                                elif (isx == False) and ('H' not in currentCommand):
                                        newPathList.append(str(num+translationy))
                                else:
                                        newPathList.append(num+translationx)
                                
                                numCount = numCount + 1 # Increment counter
                                # If numCount is even, next iteration is x, if odd, y
                                if (numCount % 2) == 0:
                                        isx = True
                                else: 
                                        isx = False
                        # Occurs if item is non numeric
                        except:
                                currentCommand = str(item) # Sets new command value
                                numCount = 2 # Reset counter
                                newPathList.append(item) 
                                continue
                # else if letter is not capital, translate onto m only.
        
        print(newPathList)       

        
        return newPathList

def convertReltoAbs(pathList):
        newPathList = []
        for i,path in enumerate(pathList):
                if 'm' in path:
                        relx = pathList[i+1] #Gets next number next to m (i.e. the rel moveto x val)
                        rely = pathList[i+2]
                        # No change necessary if m = 0 0 apart from relabelling
                        if (float(relx) == 0) and (float(rely) == 0):
                                newPathList = pathList
                                newPathList = [x.upper() for x in pathList]
                        # TODO: Do else version!!!!s
        return newPathList


applyTransforms('/Users/Jordan/Documents/Imperial/ME4/FYP/Data_Extraction_Project/Python/vector/input.svg')
