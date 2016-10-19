import FreeCAD as fc
import Part, Sketcher, Draft

import math


def newObject(feature, name):
    obj = fc.ActiveDocument.getObject(name)
    if (obj == None):
        obj = fc.ActiveDocument.addObject(feature, name)
    return obj


def recompute():
    fc.ActiveDocument.recompute()


def newBox(name, boxSize, boxPlacement):
    box = newObject("Part::Box", name)
    box.Length = boxSize.x # x-axis
    box.Width  = boxSize.y  # y-axis
    box.Height = boxSize.z # z-axis
    box.Placement = boxPlacement

    return box


def centeredBox(name, boxSize, boxZ):
    length = boxSize.x
    width = boxSize.y
    box = newBox(name, boxSize, fc.Placement(fc.Vector(-0.5*length, -0.5*width, boxZ), fc.Rotation(0,0,0)))
    return box

def moveTo(obj, placement):
    obj.Placement = placement
    return obj


def newPlane(name, planeOrigin, planeOrientation):
    plane = newObject("Part::Plane", name)
    plane.Length = 10
    plane.Width  = 10

    plane = moveTo(plane, fc.Placement(planeOrigin, planeOrientation))

    return plane


def newSketchOnPlane(name, plane):
    sketch = newObject("Sketcher::SketchObject", name)
    sketch.Support = (plane,["Face1"])
    return sketch


def sketchClosedWire(sketch, vectorList):
    numVectors = len(vectorList)
    # Draw the lines
    for i in range(numVectors):
        iNext = (i + 1)%numVectors

        v0 = vectorList[i]
        v1 = vectorList[iNext]

        sketch.addGeometry(Part.Line(v0, v1))

    # Add the constraints
    for i in range(numVectors):
        iNext = (i + 1)%numVectors

        sketch.addConstraint(Sketcher.Constraint("Coincident", i, 2, iNext, 1))

    return sketch

def threePointCenter(p0, p1, p2):
    # Taken from wikipedia 'Circumscribed circle' page
    a = p1 - p0 
    b = p2 - p0

    a2 = a.dot(a)
    b2 = b.dot(b)
    ab = a.dot(b)
    axb = a.cross(b)
    axb2 = axb.dot(axb)
    ba2_ab2 = b*a2 - a*b2

    denom = 2.0 * axb2
    if (denom < 1e-6):
        raise Exception("Points are colinear")

    center = ba2_ab2.cross(axb)*(1.0/denom) + p0
    return center

def twoPointCenterTangent(p0, p1, center):
    # vector from center to point0
    r0 = p0 - center
    # vector from center to point1
    r1 = p1 - center
    
    # outOfPlane normal
    outOfPlane = r0.cross(r1).normalize()

    # tangent lying on arc in point 'center+r1'
    tangent = outOfPlane.cross(r1).normalize()
    return tangent


def twoPointTangentCenter(p0, p1, tangent):
    # find vector normal to tangent, in plane spanned by p0,p1,tangent
    outOfPlane = (p1 - p0).cross(tangent)
    normal = tangent.cross(outOfPlane)
    normal.normalize()
    
    # based on: norm(a) `dot` (R*n) = 0.5* ||a||
    #  => a/||a|| `dot` (R*n) = 0.5* ||a||
    #  => a `dot` (R*n) = 0.5 * sq(||a||)
    #  => a `dot` (R*n) = 0.5 * (a `dot` a)
    a = p1 - p0
    R = 0.5*(a.dot(a))/(a.dot(normal))
    center = p0 + normal*R
    return center



# Tangent is only used for sign
def twoPointCenterTangentMiddle(p0, p1, center, tangent):
    # vector from center to point0
    r = p0 - center

    # radius
    R = math.sqrt(r.dot(r))

    # normalized vector from center to middle
    v = ((p0+p1)*0.5 - center).normalize()

    # direction
    sign = 1
    if (tangent.dot(v) < 0.0):
        sign = -1

    # combine center, normalized vector, direction and radius to get the middle point
    pm = center + v*sign*R
    return pm


# TODO: provisions for colinear points
def addPolyArcGeometry(sketch, vertices):
    numVertices = len(vertices)
    # always use the first 3 points
    if (numVertices >= 3):
        sketch.addGeometry(Part.ArcOfCircle(vertices[0], vertices[1], vertices[2]), False)

    if (numVertices > 3):
        count = 3
        center = threePointCenter(vertices[0], vertices[1], vertices[2])
        tangent = twoPointCenterTangent(vertices[0], vertices[2], center)


        while (count < numVertices):
            vertex = vertices[count-1]
            vertexNext = vertices[count]
            center = twoPointTangentCenter(vertex, vertexNext, tangent)

            # For debugging:
            #sketch.addGeometry(Part.Point(center))
            #sketch.addGeometry(Part.Line(vertex, vertex +tangent*5))

            vertexMiddle = twoPointCenterTangentMiddle(vertex, vertexNext, center, tangent)
            sketch.addGeometry(Part.ArcOfCircle(vertex, vertexMiddle, vertexNext), False)

            tangent = twoPointCenterTangent(vertex, vertexNext, center)
            count = count + 1


    return sketch


def sketchPolyArc(sketch, vertexList):
    numVertices = len(vertexList)
    if   (numVertices <= 1):
        raise Exception("Need at least two vertices")
    elif (numVertices == 2):
        sketch.addGeometry(Part.Line(vertexList[0], vertexList[1]), False)
    elif (numVertices >= 3):
        sketch = addPolyArcGeometry(sketch, vertexList)

    return sketch

def extrudeSketch(name, sketch, dirVector):
    extrusion = newObject("Part::Extrusion", name)
    extrusion.Base = sketch
    extrusion.Solid = (True)
    extrusion.Dir = dirVector
    extrusion.Label = name
    extrusion.TaperAngle = (0)
    return extrusion


# Combine obj1 and obj2 into a fusion part
def fuse(name, objList):
    if (objList.__len__() == 1):
        return objList[0]
    else:
        fusion = newObject("Part::MultiFuse", name)
        fusion.Shapes = objList
        return fusion


def boolFuse(name, obj1, obj2):
    return fuse(name, [obj1, obj2])


# Remove obj2 from obj1 creating a new cut part
def cut(name, obj1, obj2):
    cut = newObject("Part::Cut", name)
    cut.Base = obj1
    cut.Tool = obj2
    return cut


def sketchToBox(baseName, planeOrigin, planeOrientation, sketchVectorList, direction):
    plane  = newPlane(baseName+"Plane", planeOrigin, planeOrientation)

    sketch = newSketchOnPlane(baseName+"Sketch", plane)
    sketch = sketchClosedWire(sketch, sketchVectorList)

    extrusion = extrudeSketch(baseName+"Extrusion", sketch, direction)
    return extrusion


# Apply any boolean operation to the workpiece and an extruded sketch
def simpleBoolean(workpiece, baseName, boolOp, planeOrigin, planeOrientation, sketchVectorList,
        direction):
    extrusion = sketchToBox(baseName, planeOrigin, planeOrientation, sketchVectorList, direction)
    
    workpiece = boolOp(baseName, workpiece, extrusion)
    return workpiece


# This is done by rotating a vector
def rotationToNormal(rotation):
    matrix = fc.Placement(fc.Vector(0,0,0), rotation).toMatrix()

    vector = fc.Placement(fc.Vector(0,0,1), fc.Rotation(0,0,0)).toMatrix()

    matrixOut = matrix*vector
    vectorOut = fc.Placement(matrixOut).Base.normalize()
    return vectorOut


# Returns two objects!
def simpleMirror(name, workpiece, planeOrigin, planeOrientation):
    mirror = newObject("Part::Mirroring", name)
    mirror.Source = workpiece
    mirror.Base = planeOrigin
    mirror.Normal = rotationToNormal(planeOrientation)
    return mirror

def mirrorFuse(baseName, workpiece, planeOrigin, planeOrientation):
    mirror = simpleMirror(baseName+"Mirror", workpiece, planeOrigin, planeOrientation)
    workpiece = boolFuse(baseName+"Fuse", workpiece, mirror)
    return workpiece
    
    
# Extrude a sketch and merge the solid with the workpiece
#  The first and last points of sketchVectorList are connected in order to form a closed wire
def fuseSimpleExtrusion(workpiece, baseName, planeOrigin, planeOrientation, sketchVectorList,
        extrusionDirection):
    workpiece = simpleBoolean(workpiece, baseName, boolFuse, planeOrigin, planeOrientation, sketchVectorList,
            extrusionDirection)
    return workpiece


def simplePocket(workpiece, baseName, planeOrigin, planeOrientation, sketchVectorList, 
        pocketDirection):
    workpiece = simpleBoolean(workpiece, baseName, cut, planeOrigin, planeOrientation, sketchVectorList,
            pocketDirection)
    return workpiece


def singleCopy(name, obj, placement):
    objCopy = newObject("Part::Feature", name)
    objCopy.Shape = obj.Shape
    objCopy.Placement = placement
    return objCopy
    

def manyCopies(baseName, obj, placements):
    objCopies = []
    for i in range(placements.__len__()):
        name = baseName+str(i)
        objCopy = singleCopy(name, obj, placements[i])
        objCopies.append(objCopy)

    return objCopies


def removeMany(baseName, workpiece, tool, placementList):
    objectCopies = manyCopies(baseName, tool, placementList)
    fusedCopies = fuse(baseName+"Fuse", objectCopies)
    workpiece = cut(baseName+"Cut", workpiece, fusedCopies)

    return workpiece


def scaledCopy(obj, center, scaleFactor):
    copy   = Draft.clone(obj)
    scaled = Draft.scale([obj], delta=fc.Vector(scaleFactor, scaleFactor, scaleFactor), center=center, copy=True)
    return scaled


def polyArcSketch(baseName, planeOrigin, planeOrientation, vertexList):
    plane = newPlane(baseName+"Plane", planeOrigin, planeOrientation)
    sketch = newSketchOnPlane(baseName, plane)
    sketch = sketchPolyArc(sketch, vertexList)
    return sketch


def sketchesToSweep(name, sweepSketches, spineSketch):
    sweep = newObject('Part::Sweep', name)
    sweep.Sections = sweepSketches
    sweep.Spine = (spineSketch,["Edge1"])
    sweep.Solid = False
    sweep.Frenet = False
    return sweep


def surfaceToFaces(surface):
    # It's important to do a recompute before every Draft. command
    recompute()
    obj = Draft.downgrade(surface, force="splitFaces")
    facesUnnamed = obj[0]
    
    return facesUnnamed


def faceToWire(face):
    recompute()
    wire = Draft.downgrade(face)[0][0]
    return wire


def facesToWires(faces):
    wires = []
    for face in faces:
        wire = faceToWire(face)
        wires.append(wire)
    return wires

 
def wirePairToRuledSurface(name, wire1, wire2):
    ruledSurface = newObject("Part::RuledSurface", name)
    ruledSurface.Curve1 = (wire1,[''])
    ruledSurface.Curve2 = (wire2,[''])
    return ruledSurface


def wirePairsToRuledSurfaces(baseName, wires1, wires2):
    ruledSurfaces = []
    for i in range(len(wires1)):
        ruledSurfaces.append(wirePairToRuledSurface(baseName+str(i), wires1[i], wires2[i]))
    return ruledSurfaces
     

def ruledSurfaceToFaces(baseName, ruledSurface):
    faces = surfaceToFaces(ruledSurface)
    return faces


def facesToShell(name, faces):
    _ = Part.Shell(faces)
    if _.isNull():
        raise RuntimeError('Error: failed to create shell')
    shell = newObject("Part::Feature", name)
    shell.Shape = _.removeSplitter()
    del _
    return shell


def facesToFaces(faces):
    facesFaces = []
    for face in faces:
        facesFace = face.Shape.Face1
        facesFaces.append(facesFace)
    return facesFaces


def facePairToShell(baseName, face1, face2):
    wire1 = faceToWire(face1)
    wire2 = faceToWire(face2)

    ruledSurface = wirePairToRuledSurface(baseName+"RuledSurface", wire1, wire2)
    ruledSurfaceFaces = ruledSurfaceToFaces(baseName+"Ruled", ruledSurface)

    # Merge the faces and ruledSurface into a shell
    faces = [face1] + ruledSurfaceFaces + [face2]
    facesFaces = facesToFaces(faces)
    shell = facesToShell(baseName+"Shell", facesFaces)
    return shell


def shellToSolid(baseName, shell):
    _ = Part.Solid(shell.Shape)
    if _.isNull():
        raise RuntimeError('Error: failed to create solid')
    solid = newObject("Part::Feature", baseName+"Solid")
    solid.Shape = _.removeSplitter()
    return solid


def facePairToPad(baseName, face1, face2):
    shell = facePairToShell(baseName, face1, face2)

    solid = shellToSolid(baseName, shell)
    return solid


def facePairsToPad(baseName, faces1, faces2):
    solids = []
    for i in range(len(faces1)):
        face1 = faces1[i]
        face2 = faces2[i]
        solid = facePairToPad(baseName+str(i), face1, face2)
        solids.append(solid)

    # Merge all the solids
    pad = fuse(baseName+"Fuse", solids)
    return pad


def surfacePairToPad(baseName, surface1, surface2):
    faces1 = surfaceToFaces(surface1)
    faces2 = surfaceToFaces(surface2)
    pad = facePairsToPad(baseName, faces1, faces2)
    return pad
