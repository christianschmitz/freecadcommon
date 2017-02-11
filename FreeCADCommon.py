import math
import os

import FreeCAD as fc
import Part, Sketcher, Draft, Mesh


## Basic operations
def autoName(feature):
    numObjects = len(fc.ActiveDocument.Objects)
    name = feature.split(':')[-1] + str(numObjects)
    return name


def newObject(feature):
    name = autoName(feature)
    obj = fc.ActiveDocument.getObject(name)
    if (obj != None):
        fc.ActiveDocument.removeObject(name)
    obj = fc.ActiveDocument.addObject(feature, name)
    if (hasattr(obj, "Label")):
        obj.Label = name

    return obj


def recompute():
    fc.ActiveDocument.recompute()


def show(obj):
    obj.ViewObject.show()
    return


def hide(obj):
    obj.ViewObject.hide()
    return


def hideAll():
    for obj in fc.ActiveDocument.Objects:
        hide(obj)
    return 


def showOnly(obj):
    hideAll()
    show(obj)
    return


def deleteAll():
    for obj in fc.ActiveDocument.Objects:
        name = obj.Name
        fc.ActiveDocument.removeObject(name)
    return


def cleanDocument():
    if (None == fc.ActiveDocument):
        name = "Unnamed"
        fc.newDocument(name)
        fc.setActiveDocument(name)
    else:
        deleteAll()
    return


# u: around local z-axis
# v: around local y-axis
# w: around local x-axis
def ezPlacement(x=0, y=0, z=0, u=0, v=0, w=0):
    return fc.Placement(fc.Vector(x,y,z), 
            fc.Rotation(u,v,w))


def fuse(objects):
    if (len(objects) == 1):
        return objects[0]
    else:
        fusion = newObject("Part::MultiFuse")
        fusion.Shapes = objects
        return fusion


# Remove obj2 from obj1 creating a new cut part
def cut(obj1, obj2):
    c = newObject("Part::Cut")
    c.Base = obj1
    c.Tool = obj2
    c.Label = c.Name
    return c


# Left-most is most global (position vectors are left multiplied by matrices)
def foldPlacements(placements):
    matrix = placements[0].toMatrix()
    for p in placements[1:]:
        matrix = matrix*p.toMatrix()

    return fc.Placement(matrix)


# input: two lists
def move(placements1, placements2):
    folded2 = foldPlacements(placements2)
    placements3 = []
    for p in placements1:
        placements3.append(foldPlacements([p, folded2]))
    return placements3


## Basic shapes
def newBox(place, lx, ly, lz, relLocalPos = fc.Vector(0,0,0)):
    box = newObject("Part::Box")
    box.Length = lx
    box.Width  = ly
    box.Height = lz
    box.Placement = foldPlacements([place,
        ezPlacement(-lx*relLocalPos.x, -ly*relLocalPos.y, 
            -lz*relLocalPos.z)])

    return box


def newCylinder(place, radius, height, relLocalPos = fc.Vector(0,0,0)):
    cylinder = newObject("Part::Cylinder")
    cylinder.Label = cylinder.Name
    cylinder.Radius = radius
    cylinder.Height = height

    cylinder.Placement = foldPlacements([place, 
        ezPlacement(-radius*relLocalPos.x, -radius*relLocalPos.y,
            -height*relLocalPos.z)])

    return cylinder


def newSphere(place, radius, relPos = fc.Vector(0,0,0)):
    sphere = newObject("Part::Sphere")
    sphere.Label = sphere.Name
    sphere.Radius = radius

    sphere.Placement = foldPlacements([place,
        ezPlacement(-radius*relPos.x, -radius*relPos.y, 
            -radius*relPos.z)])
    return sphere


def newWedge(place, lx1, lx2, ly, lz, relPos = fc.Vector(0,0,0)):
    wedge = newObject("Part::Wedge")
    wedge.Xmin = 0
    wedge.Ymin = 0
    wedge.Zmin = 0
    wedge.X2min= 0
    wedge.Z2min= 0
    wedge.Xmax = lx1
    wedge.Ymax = ly1
    wedge.Zmax = lz1
    wedge.X2max = lx2
    wedge.Z2max = lz1
    wedge.Placement = foldPlacements([place,
        ezPlacement(-max(lx1,lx2)*relPos.x, -ly1*relPos.y, -lz1*relPos.z)])
    return wedge


def newSimpleEllipsoid(place, rx, ry, rz, latSouth=-90, latNorth=90, longitude=360,
        relPos = fc.Vector(0,0,0)):
    ellipsoid = newObject("Part::Ellipsoid")
    ellipsoid.Radius1 = rz
    ellipsoid.Radius2 = rx
    ellipsoid.Radius3 = ry
    ellipsoid.Angle1 = latSouth
    ellipsoid.Angle2 = latNorth
    ellipsoid.Angle3 = longitude
    ellipsoid.Placement = foldPlacements([place,
        ezPlacement(-rx*relPos.x, -ry*relPos.y, -rz*relPos.z)])
    ellipsoid.Label = ellipsoid.Name
    return ellipsoid


def newPlane(place, lx = 10, ly = 10):
    plane = newObject("Part::Plane")
    plane.Length = lx
    plane.Width  = ly
    plane.Placement = place

    return plane


def newSketch(plane):
    sketch = newObject("Sketcher::SketchObject")
    sketch.Support = (plane,["Face1"])
    return sketch


## Compound shapes
def newHollowCylinder(place, innerRadius, outerRadius, height,
        relLocalPos = fc.Vector(0,0,0)):
    innerCylinder = newCylinder(place, innerRadius, height, relLocalPos)
    cylinder = newCylinder(place, outerRadius, height, relLocalPos)
    cylinder = cut(cylinder, innerCylinder)
    return cylinder


# TODO: angles
def newHollowSphere(place, innerRadius, radius,
        relPos = fc.Vector(0,0,0)):
    innerSphere = newSphere(place, innerRadius, relPos)
    sphere = newSphere(place, radius, relPos)
    return cut(sphere, innerSphere)


# TODO: input check
def newEllipsoid(place, rx, ry, rz, latSouth=-90, latNorth=90, longWest=0, longEast=360,
        relPos = fc.Vector(0,0,0)):
    ellipsoid = newSimpleEllipsoid(place, rx, ry, rz, latSouth, latNorth, longEast, relPos)
    if not (longWest == 0):
        if (longWest > 0):
            ellipsoidWest = newSimpleEllipsoid(place, rx, ry, rz, latSouth, latNorth, longWest, relPos)
            ellipsoid = cut(ellipsoid, ellipsoidWest)
        elif (longWest < 0):
            ellipsoidWest = newSimpleEllipsoid(place, rx, ry, rz, latSouth, latNorth, 360, relPos)
            ellipsoidWestCut = newSimpleEllipsoid(place, rx, ry, rz, latSouth, latNorth, 360+longWest, relPos)
            ellipsoidWest = cut(ellipsoidWest, ellipsoidWestCut)
            ellipsoid = fuse([ellipsoid, ellipsoidWest])
    return ellipsoid


def newHollowEllipsoid(place, innerRx, rx, ry, rz, 
        latSouth=-90, latNorth=90, longWest =0, longEast=360,
        relPos = fc.Vector(0,0,0)):
    a = innerRx/float(rx)
    aInv = 1.0/a
    innerEllipsoid = newEllipsoid(place, innerRx, a*ry, a*rz, 
            latSouth, latNorth, longWest, longEast, relPos*aInv)
    ellipsoid = newEllipsoid(place, rx, ry, rz,
            latSouth, latNorth, longWest, longEast, relPos)

    return cut(ellipsoid, innerEllipsoid)


# rz1 = bottom oval radius
# rz2 = top oval radius
def newOval(place_, rx, ry, rz1, rz2,
        latSouth = -90, latNorth=90, longWest=0, longEast=360,
        relPos = fc.Vector(0,0,0)):

    if relPos[2] < 0:
        z = -relPos[2]*rz1
    else:
        z = -relPos[2]*rz2

    diff = ezPlacement(-rx*relPos[0], -ry*relPos[1], z)

    place = foldPlacements([place_, diff])
    ellipsoids = []
    if latNorth > 0:
        ellipsoids.append(newEllipsoid(place, rx, ry, rz2, 0, latNorth, longWest, longEast))
    if latSouth < 0:
        ellipsoids.append(newEllipsoid(place, rx, ry, rz1, latSouth, 0, longWest, longEast))

    oval = fuse(ellipsoids)
    return oval


def newHollowOval(place, innerRx, rx, ry, rz1, rz2,
        latSouth=-90, latNorth=90, longWest=0, longEast=360,
        relPos = fc.Vector(0,0,0)):
    a = innerRx/float(rx)
    aInv = 1.0/a
    innerOval = newOval(place, innerRx, a*ry, a*rz1, a*rz2,
            latSouth, latNorth, longWest, longEast, relPos*a)
    oval = newOval(place, rx, ry, rz1, rz2,
            latSouth, latNorth, longWest, longEast, relPos)
    return cut(oval, innerOval)


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


def loft(name, sketches):
    obj = newObject("Part::Loft", name)
    obj.Sections = sketches
    obj.Solid = True
    obj.Ruled = False
    obj.Closed = False
    return obj


def sketchToBox(baseName, planeOrigin, planeOrientation, sketchVectorList, dirVector):
    plane  = newPlane(baseName+"Plane", planeOrigin, planeOrientation)

    sketch = newSketchOnPlane(baseName+"Sketch", plane)
    sketch = sketchClosedWire(sketch, sketchVectorList)

    extrusion = extrudeSketch(baseName+"Extrusion", sketch, dirVector)
    return extrusion


# Apply any boolean operation to the workpiece and an extruded sketch
def simpleBoolean(workpiece, baseName, boolOp, planeOrigin, planeOrientation, sketchVectorList,
        direction):
    extrusion = sketchToBox(baseName, planeOrigin, planeOrientation, sketchVectorList, direction)
    
    workpiece = boolOp(baseName, workpiece, extrusion)
    return workpiece


def rotateVector(v, rotation):
    matrix = fc.Placement(fc.Vector(0,0,0), rotation).toMatrix()

    vector = fc.Placement(v, fc.Rotation(0,0,0)).toMatrix()
    matrixOut = matrix*vector
    vectorOut = fc.Placement(matrixOut).Base
    return vectorOut


# This is done by rotating a vector
def rotationToNormal(rotation):
    vector = rotateVector(fc.Vector(0,0,1), rotation)
    vectorOut = vector.normalize()
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


def singleCopy(obj, placement):
    objCopy = newObject("Part::Feature")
    objCopy.Shape = obj.Shape
    objCopy.Placement = placement
    return objCopy
    

def manyCopies(obj, placements):
    objCopies = []
    for i in range(placements.__len__()):
        objCopy = singleCopy(obj, placements[i])
        objCopies.append(objCopy)

    return objCopies


def removeMany(baseName, workpiece, tool, placementList):
    objectCopies = manyCopies(tool, placementList)
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
    sweep = newObject('Part::Sweep')
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
    obj = Draft.downgrade(face)[0]
    if (len(obj) == 1):
        wire = obj[0]
    else:
        raise Exception("Error in faceToWire: downgrade failed")

    return wire


def facesToWires(faces):
    wires = []
    for face in faces:
        wire = faceToWire(face)
        wires.append(wire)
    return wires

 
def wirePairToRuledSurface(name, wire1, wire2):
    ruledSurface = newObject("Part::RuledSurface")
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
    shell = newObject("Part::Feature")
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
    solid = newObject("Part::Feature")
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

def edgesToFace(name, edges):
    recompute()
    shape = Part.makeFilledFace(Part.__sortEdges__(edges))
    if shape.isNull():
        raise Exception("Failed to create face")

    face = newObject("Part::Feature")
    face.Shape = shape
    return face

def facesToShell(name, faces):
    recompute()
    shape = Part.Shell(faces)
    if shape.isNull():
        raise Exception("Failed to create shell")
    shell = newObject("Part::Feature")
    shell.Shape = shape
    return shell


def cappedSweepToShell(name, sweep, cap1, cap2):
    faces = []
    faces.extend(sweep.Shape.Faces)
    faces.extend(cap1.Shape.Faces)
    faces.extend(cap2.Shape.Faces)

    shell = facesToShell(name, faces)
    return shell

def shellToSolid(name, shell):
    shape = shell.Shape
    if shape.ShapeType != 'Shell':
        raise Exception("Part is not a shell")

    solidShape = Part.Solid(shape)
    if solidShape.isNull():
        raise Exception("Failed to create shell")

    solid = newObject("Part::Feature")
    solid.Shape = solidShape
    return solid


def exportStl(objs, fname):
    if not isinstance(objs, list):
        raise Exception("input to exportStl() must be a list")
    Mesh.export(objs, os.path.abspath(fname))
    return





def extrudeFace(name, face, dirVector):
    wire = faceToWire(face)
    solid = extrudeSketch(name, wire, dirVector)
    return solid


def genObjects(placements, fn, isFused = True):
    solids = []
    for p in placements:
        solids.append(fn(p))

    if (isFused):
        return fuse(solids)
    else:
        return solids


# apply a series of fuses and cuts to solids
# returns a solid
# the component functions must return solids as well
# the components dict acts like a cache
def generateConfiguration(componentI, configurations, params, components = {}):
    solid = None
    component = None

    if isinstance(componentI, dict):
        component = componentI # TODO: does this really happen?
    else:
        component = configurations[componentI]

    if not isinstance(componentI, dict) and componentI in components: 
        solid = components[componentI]
    if isinstance(component, dict):
        if not len(component) == 1:
            raise Exception("Configuration dict can only have one entry\n")
        elif "fuse" in component:
            solids = []
            for compId in component["fuse"]:
                solids.append(generateConfiguration(compId, configurations, params, components))
            solid = fuse(solids)
        elif "cut" in component:
            compIds = component["cut"]
            solid1 = generateConfiguration(compIds[0], configurations, params, components)
            solid2 = generateConfiguration(compIds[1], configurations, params, components)
            solid = cut(solid1, solid2)

        elif "pattern" in component:
            placements = component["pattern"][0]
            baseId = component["pattern"][1]
            baseSolid = generateConfiguration(baseId, configurations, params, components)
            copiedSolids = manyCopies(baseSolid, placements)
            solid = fuse(copiedSolids)
        else:
            op = ""
            for key in component:
                op = key
            raise Exception("Operation \n" + op + "\" not recognized")

        recompute()
        if not isinstance(componentI, dict):
            if not componentI in components:
                components[componentI] = solid
            if "intermStlPrefix" in params:
                exportStl([solid], params["intermStlPrefix"] + str(componentI) + ".stl")

    else: # needs to be a generating function
        if componentI in components:
            solid = components[componentI]
        else:
            solid = component(params)
            components[componentI] = solid
            recompute()
            if "intermStlPrefix" in params:
                exportStl([solid], params["intermStlPrefix"] + str(componentI) + ".stl")

    return solid
