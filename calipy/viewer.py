import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import math
import numpy as np
import numpy.matlib
import os
import json
import cv2

#             Z
#           /
#        /
#     /
#    ---------- X
#    |
#    |
#    |
#    Y

class vtkRenderer():
    def __init__(self, widget=None):
        self.ren = vtk.vtkRenderer()

        if widget is not None:
            # Qt Widget Mode
            self.qtwidget_mode = True
            
            #### Init
            # self.vtkWidget = QVTKRenderWindowInteractor(self.centralwidget)
            # self.vtkWidget.setGeometry(0,0,200,200)
            # self.vtkRenderer = calipy.vtkRenderer(self.vtkWidget)

            # Qt Widget
            self.vtkWidget = widget
            self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
            self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
            self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.iren.Initialize()
            self.iren.Start()
        else:
            # Window Mode
            self.qtwidget_mode = False

            # Make empty window
            self.renWin = vtk.vtkRenderWindow()
            self.renWin.AddRenderer(self.ren)
            self.renWin.SetSize(960, 540)

            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.iren.SetRenderWindow(self.renWin)
            self.iren.Initialize()

        self.ren.SetBackground(0, 0.1, 0)

        self.actor_list = {}

        axes = vtk.vtkAxesActor()
        self.ren.AddActor(axes)
        self.actor_list["axes"] = axes
        self.ren.ResetCamera()

        self.iren.AddObserver('LeftButtonPressEvent', self.pushLeftButtonPressEventOnVTK, 1.0)

    # Add Event for get Position
    def pushLeftButtonPressEventOnVTK(self, obj, ev):
        clickPos = self.iren.GetEventPosition()
        #print(clickPos)
        picker = vtk.vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.ren)
        print(picker.GetPickPosition())


    def setMainCamera(self, R = np.eye(3), t = np.zeros((3,1)), fov = 80):
        camera = vtk.vtkCamera()
        camera.SetPosition(t[0,0],t[1,0],t[2,0])
        #camera.SetFocalPoint(0,1,0)
        focalpoint = np.array([[0],[0],[1]])
        focalpoint = np.dot(R,focalpoint) + t
        camera.SetFocalPoint(focalpoint[0],focalpoint[1],focalpoint[2])
        ref = np.array([[0],[-1],[0]])
        cam_up = np.dot(R, ref)
        #camera.SetPosition(0,1,0)
        #camera.SetViewUp(0,1,0)
        camera.SetViewUp(cam_up[0],cam_up[1],cam_up[2])
        camera.SetViewAngle(fov)
        self.ren.SetActiveCamera(camera)

    def setMainCameraToSeeTarget(self, t = np.zeros((3,1)), target = np.zeros((3,1)), fov = 80):
        camera = vtk.vtkCamera()
        camera.SetPosition(t[0,0],t[1,0],t[2,0])
        #print("Position :", t)
        #camera.SetFocalPoint(0,1,0)
        #focalpoint = np.array([[0],[0],[1]])
        #focalpoint = np.dot(R,focalpoint) + t
        target_focalpoint = (target - t).ravel()
        #print(target_focalpoint)
        target_focalpoint = target_focalpoint / np.linalg.norm(target_focalpoint)
        #print("focalpoint", target)
        camera.SetFocalPoint(target[0],target[1],target[2])
        ref = np.array([[0],[-1],[0]]).ravel()
        #print(focalpoint, ref)
        ref_right = np.cross(target_focalpoint, ref)
        ref_right = ref_right / np.linalg.norm(ref_right)
        #print(ref_right, focalpoint)
        cam_up = np.cross(ref_right, target_focalpoint)
        cam_up = cam_up / np.linalg.norm(cam_up)
        print("Up",cam_up)
        #cam_up = np.dot(R, ref)
        #camera.SetPosition(0,1,0)
        #camera.SetViewUp(0,1,0)
        camera.SetViewUp(cam_up[0],cam_up[1],cam_up[2])
        camera.SetViewAngle(fov)
        self.ren.SetActiveCamera(camera)

    def getActorList(self):
        return self.actor_list.keys()

    def removeActorByName(self, name):
        #print(self.actor_list)
        if name in self.actor_list.keys():
            actor = self.actor_list.pop(name)
            self.ren.RemoveActor(actor)
            #print("remove! ", name)
            
    def addText(self, name, text, pos_x, pos_y):
        self.removeActorByName(name)
        textActor = vtk.vtkTextActor()
        textActor.SetInput( text )
        textActor.SetPosition( pos_x, pos_y )
        textActor.GetTextProperty().SetFontSize ( 50 )
        textActor.GetTextProperty().SetColor ( 1.0, 1.0, 1.0 )
        self.ren.AddActor2D(textActor)
        self.actor_list[name] = textActor
    
    def addPlane(self, name, point1, point2, point3, color=np.array([255.0,255.0,255.0]), opacity=1.0):
        self.removeActorByName(name)

        # Create a plane
        planeSource = vtk.vtkPlaneSource()
        # planeSource.SetOrigin(center_point[0], center_point[1], center_point[2])
        # #planeSource.SetNormal(normal_vector[0], normal_vector[1], normal_vector[2])
        # #print(dir(planeSource))
        # planeSource.SetPoint1(top_left_point[0], top_left_point[1], top_left_point[2])
        # planeSource.SetPoint2(bot_right_point[0], bot_right_point[1], bot_right_point[2])
        # planeSource.SetXResolution(10)
        # planeSource.SetYResolution(340)
        planeSource.SetOrigin(point1[0], point1[1], point1[2])
        planeSource.SetPoint1(point2[0], point2[1], point2[2])
        planeSource.SetPoint2(point3[0], point3[1], point3[2])
        planeSource.SetXResolution(10)
        planeSource.SetYResolution(340)

        planeSource.Update()

        plane = planeSource.GetOutput()

        # Create a mapper and actor
        polygonMapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            polygonMapper.SetInputConnection(polygon.GetProducerPort())
        else:
            polygonMapper.SetInputData(plane)
            polygonMapper.Update()

        polygonActor = vtk.vtkActor()
        polygonActor.SetMapper(polygonMapper)
        polygonActor.GetProperty().SetColor([color[0],color[1],color[2]])
        polygonActor.GetProperty().SetOpacity(opacity)
        #actor.GetProperty().SetColor(colors->GetColor3d("Cyan").GetData());

        self.ren.AddActor(polygonActor)
        self.actor_list[name] = polygonActor

    def addPlanWithTexture(self, name, point1, point2, point3, path, opacity=1.0):
        self.removeActorByName(name)

        #png_file = vtk.vtkPNGReader()
        #print(png_file.CanReadFile(path))

        # Read the image which will be the texture
        #vtkSmartPointer<vtkJPEGReader> jPEGReader = vtkSmartPointer<vtkJPEGReader>::New();
        #jPEGReader->SetFileName ( inputFilename.c_str() );
        img = vtk.vtkJPEGReader()
        img.SetFileName(path)
        
        #print(img.CanReadFile(path))
        #print(path)

        # Create a plane
        #vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
        #plane->SetCenter(0.0, 0.0, 0.0);
        #plane->SetNormal(0.0, 0.0, 1.0);
        plane = vtk.vtkPlaneSource()
        # planeSource.SetOrigin(center_point[0], center_point[1], center_point[2])
        # #planeSource.SetNormal(normal_vector[0], normal_vector[1], normal_vector[2])
        # #print(dir(planeSource))
        # planeSource.SetPoint1(top_left_point[0], top_left_point[1], top_left_point[2])
        # planeSource.SetPoint2(bot_right_point[0], bot_right_point[1], bot_right_point[2])
        # planeSource.SetXResolution(10)
        # planeSource.SetYResolution(340)
        #plane.SetCenter(0.0,0.0,0.0)
        #plane.SetNormal(0.0,0.0,1.0)
        plane.SetOrigin(point1[0], point1[1], point1[2])
        plane.SetPoint1(point2[0], point2[1], point2[2])
        plane.SetPoint2(point3[0], point3[1], point3[2])
        plane.SetXResolution(1920)
        plane.SetYResolution(1080)

        # Apply the texture
        #vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
        #texture->SetInputConnection(jPEGReader->GetOutputPort());
        texture = vtk.vtkTexture()
        texture.SetInputConnection(img.GetOutputPort())

        #vtkSmartPointer<vtkTextureMapToPlane> texturePlane = vtkSmartPointer<vtkTextureMapToPlane>::New();
        #texturePlane->SetInputConnection(plane->GetOutputPort());
        texturePlane = vtk.vtkTextureMapToPlane()
        texturePlane.SetInputConnection(plane.GetOutputPort())

        #planeSource.Update()
        #plane = planeSource.GetOutput()

        #vtkSmartPointer<vtkPolyDataMapper> planeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        #planeMapper->SetInputConnection(texturePlane->GetOutputPort());
        planeMapper = vtk.vtkPolyDataMapper()
        planeMapper.SetInputConnection(texturePlane.GetOutputPort())

        #vtkSmartPointer<vtkActor> texturedPlane = vtkSmartPointer<vtkActor>::New();
        #texturedPlane->SetMapper(planeMapper);
        #texturedPlane->SetTexture(texture);
        texturedPlane = vtk.vtkActor()
        texturedPlane.SetMapper(planeMapper)
        texturedPlane.SetTexture(texture)

        # Create a mapper and actor
        #polygonMapper = vtk.vtkPolyDataMapper()
        #if vtk.VTK_MAJOR_VERSION <= 5:
        #    polygonMapper.SetInputConnection(texturePlane.GetProducePort())
        #else:
        #    polygonMapper.SetInputData(texturePlane.GetOutput())
        #    polygonMapper.Update()

        #polygonActor = vtk.vtkActor()
        #polygonActor.SetMapper(polygonMapper)
        #polygonActor.SetTexture(texture)
        #polygonActor.GetProperty().SetColor([color[0],color[1],color[2]])
        #polygonActor.GetProperty().SetOpacity(opacity)
        #actor.GetProperty().SetColor(colors->GetColor3d("Cyan").GetData());

        self.ren.AddActor(texturedPlane)
        self.actor_list[name] = texturedPlane

    def addLines(self, name, points, idx_list = None, line_width = 1, color=np.array([255.0,255.0,255.0])): # points => numpy vector [3, 0~n]
        self.removeActorByName(name)
        vtkpoints = vtk.vtkPoints()
        vtklines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)

        points_size = points.shape[0] 
        vtkpoints.SetNumberOfPoints(points_size)
        for idx, point in enumerate(points):
            vtkpoints.SetPoint(idx, point[0], point[1], point[2])
            colors.InsertNextTuple(color)
        colors.SetName(name+"_colors")

        if idx_list is None:
            vtklines.InsertNextCell(points_size)
            for idx in range(points_size):
                vtklines.InsertCellPoint(idx)
        else:
            vtklines.InsertNextCell(len(idx_list))
            for idx in idx_list:
                vtklines.InsertCellPoint(idx)

        polygon = vtk.vtkPolyData()
        polygon.SetPoints(vtkpoints)
        polygon.SetLines(vtklines)
        polygon.GetCellData().SetScalars(colors)

        polygonMapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            polygonMapper.SetInputConnection(polygon.GetProducerPort())
        else:
            polygonMapper.SetInputData(polygon)
            polygonMapper.Update()

        polygonActor = vtk.vtkActor()
        polygonActor.SetMapper(polygonMapper)
        polygonActor.GetProperty().SetLineWidth(line_width)

        self.ren.AddActor(polygonActor)
        self.actor_list[name] = polygonActor

    def addCamera(self, name, R = np.eye(3), t = np.zeros((3,1)), cs = 0.1, line_width = 2, color=np.array([255,255,255])):
        self.removeActorByName(name)
        camera_points = np.zeros((12,3))
        camera_points[0,:] = np.array([-cs/2, -cs/2, cs])
        camera_points[1] = np.array([ cs/2, -cs/2, cs])
        camera_points[2] = np.array([-cs/2,  cs/2, cs])
        camera_points[3] = np.array([ cs/2,  cs/2, cs])
        camera_points[4] = np.array([-cs/4, -cs/4, cs/2])
        camera_points[5] = np.array([ cs/4, -cs/4, cs/2])
        camera_points[6] = np.array([-cs/4,  cs/4, cs/2])
        camera_points[7] = np.array([ cs/4,  cs/4, cs/2])
        camera_points[8] = np.array([-cs/4, -cs/4, 0])
        camera_points[9] = np.array([ cs/4, -cs/4, 0])
        camera_points[10] = np.array([-cs/4,  cs/4, 0])
        camera_points[11] = np.array([ cs/4,  cs/4, 0])

        camera_points = np.transpose(camera_points)
        camera_points = np.dot(R, camera_points) + np.matlib.repmat(t, 1, camera_points.shape[1])
        camera_points = np.transpose(camera_points)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(12)
        colors = vtk.vtkUnsignedCharArray()
        points.SetNumberOfPoints(12)
        colors.SetNumberOfComponents(3)

        for idx, point in enumerate(camera_points):
            points.SetPoint(idx, point[0], point[1], point[2])
            colors.InsertNextTuple(color)
        colors.SetName(name+"_colors")

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(24)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        lines.InsertCellPoint(3)
        lines.InsertCellPoint(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(4)
        lines.InsertCellPoint(5)
        lines.InsertCellPoint(7)
        lines.InsertCellPoint(6)
        lines.InsertCellPoint(4)
        lines.InsertCellPoint(8)
        lines.InsertCellPoint(9)
        lines.InsertCellPoint(11)
        lines.InsertCellPoint(10)
        lines.InsertCellPoint(8)
        lines.InsertCellPoint(9)
        lines.InsertCellPoint(5)
        lines.InsertCellPoint(1)
        lines.InsertCellPoint(3)
        lines.InsertCellPoint(7)
        lines.InsertCellPoint(11)
        lines.InsertCellPoint(10)
        lines.InsertCellPoint(6)
        lines.InsertCellPoint(2)

        polygon = vtk.vtkPolyData()
        polygon.SetPoints(points)
        polygon.SetLines(lines)
        polygon.GetCellData().SetScalars(colors)

        polygonMapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            polygonMapper.SetInputConnection(polygon.GetProducerPort())
        else:
            polygonMapper.SetInputData(polygon)
            polygonMapper.Update()

        polygonActor = vtk.vtkActor()
        polygonActor.SetMapper(polygonMapper)
        polygonActor.GetProperty().SetPointSize(0.1)
        polygonActor.GetProperty().SetLineWidth(line_width)
        self.ren.AddActor(polygonActor)
        self.actor_list[name] = polygonActor

    def drawPoints(self, name, point_list, input_color=np.array([[255,0,0]]), point_size = 2):
        self.removeActorByName(name)
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        #colors.SetName("Colors")
        #colors.SetNumberOfComponents(3)

        if input_color.shape[0] == 1:
            color_list = np.ones(point_list.shape) * input_color[0]
        else:
            color_list = input_color

        for point, color in zip(point_list, color_list):
            id = points.InsertNextPoint(point.tolist())
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)
            colors.InsertNextTuple(color)

        point = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(points)
        point.SetVerts(vertices)
        point.GetPointData().SetScalars(colors)

        polygonMapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            polygonMapper.SetInputConnection(ps.GetProducerPort())
        else:
            polygonMapper.SetInputData(point)
            polygonMapper.Update()
        polygonActor = vtk.vtkActor()
        polygonActor.SetMapper(polygonMapper)
        polygonActor.GetProperty().SetPointSize(point_size)
        self.ren.AddActor(polygonActor)
        self.actor_list[name] = polygonActor

    def render(self):
        self.iren.Render()
        if self.qtwidget_mode == False:
            self.iren.Start()

if __name__ == "__main__":
    window_width = 1.18
    window_height = 0.75
    window_points = [[-window_width/2, -window_height*math.cos((5.0/180.0) * math.pi), -window_height*math.sin((5.0/180.0) * math.pi)],
                     [ window_width/2, -window_height*math.cos((5.0/180.0) * math.pi), -window_height*math.sin((5.0/180.0) * math.pi)],
                     [-window_width/2, 0, 0],
                     [ window_width/2, 0, 0]]
    index = np.array([0,1,3,2,0])

    ren = vtkRenderer()
    ren.addLines(np.transpose(window_points), index)
    ren.showImage()