import argparse
import ast
import codecs
import json
import numpy as np 
import os.path
import subprocess 
import platform 
import sys
import xlrd
import glob 
from functools import partial 
from PIL import Image   

from PyQt5.QtCore import QSize, Qt, QPoint, QByteArray, QTimer, QFileInfo, QPointF, QProcess
from PyQt5.QtGui import QImage, QCursor, QPixmap, QImageReader
from PyQt5.QtWidgets import QMainWindow, QListWidget, QVBoxLayout, QToolButton, QHBoxLayout, QDockWidget, QWidget, \
    QSlider, QGraphicsOpacityEffect, QMessageBox, QListView, QScrollArea, QWidgetAction, QApplication, QLabel, QGridLayout, \
    QFileDialog, QListWidgetItem, QComboBox, QDialog

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
sys.path.append("..")
sys.path.insert(0, './')

from ocr import PaddleOCR 
from libs.constants import *
from libs.utils import *
from libs.labelColor import label_colormap
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR, DEFAULT_LOCK_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.autoDialog import AutoDialog
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem
from libs.editinlist import EditInList
from libs.unique_label_qlist_widget import UniqueLabelQListWidget
from libs.keyDialog import KeyDialog

from libs.augment import ( 
    Contrast, Brightness, JpegCompression, Pixelate, 
    GlassBlur, GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise,
    AutoContrast, Sharpness, Color, Stretch, Distort, 
    Fog, Frost, Snow, Rain, Shadow, Curve, Distortion_Sin, Distortion_Cos   
)  


__appname__ = 'labelMaker'

LABEL_COLORMAP = label_colormap()


class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self,
                 lang="ko",
                 gpu=True,
                 kie_mode=True,
                 default_filename=None,
                 default_predefined_class_file=None,
                 default_save_dir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        self.setWindowState(Qt.WindowMaximized)  # set window max
        self.activateWindow()  # PPOCRLabel goes to the front when activate

        lang = 'ko'
        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings
        self.lang = lang

        # Load string bundle for i18n
        if lang not in ['ko', 'ch', 'en']:
            lang = 'en'

        if lang =='ch': locale = 'zh-CN'
        elif lang =='ko': locale = 'ko'
        else: locale = 'en'

        self.stringBundle = StringBundle.getBundle(localeStr=locale) # 'en'
        getStr = lambda strId: self.stringBundle.getString(strId)


        # KIE setting
        self.kie_mode = kie_mode
        self.key_previous_text = ""
        self.existed_key_cls_set = set()
        self.key_dialog_tip = getStr('keyDialogTip')

        self.defaultSaveDir = default_save_dir
        self.ocr = PaddleOCR(use_angle_cls=False,
                             det=True,
                             cls=False,
                             use_gpu=True,
                             lang=lang,
                             show_log=False) 

        # For loading all image under a directory
        self.mImgList = []
        self.mImgList5 = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None
        self.result_dic = []
        self.result_dic_locked = []
        self.changeFileFolder = False
        self.haveAutoReced = False
        self.labelFile = None
        self.currIndex = 0

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = ""

        # Load predefined classes to the list
        self.loadPredefinedClasses(default_predefined_class_file)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)
        self.autoDialog = AutoDialog(parent=self)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.itemsToShapesbox = {}
        self.shapesToItemsbox = {}
        self.prevLabelText = getStr('tempLabel')
        self.noLabelText = getStr('nullLabel')
        self.model = 'paddle'
        self.PPreader = None
        self.autoSaveNum = 5

        #  ================== File List  ==================

        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemClicked.connect(self.fileitemDoubleClicked)
        self.fileListWidget.setIconSize(QSize(25, 25))
        filelistLayout.addWidget(self.fileListWidget)

        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.fileListName = getStr('fileList')
        self.fileDock = QDockWidget(self.fileListName, self)
        self.fileDock.setObjectName(getStr('files'))
        self.fileDock.setWidget(fileListContainer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.fileDock)

        #  ================== Key List  ==================
        if self.kie_mode:
            self.keyList = UniqueLabelQListWidget()

            # set key list height
            key_list_height = int(QApplication.desktop().height() // 4)
            if key_list_height < 50:
                key_list_height = 50
            self.keyList.setMaximumHeight(key_list_height)

            self.keyListDockName = getStr('keyListTitle')
            self.keyListDock = QDockWidget(self.keyListDockName, self)
            self.keyListDock.setWidget(self.keyList)
            self.keyListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
            filelistLayout.addWidget(self.keyListDock)

        self.AutoRecognition = QToolButton()
        self.AutoRecognition.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.AutoRecognition.setIcon(newIcon('Auto'))

        self.AutoCurRecognition = QToolButton()
        self.AutoCurRecognition.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.AutoCurRecognition.setIcon(newIcon('Auto'))

        autoRecLayout = QHBoxLayout()
        autoRecLayout.setContentsMargins(0, 0, 0, 0)
        autoRecLayout.addWidget(self.AutoRecognition)
        autoRecLayout.addWidget(self.AutoCurRecognition) 
        autoRecContainer = QWidget()
        autoRecContainer.setLayout(autoRecLayout)
        filelistLayout.addWidget(autoRecContainer) 

        #  ================== Right Area  ==================
        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Buttons
        self.editButton = QToolButton()
        self.reRecogButton = QToolButton()
        self.reRecogButton.setIcon(newIcon('reRec', 30))
        self.reRecogButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # self.tableRecButton = QToolButton()
        # self.tableRecButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.newButton = QToolButton()
        self.newButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.createpolyButton = QToolButton()
        self.createpolyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.SaveButton = QToolButton()
        self.SaveButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.DelButton = QToolButton()
        self.DelButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        leftTopToolBox = QGridLayout()
        leftTopToolBox.addWidget(self.newButton, 0, 0, 1, 1)
        leftTopToolBox.addWidget(self.createpolyButton, 0, 1, 1, 1)
        leftTopToolBox.addWidget(self.reRecogButton, 1, 0, 1, 1)
        # leftTopToolBox.addWidget(self.tableRecButton, 1, 1, 1, 1)

        leftTopToolBoxContainer = QWidget()
        leftTopToolBoxContainer.setLayout(leftTopToolBox)
        listLayout.addWidget(leftTopToolBoxContainer)

        #  ================== Label List  ==================
        # Create and add a widget for showing current label items
        self.labelList = EditInList()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.clicked.connect(self.labelList.item_clicked)

        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelListDockName = getStr('recognitionResult')
        self.labelListDock = QDockWidget(self.labelListDockName, self)
        self.labelListDock.setWidget(self.labelList)
        self.labelListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        listLayout.addWidget(self.labelListDock)

        #  ================== Detection Box  ==================
        self.BoxList = QListWidget()

        # self.BoxList.itemActivated.connect(self.boxSelectionChanged)
        self.BoxList.itemSelectionChanged.connect(self.boxSelectionChanged)
        self.BoxList.itemDoubleClicked.connect(self.editBox)
        # Connect to itemChanged to detect checkbox changes.
        self.BoxList.itemChanged.connect(self.boxItemChanged)
        self.BoxListDockName = getStr('detectionBoxposition')
        self.BoxListDock = QDockWidget(self.BoxListDockName, self)
        self.BoxListDock.setWidget(self.BoxList)
        self.BoxListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        listLayout.addWidget(self.BoxListDock)

        #  ================== Lower Right Area  ==================
        leftbtmtoolbox = QHBoxLayout()
        leftbtmtoolbox.addWidget(self.SaveButton)
        leftbtmtoolbox.addWidget(self.DelButton)
        leftbtmtoolboxcontainer = QWidget()
        leftbtmtoolboxcontainer.setLayout(leftbtmtoolbox)
        listLayout.addWidget(leftbtmtoolboxcontainer)

        self.dock = QDockWidget(getStr('boxLabelText'), self)
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        #  ================== Zoom Bar  ==================
        self.imageSlider = QSlider(Qt.Horizontal)
        self.imageSlider.valueChanged.connect(self.CanvasSizeChange)
        self.imageSlider.setMinimum(-9)
        self.imageSlider.setMaximum(510)
        self.imageSlider.setSingleStep(1)
        self.imageSlider.setTickPosition(QSlider.TicksBelow)
        self.imageSlider.setTickInterval(1)

        op = QGraphicsOpacityEffect()
        op.setOpacity(0.2)
        self.imageSlider.setGraphicsEffect(op)

        self.imageSlider.setStyleSheet("background-color:transparent")
        self.imageSliderDock = QDockWidget(getStr('ImageResize'), self)
        self.imageSliderDock.setObjectName(getStr('IR'))
        self.imageSliderDock.setWidget(self.imageSlider)
        self.imageSliderDock.setFeatures(QDockWidget.DockWidgetFloatable)
        self.imageSliderDock.setAttribute(Qt.WA_TranslucentBackground)
        self.addDockWidget(Qt.RightDockWidgetArea, self.imageSliderDock)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)
        self.zoomWidgetValue = self.zoomWidget.value()

        self.msgBox = QMessageBox()

        #  ================== Thumbnail ==================
        hlayout = QHBoxLayout()
        m = (0, 0, 0, 0)
        hlayout.setSpacing(0)
        hlayout.setContentsMargins(*m)
        self.preButton = QToolButton()
        self.preButton.setIcon(newIcon("prev", 40))
        self.preButton.setIconSize(QSize(40, 100))
        self.preButton.clicked.connect(self.openPrevImg)
        self.preButton.setStyleSheet('border: none;')
        self.preButton.setShortcut('a')
        self.iconlist = QListWidget()
        self.iconlist.setViewMode(QListView.IconMode)
        self.iconlist.setFlow(QListView.TopToBottom)
        self.iconlist.setSpacing(10)
        self.iconlist.setIconSize(QSize(50, 50))
        self.iconlist.setMovement(QListView.Static)
        self.iconlist.setResizeMode(QListView.Adjust)
        self.iconlist.itemClicked.connect(self.iconitemDoubleClicked)
        self.iconlist.setStyleSheet("QListWidget{ background-color:transparent; border: none;}")
        self.iconlist.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nextButton = QToolButton()
        self.nextButton.setIcon(newIcon("next", 40))
        self.nextButton.setIconSize(QSize(40, 100))
        self.nextButton.setStyleSheet('border: none;')
        self.nextButton.clicked.connect(self.openNextImg)
        self.nextButton.setShortcut('d')

        hlayout.addWidget(self.preButton)
        hlayout.addWidget(self.iconlist)
        hlayout.addWidget(self.nextButton)

        iconListContainer = QWidget()
        iconListContainer.setLayout(hlayout)
        iconListContainer.setFixedHeight(100)

        #  ================== Canvas ==================
        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(partial(self.newShape, False))
        self.canvas.shapeMoved.connect(self.updateBoxlist)  # self.setDirty
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        centerLayout = QVBoxLayout()
        centerLayout.setContentsMargins(0, 0, 0, 0)
        centerLayout.addWidget(scroll)
        centerLayout.addWidget(iconListContainer, 0, Qt.AlignCenter)
        centerContainer = QWidget()
        centerContainer.setLayout(centerLayout)

        self.setCentralWidget(centerContainer)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        self.dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable)
        self.fileDock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        #  ================== Actions ==================
        action = partial(newAction, self)
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        opendir = action(getStr('openDir'), self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        open_dataset_dir = action(getStr('openDatasetDir'), self.openDatasetDirDialog,
                                  'Ctrl+p', 'open', getStr('openDatasetDir'), enabled=False)

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+V', 'verify', getStr('saveDetail'), enabled=False)

        alcm = action(getStr('choosemodel'), self.autolcm,
                      'Ctrl+M', 'next', getStr('tipchoosemodel')) 

        # parent, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False, enabled=True                       

        exportCropImg = action(getStr('exportCropImg'), self.exportCropImg, 
                    'Ctrl+Shift+D', 'save', getStr('exportCropImgDetail'), enabled=True)

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor,
                        'Ctrl+L', 'color_line', getStr('boxLineColorDetail'))

        createMode = action(getStr('crtBox'), self.setCreateMode,
                            'w', 'new', getStr('crtBoxDetail'), enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action(getStr('crtBox'), self.createShape,
                        'w', 'objects', getStr('crtBoxDetail'), enabled=False)

        delete = action(getStr('delBox'), self.deleteSelectedShape,
                        'backspace', 'delete', getStr('delBoxDetail'), enabled=False)

        copy = action(getStr('dupBox'), self.copySelectedShape,
                      'Ctrl+C', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        hideAll = action(getStr('hideBox'), partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action(getStr('showBox'), partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))
        showSteps = action(getStr('steps'), self.showStepsDialog, None, 'help', getStr('steps'))
        showKeys = action(getStr('keys'), self.showKeysDialog, None, 'help', getStr('keys'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        #  ================== New Actions ==================

        edit = action(getStr('editLabel'), self.editLabel,
                      'Ctrl+E', 'edit', getStr('editLabelDetail'), enabled=False)

        AutoRec = action(getStr('autoRecognition'), self.autoRecognition,
                         '', 'Auto', getStr('autoRecognition'), enabled=False)
        
        AutoCurRec = action(getStr('autoCurRecognition'), self.autoCurRecognition,
                         '', 'Auto', getStr('autoCurRecognition'), enabled=False)

        reRec = action(getStr('reRecognition'), self.reRecognition,
                       'Ctrl+Shift+R', 'reRec', getStr('reRecognition'), enabled=False)

        singleRere = action(getStr('singleRe'), self.singleRerecognition,
                            'Ctrl+R', 'reRec', getStr('singleRe'), enabled=False) 
        
        singleExport = action(getStr('singleExport'), self.exportSingleImg,
                            '', 'save', getStr('singleExport'), enabled=False) 

        createpoly = action(getStr('creatPolygon'), self.createPolygon,
                            'q', 'new', getStr('creatPolygon'), enabled=False)
        
        # tableRec = action(getStr('TableRecognition'), self.TableRecognition,
        #                 '', 'Auto', getStr('TableRecognition'), enabled=False)

        # cellreRec = action(getStr('cellreRecognition'), self.cellreRecognition,
        #                 '', 'reRec', getStr('cellreRecognition'), enabled=False) 
        

        exportDet = action(getStr('exportDet'), self.exportDet,
                            '', 'save', getStr('exportDet'), enabled=False)
        exportRec = action(getStr('exportRec'), self.exportRec,
                         '', 'save', getStr('exportRec'), enabled=False)
        
        exportKie = action(getStr('exportKie'), self.exportKie,
                            '', 'save', getStr('exportKie'), enabled=False)
        

        exportDetMM = action(getStr('exportDetMM'), self.exportDetMM,
                            '', 'save', getStr('exportDetMM'), enabled=False)
        exportRecMM = action(getStr('exportRecMM'), self.exportRecMM,
                         '', 'save', getStr('exportRecMM'), enabled=False)
        
        exportKieMM = action(getStr('exportKieMM'), self.exportKieMM,
                            '', 'save', getStr('exportKieMM'), enabled=False)
        
        makeRecData = action(getStr('makeRecData'), self.makeRecData,
                            '', 'save', getStr('makeRecData'), enabled=False)
        
        saveLabel = action(getStr('saveLabel'), self.saveLabelFile,  #
                           'Ctrl+S', 'save', getStr('saveLabel'), enabled=False) 
        

        undoLastPoint = action(getStr("undoLastPoint"), self.canvas.undoLastPoint,
                               'Ctrl+Z', "undo", getStr("undoLastPoint"), enabled=False)

        rotateLeft = action(getStr("rotateLeft"), partial(self.rotateImgAction, 1),
                            'Ctrl+Alt+L', "rotateLeft", getStr("rotateLeft"), enabled=False)

        rotateRight = action(getStr("rotateRight"), partial(self.rotateImgAction, -1),
                             'Ctrl+Alt+R', "rotateRight", getStr("rotateRight"), enabled=False)

        undo = action(getStr("undo"), self.undoShapeEdit,
                      'Ctrl+Z', "undo", getStr("undo"), enabled=False)

        change_cls = action(getStr("keyChange"), self.change_box_key,
                            'Ctrl+X', "edit", getStr("keyChange"), enabled=False)

        lock = action(getStr("lockBox"), self.lockSelectedShape,
                      None, "lock", getStr("lockBoxDetail"), enabled=False)

        self.editButton.setDefaultAction(edit)
        self.newButton.setDefaultAction(create)
        self.createpolyButton.setDefaultAction(createpoly)
        self.DelButton.setDefaultAction(exportCropImg)
        self.SaveButton.setDefaultAction(save)
        self.AutoRecognition.setDefaultAction(AutoRec)
        self.AutoCurRecognition.setDefaultAction(AutoCurRec) 
        self.reRecogButton.setDefaultAction(reRec)
        # self.tableRecButton.setDefaultAction(tableRec)
        # self.preButton.setDefaultAction(openPrevImg)
        # self.nextButton.setDefaultAction(openNextImg)

        #  ================== Zoom layout ==================
        zoomLayout = QHBoxLayout()
        zoomLayout.addStretch()
        self.zoominButton = QToolButton()
        self.zoominButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.zoominButton.setDefaultAction(zoomIn)
        self.zoomoutButton = QToolButton()
        self.zoomoutButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.zoomoutButton.setDefaultAction(zoomOut)
        self.zoomorgButton = QToolButton()
        self.zoomorgButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.zoomorgButton.setDefaultAction(zoomOrg)
        zoomLayout.addWidget(self.zoominButton)
        zoomLayout.addWidget(self.zoomorgButton)
        zoomLayout.addWidget(self.zoomoutButton)

        zoomContainer = QWidget()
        zoomContainer.setLayout(zoomLayout)
        zoomContainer.setGeometry(0, 0, 30, 150)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))

        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction(getStr('drawSquares'), self)
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        # Store actions for further handling.
        self.actions = struct(save=save, resetAll=resetAll, exportCropImg=exportCropImg,
                              lineColor=color1, create=create, createpoly=createpoly, delete=delete, edit=edit, copy=copy,
                              exportRec=exportRec, singleRere=singleRere, singleExport=singleExport, 
                              AutoRec=AutoRec, AutoCurRec=AutoCurRec, 
                              reRec=reRec,  
                              createMode=createMode, editMode=editMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions, saveLabel=saveLabel, change_cls=change_cls,
                              undo=undo, undoLastPoint=undoLastPoint, open_dataset_dir=open_dataset_dir,
                              rotateLeft=rotateLeft, rotateRight=rotateRight, lock=lock, exportKie=exportKie, exportDet=exportDet, 
                              exportRecMM=exportRecMM, exportDetMM=exportDetMM, exportKieMM=exportKieMM, 
                              makeRecData=makeRecData, 
                              fileMenuActions=(opendir, open_dataset_dir, saveLabel, exportKie, exportDet, resetAll, quit),
                              beginner=(), advanced=(),
                              editMenu=(createpoly, edit, copy, delete, singleRere, singleExport, None, undo, undoLastPoint,
                                        None, rotateLeft, rotateRight, None, color1, self.drawSquaresOption, lock,
                                        None, change_cls),

                              beginnerContext=(
                                  create, createpoly, edit, copy, delete, singleRere,  
                                  rotateLeft, rotateRight, lock, singleExport, None, change_cls),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(create, createpoly, createMode, editMode),
                              onShapesPresent=(hideAll, showAll))

        # menus
        self.menus = struct(
            file=self.menu('&' + getStr('mfile')),
            edit=self.menu('&' + getStr('medit')),
            view=self.menu('&' + getStr('mview')),
            autolabel=self.menu('&OCR'),
            help=self.menu('&' + getStr('mhelp')),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        self.labelDialogOption = QAction(getStr('labelDialogOption'), self)
        self.labelDialogOption.setShortcut("Ctrl+Shift+L")
        self.labelDialogOption.setCheckable(True)
        self.labelDialogOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.labelDialogOption.triggered.connect(self.speedChoose)

        self.autoSaveOption = QAction(getStr('autoSaveMode'), self)
        self.autoSaveOption.setCheckable(True)
        self.autoSaveOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.autoSaveOption.triggered.connect(self.autoSaveFunc)

        addActions(self.menus.file,
                   (opendir, open_dataset_dir, 
                    None, saveLabel, exportDet, exportRec, exportKie, 
                    None, exportDetMM, exportRecMM, exportKieMM, 
                    None, makeRecData, 
                    None, self.autoSaveOption, 
                    None, resetAll, exportCropImg,
                    quit))

        addActions(self.menus.help, (showKeys, showSteps, showInfo))
        addActions(self.menus.view, (
            self.displayLabelOption, self.labelDialogOption,
            None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        addActions(self.menus.autolabel, (AutoRec, AutoCurRec, reRec, alcm, None, help))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(default_filename)
        self.lastOpenDir = None
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        # Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(1200, 800))

        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        # ADD:
        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        self.keyDialog = None

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(dirpath=self.filePath, silent=True)

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    def noShapes(self):
        return not self.itemsToShapes

    def populateModeActions(self):
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        self.menus.edit.clear()
        actions = (self.actions.create,)  # if self.beginner() else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)
        self.actions.createpoly.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.itemsToShapesbox.clear()  # ADD
        self.shapesToItemsbox.clear()
        self.labelList.clear()
        self.BoxList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        # self.comboBox.cb.clear()
        self.result_dic = []

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def currentBox(self):
        items = self.BoxList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def showInfoDialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def showStepsDialog(self):
        msg = stepsInfo(self.lang)
        QMessageBox.information(self, u'Information', msg)

    def showKeysDialog(self):
        msg = keysInfo(self.lang)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)
        self.actions.createpoly.setEnabled(False)
        self.canvas.fourpoint = False

    def createPolygon(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.canvas.fourpoint = True
        self.actions.create.setEnabled(False)
        self.actions.createpoly.setEnabled(False)
        self.actions.undoLastPoint.setEnabled(True)

    def rotateImg(self, filename, k, _value):
        self.actions.rotateRight.setEnabled(_value)
        pix = cv2.imread(filename)
        pix = np.rot90(pix, k)
        cv2.imwrite(filename, pix)
        self.canvas.update()
        self.loadFile(filename)

    def rotateImgWarn(self):
        self.msgBox.warning(self, "Warn", "\n The picture already has a label box, "
                                              "and rotation will disrupt the label. "
                                              "It is recommended to clear the label box and rotate it.")

    def rotateImgAction(self, k=1, _value=False):

        filename = self.mImgList[self.currIndex]

        if os.path.exists(filename):
            if self.itemsToShapesbox:
                self.rotateImgWarn()
            else:
                self.saveFile()
                self.dirty = False
                self.rotateImg(filename=filename, k=k, _value=True)
        else:
            self.rotateImgWarn()
            self.actions.rotateRight.setEnabled(False)
            self.actions.rotateLeft.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)
            self.actions.createpoly.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            # item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    # =================== detection box related functions ===================
    def boxItemChanged(self, item):
        shape = self.itemsToShapesbox[item]

        box = ast.literal_eval(item.text())
        # print('shape in labelItemChanged is',shape.points)
        if box != [(int(p.x()), int(p.y())) for p in shape.points]:
            # shape.points = box
            shape.points = [QPointF(p[0], p[1]) for p in box]

            # QPointF(x,y)
            # shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, True)  # item.checkState() == Qt.Checked

    def editBox(self):  # ADD
        if not self.canvas.editing():
            return
        item = self.currentBox()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())

        imageSize = str(self.image.size())
        width, height = self.image.width(), self.image.height()
        if text:
            try:
                text_list = eval(text)
            except:
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Please enter the correct format')
                msg_box.exec_()
                return
            if len(text_list) < 4:
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Please enter the coordinates of 4 points')
                msg_box.exec_()
                return
            for box in text_list:
                if box[0] > width or box[0] < 0 or box[1] > height or box[1] < 0:
                    msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Out of picture size')
                    msg_box.exec_()
                    return

            item.setText(text)
            # item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def updateBoxlist(self):
        self.canvas.selectedShapes_hShape = []
        if self.canvas.hShape != None:
            self.canvas.selectedShapes_hShape = self.canvas.selectedShapes + [self.canvas.hShape]
        else:
            self.canvas.selectedShapes_hShape = self.canvas.selectedShapes
        for shape in self.canvas.selectedShapes_hShape:
            item = self.shapesToItemsbox[shape]  # listitem
            text = [(int(p.x()), int(p.y())) for p in shape.points]
            item.setText(str(text))
        self.actions.undo.setEnabled(True)
        self.setDirty()

    def indexTo5Files(self, currIndex):
        if currIndex < 2:
            return self.mImgList[:5]
        elif currIndex > len(self.mImgList) - 3:
            return self.mImgList[-5:]
        else:
            return self.mImgList[currIndex - 2: currIndex + 3]

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        self.currIndex = self.mImgList.index(ustr(os.path.join(os.path.abspath(self.dirname), item.text())))
        filename = self.mImgList[self.currIndex]
        if filename:
            self.mImgList5 = self.indexTo5Files(self.currIndex)
            # self.additems5(None)
            self.loadFile(filename)

    def iconitemDoubleClicked(self, item=None):
        self.currIndex = self.mImgList.index(ustr(os.path.join(item.toolTip())))
        filename = self.mImgList[self.currIndex]
        if filename:
            self.mImgList5 = self.indexTo5Files(self.currIndex)
            # self.additems5(None)
            self.loadFile(filename)

    def CanvasSizeChange(self):
        if len(self.mImgList) > 0 and self.imageSlider.hasFocus():
            self.zoomWidget.setValue(self.imageSlider.value())

    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            self.shapesToItems[shape].setSelected(True)
            self.shapesToItemsbox[shape].setSelected(True)

        self.labelList.scrollToItem(self.currentItem())  # QAbstractItemView.EnsureVisible
        self.BoxList.scrollToItem(self.currentBox())

        if self.kie_mode:
            if len(self.canvas.selectedShapes) == 1 and self.keyList.count() > 0:
                selected_key_item_row = self.keyList.findItemsByLabel(self.canvas.selectedShapes[0].key_cls,
                                                                      get_row=True)
                if isinstance(selected_key_item_row, list) and len(selected_key_item_row) == 0:
                    key_text = self.canvas.selectedShapes[0].key_cls
                    item = self.keyList.createItemFromLabel(key_text)
                    self.keyList.addItem(item)
                    rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                    self.keyList.setItemLabel(item, key_text, rgb)
                    selected_key_item_row = self.keyList.findItemsByLabel(self.canvas.selectedShapes[0].key_cls,
                                                                          get_row=True)

                self.keyList.setCurrentRow(selected_key_item_row)

        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.singleRere.setEnabled(n_selected)
        self.actions.singleExport.setEnabled(n_selected) 
        # self.actions.cellreRec.setEnabled(n_selected)
        self.actions.delete.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)
        self.actions.lock.setEnabled(n_selected)
        self.actions.change_cls.setEnabled(n_selected)

    def addLabel(self, shape):
        shape.paintLabel = self.displayLabelOption.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked) if shape.difficult else item.setCheckState(Qt.Checked)
        # Checked means difficult is False
        # item.setBackground(generateColorByText(shape.label))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        # print('item in add label is ',[(p.x(), p.y()) for p in shape.points], shape.label)

        # ADD for box
        item = HashableQListWidgetItem(str([(int(p.x()), int(p.y())) for p in shape.points]))
        self.itemsToShapesbox[item] = shape
        self.shapesToItemsbox[shape] = item
        self.BoxList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self.updateComboBox()

        # update show counting
        self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
        self.labelListDock.setWindowTitle(self.labelListDockName + f" ({self.labelList.count()})")

    def remLabels(self, shapes):
        if shapes is None:
            # print('rm empty label')
            return
        for shape in shapes:
            item = self.shapesToItems[shape]
            self.labelList.takeItem(self.labelList.row(item))
            del self.shapesToItems[shape]
            del self.itemsToShapes[item]
            self.updateComboBox()

            # ADD:
            item = self.shapesToItemsbox[shape]
            self.BoxList.takeItem(self.BoxList.row(item))
            del self.shapesToItemsbox[shape]
            del self.itemsToShapesbox[item]
            self.updateComboBox()

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, key_cls, difficult in shapes:
            shape = Shape(label=label, line_color=line_color, key_cls=key_cls)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            # shape.locked = False
            shape.close()
            s.append(shape)

            self._update_shape_color(shape)
            self.addLabel(shape)

        self.updateComboBox()
        self.canvas.loadShapes(s)

    def singleLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        item.setText(shape.label)
        self.updateComboBox()

        # ADD:
        item = self.shapesToItemsbox[shape]
        item.setText(str([(int(p.x()), int(p.y())) for p in shape.points]))
        self.updateComboBox()

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]

        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append("")
        uniqueTextList.sort()

        # self.comboBox.update_items(uniqueTextList)

    def saveLabels(self, annotationFilePath, mode='Auto'):
        # Mode is Auto means that labels will be loaded from self.result_dic totally, which is the output of ocr model
        annotationFilePath = ustr(annotationFilePath)

        def format_shape(s):
            # print('s in saveLabels is ',s)
            return dict(label=s.label,  # str
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(int(p.x()), int(p.y())) for p in s.points],  # QPonitF
                        difficult=s.difficult,
                        key_cls=s.key_cls)  # bool

        if mode == 'Auto':
            shapes = []
        else:
            shapes = [format_shape(shape) for shape in self.canvas.shapes if shape.line_color != DEFAULT_LOCK_COLOR]
        # Can add differrent annotation formats here
        for box in self.result_dic:
            trans_dic = {"label": box[1][0], "points": box[0], "difficult": False}
            if self.kie_mode:
                trans_dic.update({"key_cls": "None"})
            if trans_dic["label"] == "" and mode == 'Auto':
                continue
            shapes.append(trans_dic)

        try:
            trans_dic = []
            for box in shapes:
                trans_dict = {"transcription": box['label'], "points": box['points'], "difficult": box['difficult']}
                if self.kie_mode:
                    trans_dict.update({"key_cls": box['key_cls']})
                trans_dic.append(trans_dict)
            self.PPlabel[annotationFilePath] = trans_dic
            if mode == 'Auto':
                self.Cachelabel[annotationFilePath] = trans_dic

            # else:
            #     self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
            #                         self.lineColor.getRgb(), self.fillColor.getRgb())
            # print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            return True
        except:
            self.errorMessage(u'Error saving label data', u'Error saving label data')
            return False

    def copySelectedShape(self):
        for shape in self.canvas.copySelectedShape():
            self.addLabel(shape)
        # fix copy and delete
        # self.shapeSelectionChanged(True)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(self.itemsToShapes[item])
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def boxSelectionChanged(self):
        if self._noSelectionSlot:
            # self.BoxList.scrollToItem(self.currentBox(), QAbstractItemView.PositionAtCenter)
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.BoxList.selectedItems():
                selected_shapes.append(self.itemsToShapesbox[item])
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            # shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        elif not ((item.checkState() == Qt.Unchecked) ^ (not shape.difficult)):
            shape.difficult = True if item.checkState() == Qt.Unchecked else False
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, True)  # item.checkState() == Qt.Checked
            # self.actions.save.setEnabled(True)

    # Callback functions:
    def newShape(self, value=True):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if len(self.labelHist) > 0:
            self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        if value:
            text = self.labelDialog.popUp(text=self.prevLabelText)
            self.lastLabel = text
        else:
            text = self.prevLabelText

        if text is not None:
            self.prevLabelText = self.stringBundle.getString('tempLabel')

            shape = self.canvas.setLastLabel(text, None, None, None)  # generate_color, generate_color
            if self.kie_mode:
                key_text, _ = self.keyDialog.popUp(self.key_previous_text)
                if key_text is not None:
                    shape = self.canvas.setLastLabel(text, None, None, key_text)  # generate_color, generate_color
                    self.key_previous_text = key_text
                    if not self.keyList.findItemsByLabel(key_text):
                        item = self.keyList.createItemFromLabel(key_text)
                        self.keyList.addItem(item)
                        rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                        self.keyList.setItemLabel(item, key_text, rgb)

                    self._update_shape_color(shape)
                    self.keyDialog.addLabelHistory(key_text)

            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
                self.actions.createpoly.setEnabled(True)
                self.actions.undoLastPoint.setEnabled(False)
                self.actions.undo.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.key_cls, self.kie_mode)
        shape.line_color = QColor(r, g, b)
        shape.vertex_fill_color = QColor(r, g, b)
        shape.hvertex_fill_color = QColor(255, 255, 255)
        shape.fill_color = QColor(r, g, b, 128)
        shape.select_line_color = QColor(255, 255, 255)
        shape.select_fill_color = QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label, kie_mode):
        shift_auto_shape_color = 2  # use for random color
        if kie_mode and label != "None":
            item = self.keyList.findItemsByLabel(label)[0]
            label_id = self.keyList.indexFromItem(item).row() + 1
            label_id += shift_auto_shape_color
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        else:
            return (0, 255, 0)

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)
        self.imageSlider.setValue(self.zoomWidget.value() + increment)  # set zoom slider value

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            self.canvas.setShapeVisible(shape, value)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        if self.dirty:
            self.mayContinue()
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)
        # Fix bug: An index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        # unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item

        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                print('unicodeFilePath is', unicodeFilePath)
                fileWidgetItem.setSelected(True)
                self.iconlist.clear()
                self.additems5(None)

                for i in range(5):
                    item_tooltip = self.iconlist.item(i).toolTip()
                    # print(i,"---",item_tooltip)
                    if item_tooltip == ustr(filePath):
                        titem = self.iconlist.item(i)
                        titem.setSelected(True)
                        self.iconlist.scrollToItem(titem)
                        break
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()
                self.iconlist.clear()

        # if unicodeFilePath and self.iconList.count() > 0:
        #     if unicodeFilePath in self.mImgList:

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            self.canvas.verified = False
            cvimg = cv2.imdecode(np.fromfile(unicodeFilePath, dtype=np.uint8), 1)
            height, width, depth = cvimg.shape
            cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
            image = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)

            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))

            if self.validFilestate(filePath) is True:
                self.setClean()
            else:
                self.dirty = False
                self.actions.save.setEnabled(True)
            if len(self.canvas.lockedShapes) != 0:
                self.actions.save.setEnabled(True)
                self.setDirty()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            self.showBoundingBoxFromPPlabel(filePath)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count() - 1))
                self.labelList.item(self.labelList.count() - 1).setSelected(True)

            # show file list image count
            select_indexes = self.fileListWidget.selectedIndexes()
            if len(select_indexes) > 0:
                self.fileDock.setWindowTitle(self.fileListName + f" ({select_indexes[0].row() + 1}"
                                                                 f"/{self.fileListWidget.count()})")
            # update show counting
            self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
            self.labelListDock.setWindowTitle(self.labelListDockName + f" ({self.labelList.count()})")

            self.canvas.setFocus(True)
            return True
        return False

    def showBoundingBoxFromPPlabel(self, filePath):
        width, height = self.image.width(), self.image.height()
        imgidx = self.getImglabelidx(filePath)
        shapes = []
        # box['ratio'] of the shapes saved in lockedShapes contains the ratio of the
        # four corner coordinates of the shapes to the height and width of the image
        for box in self.canvas.lockedShapes:
            key_cls = 'None' if not self.kie_mode else box['key_cls']
            if self.canvas.isInTheSameImage:
                shapes.append((box['transcription'], [[s[0] * width, s[1] * height] for s in box['ratio']],
                               DEFAULT_LOCK_COLOR, key_cls, box['difficult']))
            else:
                shapes.append(('锁定框：待检测', [[s[0] * width, s[1] * height] for s in box['ratio']],
                               DEFAULT_LOCK_COLOR, key_cls, box['difficult']))
        if imgidx in self.PPlabel.keys():
            for box in self.PPlabel[imgidx]:
                key_cls = 'None' if not self.kie_mode else box.get('key_cls', 'None')
                shapes.append((box['transcription'], box['points'], None, key_cls, box.get('difficult', False)))

        self.loadLabels(shapes)
        self.canvas.verified = False

    def validFilestate(self, filePath):
        if filePath not in self.fileStatedict.keys():
            return None
        elif self.fileStatedict[filePath] == 1:
            return True
        else:
            return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))
        self.imageSlider.setValue(self.zoomWidget.value())  # set zoom slider value

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e - 110
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        else:
            settings = self.settings
            # If it loads images from dir, don't load it at the beginning
            if self.dirname is None:
                settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
            else:
                settings[SETTING_FILENAME] = ''

            settings[SETTING_WIN_SIZE] = self.size()
            settings[SETTING_WIN_POSE] = self.pos()
            settings[SETTING_WIN_STATE] = self.saveState()
            settings[SETTING_LINE_COLOR] = self.lineColor
            settings[SETTING_FILL_COLOR] = self.fillColor
            settings[SETTING_RECENT_FILES] = self.recentFiles
            settings[SETTING_ADVANCE_MODE] = not self._beginner
            if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
                settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
            else:
                settings[SETTING_SAVE_DIR] = ''

            if self.lastOpenDir and os.path.exists(self.lastOpenDir):
                settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
            else:
                settings[SETTING_LAST_OPEN_DIR] = ''

            settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
            settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
            settings.save()
            try:
                self.saveLabelFile()
            except:
                pass

    def loadRecent(self, filename):
        if self.mayContinue():
            print(filename, "======")
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for file in os.listdir(folderPath):
            if file.lower().endswith(tuple(extensions)):
                relativePath = os.path.join(folderPath, file)
                path = ustr(os.path.abspath(relativePath))
                images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent != True:
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                  '%s - Open Directory' % __appname__,
                                                                  defaultOpenDirPath,
                                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            targetDirPath = ustr(defaultOpenDirPath)
        self.lastOpenDir = targetDirPath
        self.importDirImages(targetDirPath)

    def openDatasetDirDialog(self):
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            if platform.system() == 'Windows':
                os.startfile(self.lastOpenDir)
            else:
                os.system('open ' + os.path.normpath(self.lastOpenDir))
            defaultOpenDirPath = self.lastOpenDir 
        else:
            self.msgBox.warning(self, "Warn",
                                "\n The original folder no longer exists, please choose the data set path again!")

            self.actions.open_dataset_dir.setEnabled(False)
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'

    def init_key_list(self, label_dict):
        if not self.kie_mode:
            return
        # load key_cls
        for image, info in label_dict.items():
            for box in info:
                if "key_cls" not in box:
                    box.update({"key_cls": "None"})
                self.existed_key_cls_set.add(box["key_cls"])
        if len(self.existed_key_cls_set) > 0:
            for key_text in self.existed_key_cls_set:
                if not self.keyList.findItemsByLabel(key_text):
                    item = self.keyList.createItemFromLabel(key_text)
                    self.keyList.addItem(item)
                    rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                    self.keyList.setItemLabel(item, key_text, rgb)

        if self.keyDialog is None:
            # key list dialog
            self.keyDialog = KeyDialog(
                text=self.key_dialog_tip,
                parent=self,
                labels=self.existed_key_cls_set,
                sort_labels=True,
                show_text_field=True,
                completion="startswith",
                fit_to_content={'column': True, 'row': False},
                flags=None
            )

    def importDirImages(self, dirpath, isDelete=False):
        if not self.mayContinue() or not dirpath:
            return
        if self.defaultSaveDir and self.defaultSaveDir != dirpath:
            self.saveLabelFile()

        if not isDelete:
            self.loadFilestate(dirpath)
            self.PPlabelpath = dirpath + '/Label.txt'
            self.PPlabel = self.loadLabelFile(self.PPlabelpath)
            self.Cachelabelpath = dirpath + '/Cache.cach'
            self.Cachelabel = self.loadLabelFile(self.Cachelabelpath)
            if self.Cachelabel:
                self.PPlabel = dict(self.Cachelabel, **self.PPlabel)

            self.init_key_list(self.PPlabel)

        self.lastOpenDir = dirpath
        self.dirname = dirpath

        self.defaultSaveDir = dirpath
        self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                     (__appname__, self.defaultSaveDir))
        self.statusBar().show()

        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.mImgList5 = self.mImgList[:5]
        self.openNextImg()
        doneicon = newIcon('done')
        closeicon = newIcon('close')
        for imgPath in self.mImgList:
            filename = os.path.basename(imgPath)
            if self.validFilestate(imgPath) is True:
                item = QListWidgetItem(doneicon, filename)
            else:
                item = QListWidgetItem(closeicon, filename)
            self.fileListWidget.addItem(item)

        print('DirPath in importDirImages is', dirpath)
        self.iconlist.clear()
        self.additems5(dirpath)
        self.changeFileFolder = True
        self.haveAutoReced = False
        self.AutoRecognition.setEnabled(True)
        self.AutoCurRecognition.setEnabled(True) 
        self.reRecogButton.setEnabled(True)
        self.actions.AutoRec.setEnabled(True)
        self.actions.reRec.setEnabled(True)
        self.actions.open_dataset_dir.setEnabled(True)
        self.actions.rotateLeft.setEnabled(True)
        self.actions.rotateRight.setEnabled(True)

        self.fileListWidget.setCurrentRow(0)  # set list index to first
        self.fileDock.setWindowTitle(self.fileListName + f" (1/{self.fileListWidget.count()})")  # show image count

    def openPrevImg(self, _value=False):
        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        self.mImgList5 = self.mImgList[:5]
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            self.mImgList5 = self.indexTo5Files(currIndex - 1)
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
            self.mImgList5 = self.mImgList[:5]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]
                self.mImgList5 = self.indexTo5Files(currIndex + 1)
            else:
                self.mImgList5 = self.indexTo5Files(currIndex)
        if filename:
            print('file name in openNext is ', filename)
            self.loadFile(filename)

    def updateFileListIcon(self, filename):
        pass

    def saveFile(self, _value=False, mode='Manual'):
        # Manual mode is used for users click "Save" manually,which will change the state of the image
        if self.filePath:
            imgidx = self.getImglabelidx(self.filePath)
            self._saveFile(imgidx, mode=mode)

    def saveLockedShapes(self):
        self.canvas.lockedShapes = []
        self.canvas.selectedShapes = []
        for s in self.canvas.shapes:
            if s.line_color == DEFAULT_LOCK_COLOR:
                self.canvas.selectedShapes.append(s)
        self.lockSelectedShape()
        for s in self.canvas.shapes:
            if s.line_color == DEFAULT_LOCK_COLOR:
                self.canvas.selectedShapes.remove(s)
                self.canvas.shapes.remove(s)

    def _saveFile(self, annotationFilePath, mode='Manual'):
        if len(self.canvas.lockedShapes) != 0:
            self.saveLockedShapes()

        if mode == 'Manual':
            self.result_dic_locked = []
            img = cv2.imread(self.filePath)
            width, height = self.image.width(), self.image.height()
            for shape in self.canvas.lockedShapes:
                box = [[int(p[0] * width), int(p[1] * height)] for p in shape['ratio']]
                # assert len(box) == 4
                result = [(shape['transcription'], 1)]
                result.insert(0, box)
                self.result_dic_locked.append(result)
            self.result_dic += self.result_dic_locked
            self.result_dic_locked = []
            if annotationFilePath and self.saveLabels(annotationFilePath, mode=mode):
                self.setClean()
                self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
                self.statusBar().show()
                currIndex = self.mImgList.index(self.filePath)
                item = self.fileListWidget.item(currIndex)
                item.setIcon(newIcon('done'))

                self.fileStatedict[self.filePath] = 1
                if len(self.fileStatedict) % self.autoSaveNum == 0:
                    self.saveFilestate()
                    self.savePPlabel(mode='Auto')

                self.fileListWidget.insertItem(int(currIndex), item)
                if not self.canvas.isInTheSameImage:
                    self.openNextImg()
                
                self.actions.saveLabel.setEnabled(True) 
                self.actions.exportDet.setEnabled(True)
                self.actions.exportRec.setEnabled(True) 
                self.actions.exportKie.setEnabled(True)  

                self.actions.exportDetMM.setEnabled(True)
                self.actions.exportRecMM.setEnabled(True) 
                self.actions.exportKieMM.setEnabled(True) 
                
                self.actions.makeRecData.setEnabled(True) 
                

        elif mode == 'Auto':
            if annotationFilePath and self.saveLabels(annotationFilePath, mode=mode):
                self.setClean()
                self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
                self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)


    # Export selected single image 
    def exportSingleImg(self): 
        img = cv2.imread(self.filePath)

        fbase = os.path.basename(self.filePath).split('.')[0]
        rec_gt_dir = os.path.dirname(self.PPlabelpath) + '/rec_sgt_' + fbase + '.txt'
        crop_img_dir = os.path.dirname(self.PPlabelpath) + '/crop_img/' 
        # os.makedirs(crop_img_dir, exist_ok=False)
            

        img_name = "" 
        for shape in self.canvas.selectedShapes:
            text = shape.label             
            box = [[int(p.x()), int(p.y())] for p in shape.points]
            if len(box) > 4:
                box = self.gen_quad_from_poly(np.array(box))
            assert len(box) == 4
            img_crop = get_rotate_crop_image(img, np.array(box, np.float32)) 

            if img_crop is None:
                msg = 'Can not recognise the detection box in ' + self.filePath + '. Please change manually'
                QMessageBox.information(self, "Information", msg)
                return

            try: 
                fexist = False 
                labels = [] 
                if os.path.isfile(rec_gt_dir): 
                    with open(rec_gt_dir, 'r', encoding='utf-8') as gf: 
                        data = gf.readlines()  

                    idx = len(data)
                    img_name = fbase + '_' + str(idx) + '.jpg'
                    cv2.imwrite(crop_img_dir + img_name, img_crop)  
                    labels.append(img_name + '\t' + text)  
                    
                else: 
                    img_name = fbase + '_0.jpg'                       
                    cv2.imwrite(crop_img_dir + img_name, img_crop)  
                    labels.append(img_name + '\t' + text) 
                
                with open(rec_gt_dir, 'a', encoding='utf-8') as f:
                    for lbl in labels: 
                        f.write(lbl + '\n') 
            except Exception as e: 
                print("Can not read image ", e)
                return 

        QMessageBox.information(self, "Information", "선택한 Crop 이미지가 저장되었습니다: \n" + str(crop_img_dir + img_name))   
  

    def exportCropImg(self):
        imgPath = self.filePath
        if imgPath is not None: 
            self.export_single_file_rec(imgPath)   

    def deleteImgDialog(self):
        yes, cancel = QMessageBox.Yes, QMessageBox.Cancel
        msg = u'현재 화면의 이미지를 휴지통에 버리시겠습니까?'
        return QMessageBox.warning(self, u'Attention', msg, yes | cancel) 


    def export_single_file_rec(self, fname):  
        if {} in [self.PPlabelpath, self.PPlabel, self.fileStatedict]:
            QMessageBox.information(self, "Information", "이미지를 확인하세요 (체크된 이미지에 한해서 Export가능합니다)")
            return

        fbase = os.path.basename(fname).split('.')[0]
        rec_gt_dir = os.path.dirname(self.PPlabelpath) + '/rec_gt_' + fbase + '.txt'
        crop_img_dir = os.path.dirname(self.PPlabelpath) + '/crop_img/'
        ques_img = []
        if not os.path.exists(crop_img_dir):
            os.mkdir(crop_img_dir)

        if not os.path.isfile(fname): 
            ques_img.apend(fname)
        else: 
            with open(rec_gt_dir, 'w', encoding='utf-8') as f:             
                idx = self.getImglabelidx(fname)
                try:
                    img = cv2.imread(fname)
                    for i, label in enumerate(self.PPlabel[idx]):
                        if label['difficult']:
                            continue
                        img_crop = get_rotate_crop_image(img, np.array(label['points'], np.float32))
                        img_name = os.path.splitext(os.path.basename(idx))[0] + '_' + str(i) + '.jpg'
                        cv2.imwrite(crop_img_dir + img_name, img_crop)
                        f.write(img_name + '\t')
                        f.write(label['transcription'] + '\n')
                except Exception as e:
                    ques_img.append(fname)
                    print("Can not read image ", e)
        if ques_img:
            QMessageBox.information(self, "Information",
                                    "이미지 경로와 라벨을 확인해주세요.\n"
                                    + "".join(str(i) + '\n' for i in ques_img))
        QMessageBox.information(self, "Information", "Crop 이미지들이 저장되었습니다(디렉토리): " + str(crop_img_dir))  


    


    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):  #
        if not self.dirty:
            return True
        else:
            discardChanges = self.discardChangesDialog()
            if discardChanges == QMessageBox.No:
                return True
            elif discardChanges == QMessageBox.Yes:
                self.canvas.isInTheSameImage = True
                self.saveFile()
                self.canvas.isInTheSameImage = False
                return True
            else:
                return False

    def discardChangesDialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        msg = u'아직 저장하지 않은 데이터가 있습니다. 저장후 계속 진행하시겠습니까?\n 변경사항을 저장하지 않으려면 "No" 버튼을 클릭하세요.'
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabels(self.canvas.deleteSelected())
        self.actions.undo.setEnabled(True)
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)
        self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
        self.labelListDock.setWindowTitle(self.labelListDockName + f" ({self.labelList.count()})")

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            for shape in self.canvas.selectedShapes: shape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            for shape in self.canvas.selectedShapes: shape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())

    def additems(self, dirpath):
        for file in self.mImgList:
            pix = QPixmap(file)
            _, filename = os.path.split(file)
            filename, _ = os.path.splitext(filename)
            item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.FastTransformation)),
                                   filename[:10])
            item.setToolTip(file)
            self.iconlist.addItem(item)

    def additems5(self, dirpath):
        for file in self.mImgList5:
            pix = QPixmap(file)
            _, filename = os.path.split(file)
            filename, _ = os.path.splitext(filename)
            pfilename = filename[:10]
            if len(pfilename) < 10:
                lentoken = 12 - len(pfilename)
                prelen = lentoken // 2
                bfilename = prelen * " " + pfilename + (lentoken - prelen) * " "
            # item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)),filename[:10])
            item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.FastTransformation)), pfilename)
            # item.setForeground(QBrush(Qt.white))
            item.setToolTip(file)
            self.iconlist.addItem(item)
        owidth = 0
        for index in range(len(self.mImgList5)):
            item = self.iconlist.item(index)
            itemwidget = self.iconlist.visualItemRect(item)
            owidth += itemwidget.width()
        self.iconlist.setMinimumWidth(owidth + 50)

    def gen_quad_from_poly(self, poly):
        """
        Generate min area quad from poly.
        """
        point_num = poly.shape[0]
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        rect = cv2.minAreaRect(poly.astype(
            np.int32))  # (center (x,y), (width, height), angle of rotation)
        box = np.array(cv2.boxPoints(rect))

        first_point_idx = 0
        min_dist = 1e4
        for i in range(4):
            dist = np.linalg.norm(box[(i + 0) % 4] - poly[0]) + \
                   np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                   np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2]) + \
                   np.linalg.norm(box[(i + 3) % 4] - poly[-1])
            if dist < min_dist:
                min_dist = dist
                first_point_idx = i
        for i in range(4):
            min_area_quad[i] = box[(first_point_idx + i) % 4]

        bbox_new = min_area_quad.tolist()
        bbox = []

        for box in bbox_new:
            box = list(map(int, box))
            bbox.append(box)

        return bbox

    def getImglabelidx(self, filePath):
        if platform.system() == 'Windows':
            spliter = '\\' 
        else:
            spliter = '/'
            # filePath = filePath.replace('\\', '/')
        filepathsplit = filePath.split(spliter)[-2:]
        return filepathsplit[0] + '/' + filepathsplit[1]

    def autoRecognition(self):
        assert self.mImgList is not None
        # print('Using model from ', self.model)

        uncheckedList = [i for i in self.mImgList if i not in self.fileStatedict.keys()]
        self.autoDialog = AutoDialog(parent=self, ocr=self.ocr, mImgList=uncheckedList, lenbar=len(uncheckedList))
        self.autoDialog.popUp()
        self.currIndex = len(self.mImgList) - 1
        self.loadFile(self.filePath)  # ADD
        self.haveAutoReced = True
        self.AutoRecognition.setEnabled(False)
        # self.actions.AutoRec.setEnabled(False)
        self.setDirty()
        self.saveCacheLabel()

        self.init_key_list(self.Cachelabel)


    def autoCurRecognition(self):
        assert self.mImgList is not None
        # print('Using model from ', self.model)

        currList = [self.filePath]
        self.autoDialog = AutoDialog(parent=self, ocr=self.ocr, mImgList=currList, lenbar=len(currList))
        self.autoDialog.popUp()
        self.currIndex = len(self.mImgList) - 1
        self.loadFile(self.filePath)  # ADD
        self.haveAutoReced = True
        # self.AutoCurRecognition.setEnabled(False)
        # self.actions.AutoCurRec.setEnabled(False)
        self.setDirty()
        with open(self.Cachelabelpath, 'w', encoding='utf-8') as f:
            for key in self.Cachelabel:
                f.write(key + '\t')
                f.write(json.dumps(self.Cachelabel[key], ensure_ascii=False) + '\n')

        self.init_key_list(self.Cachelabel) 

    def reRecognition(self):
        img = cv2.imread(self.filePath)
        # org_box = [dic['points'] for dic in self.PPlabel[self.getImglabelidx(self.filePath)]]
        if self.canvas.shapes:
            self.result_dic = []
            self.result_dic_locked = []  # result_dic_locked stores the ocr result of self.canvas.lockedShapes
            rec_flag = 0
            for shape in self.canvas.shapes:
                box = [[int(p.x()), int(p.y())] for p in shape.points]

                if len(box) > 4:
                    box = self.gen_quad_from_poly(np.array(box))
                assert len(box) == 4

                img_crop = get_rotate_crop_image(img, np.array(box, np.float32))
                if img_crop is None:
                    msg = 'Can not recognise the detection box in ' + self.filePath + '. Please change manually'
                    QMessageBox.information(self, "Information", msg)
                    return
                result = self.ocr.ocr(img_crop, cls=True, det=False)
                if result[0][0] != '':
                    if shape.line_color == DEFAULT_LOCK_COLOR:
                        shape.label = result[0][0]
                        result.insert(0, box)
                        self.result_dic_locked.append(result)
                    else:
                        result.insert(0, box)
                        self.result_dic.append(result)
                else:
                    print('Can not recognise the box')
                    if shape.line_color == DEFAULT_LOCK_COLOR:
                        shape.label = result[0][0]
                        self.result_dic_locked.append([box, (self.noLabelText, 0)])
                    else:
                        self.result_dic.append([box, (self.noLabelText, 0)])
                try:
                    if self.noLabelText == shape.label or result[1][0] == shape.label:
                        print('label no change')
                    else:
                        rec_flag += 1
                except IndexError as e:
                    print('Can not recognise the box')
            if (len(self.result_dic) > 0 and rec_flag > 0) or self.canvas.lockedShapes:
                self.canvas.isInTheSameImage = True
                self.saveFile(mode='Auto')
                self.loadFile(self.filePath)
                self.canvas.isInTheSameImage = False
                self.setDirty()
            elif len(self.result_dic) == len(self.canvas.shapes) and rec_flag == 0:
                if self.lang == 'ch':
                    QMessageBox.information(self, "Information", "识别结果保持一致！")
                else:
                    QMessageBox.information(self, "Information", "인식결과 변경 없슴!")
            else:
                print('Can not recgonise in ', self.filePath)
        else:
            QMessageBox.information(self, "Information", "Draw a box!")

    def singleRerecognition(self):
        img = cv2.imread(self.filePath)
        for shape in self.canvas.selectedShapes:
            box = [[int(p.x()), int(p.y())] for p in shape.points]
            if len(box) > 4:
                box = self.gen_quad_from_poly(np.array(box))
            assert len(box) == 4
            img_crop = get_rotate_crop_image(img, np.array(box, np.float32))
            if img_crop is None:
                msg = 'Can not recognise the detection box in ' + self.filePath + '. Please change manually'
                QMessageBox.information(self, "Information", msg)
                return
            result = self.ocr.ocr(img_crop, cls=True, det=False)
            if result[0][0] != '':
                result.insert(0, box)
                print('result in reRec is ', result)
                if result[1][0] == shape.label:
                    print('label no change')
                else:
                    shape.label = result[1][0]
            else:
                print('Can not recognise the box')
                if self.noLabelText == shape.label:
                    print('label no change')
                else:
                    shape.label = self.noLabelText
            self.singleLabel(shape)
            self.setDirty()

    #-------------------------------------------------- 
    # Export into PaddleOCR format 
    def exportKie(self):
        '''
            export Label and CSV to JSON
        '''  
        lbl_file = os.path.join(self.lastOpenDir, 'Label.txt') 
        if os.name == 'nt': 
            img_dir = ("\\").join(self.lastOpenDir.split("\\")[:-1])  
        else: 
            img_dir = ('/').join(self.lastOpenDir.split('/')[:-1])   

        output1 = os.path.join(self.lastOpenDir, 'kie_label.json')   
        output2 = os.path.join(self.lastOpenDir, 'kie_label_last.json')  

        if not os.path.isfile(lbl_file): 
            QMessageBox.warning(self, "Information", "라벨파일(Label.txt)이 없습니다. 먼저 [라벨 내보내기]를 실행하세요.") 
            return 

        # 먼저 라벨파일 저장처리 
        savedfile = [self.getImglabelidx(i) for i in self.fileStatedict.keys()]
        with open(self.PPlabelpath, 'w', encoding='utf-8') as f:
            for key in self.PPlabel:
                if key in savedfile and self.PPlabel[key] != []:
                    f.write(key + '\t')
                    f.write(json.dumps(self.PPlabel[key], ensure_ascii=False) + '\n')

        # SER Dataset 생성/저장 처리 
        self.make_kie_dataset(lbl_file, img_dir, output1, output2)
        
        msg = 'KIE JSON 파일 저장 완료: \n- {} \n- {}'.format(output1, output2)
        QMessageBox.information(self, "Information", msg)

    
    def exportDet(self):
        '''
            export Detection Labels to JSON
        '''  
        lbl_file = os.path.join(self.lastOpenDir, 'Label.txt') 
        if os.name == 'nt': 
            img_dir = ("\\").join(self.lastOpenDir.split("\\")[:-1])  
        else: 
            img_dir = ('/').join(self.lastOpenDir.split('/')[:-1])   

        output1 = os.path.join(self.lastOpenDir, 'det_label.json')    

        if not os.path.isfile(lbl_file): 
            QMessageBox.warning(self, "Information", "라벨파일(Label.txt)이 없습니다. 먼저 [라벨 내보내기]를 실행하세요.") 
            return 

        # 먼저 라벨파일 저장처리 
        savedfile = [self.getImglabelidx(i) for i in self.fileStatedict.keys()]
        with open(self.PPlabelpath, 'w', encoding='utf-8') as f:
            for key in self.PPlabel:
                if key in savedfile and self.PPlabel[key] != []:
                    f.write(key + '\t')
                    f.write(json.dumps(self.PPlabel[key], ensure_ascii=False) + '\n')
 
        self.make_det_dataset(lbl_file, img_dir, output1)
        
        msg = 'Det Label JSON 파일 저장 완료: \n- {}'.format(output1)
        QMessageBox.information(self, "Information", msg)

    def exportRec(self):
        if {} in [self.PPlabelpath, self.PPlabel, self.fileStatedict]:
            QMessageBox.information(self, "Information", "이미지를 확인하세요")
            return

        rec_gt_dir = os.path.dirname(self.PPlabelpath) + '/rec_label.txt'
        crop_img_dir = os.path.dirname(self.PPlabelpath) + '/crop_img/'
        ques_img = []
        if not os.path.exists(crop_img_dir):
            os.mkdir(crop_img_dir)

        with open(rec_gt_dir, 'w', encoding='utf-8') as f:
            for key in self.fileStatedict:
                if not os.path.isfile(key): 
                    continue  
                idx = self.getImglabelidx(key)
                try:
                    img = cv2.imread(key)
                    for i, label in enumerate(self.PPlabel[idx]):
                        if label['difficult']:
                            continue
                        img_crop = get_rotate_crop_image(img, np.array(label['points'], np.float32))
                        img_name = os.path.splitext(os.path.basename(idx))[0] + '_crop_' + str(i) + '.jpg'
                        cv2.imwrite(crop_img_dir + img_name, img_crop)
                        f.write('crop_img/' + img_name + '\t')
                        f.write(label['transcription'] + '\n')
                except Exception as e:
                    ques_img.append(key)
                    print("Can not read image ", e)
        if ques_img:
            QMessageBox.information(self, "Information",
                                    "아래의 이미지들은 저장될수 없습니다. 이미지 경로와 라벨을 확인해주세요.\n"
                                    + "".join(str(i) + '\n' for i in ques_img))
        QMessageBox.information(self, "Information", "Crop 이미지들이 저장되었습니다(디렉토리): " + str(crop_img_dir))


    # -------------------------------------------------------------- 
    # Export into MMOCR Format 

    def exportKieMM(self):
        '''
            export Label and CSV to JSON
        '''  
        lbl_file = os.path.join(self.lastOpenDir, 'Label.txt') 
        if os.name == 'nt': 
            img_dir = ("\\").join(self.lastOpenDir.split("\\")[:-1])  
        else: 
            img_dir = ('/').join(self.lastOpenDir.split('/')[:-1])   

        output1 = os.path.join(self.lastOpenDir, 'kie_label.json')   
        output2 = os.path.join(self.lastOpenDir, 'kie_label_last.json')  

        if not os.path.isfile(lbl_file): 
            QMessageBox.warning(self, "Information", "라벨파일(Label.txt)이 없습니다. 먼저 [라벨 내보내기]를 실행하세요.") 
            return 

        # 먼저 라벨파일 저장처리 
        savedfile = [self.getImglabelidx(i) for i in self.fileStatedict.keys()]
        with open(self.PPlabelpath, 'w', encoding='utf-8') as f:
            for key in self.PPlabel:
                if key in savedfile and self.PPlabel[key] != []:
                    f.write(key + '\t')
                    f.write(json.dumps(self.PPlabel[key], ensure_ascii=False) + '\n')

        # SER Dataset 생성/저장 처리 (stopword 필터 적용)
        self.make_kie_dataset(lbl_file, img_dir, output1, output2)
        
        msg = 'KIE JSON 파일 저장 완료: \n- {} \n- {}'.format(output1, output2)
        QMessageBox.information(self, "Information", msg)

    
    def exportDetMM(self):
        '''
            export Detection Labels to JSON
        '''   
        
        lbl_file = os.path.join(self.lastOpenDir, 'Label.txt') 
        if os.name == 'nt': 
            img_dir = ("\\").join(self.lastOpenDir.split("\\")[:-1])  
        else: 
            img_dir = ('/').join(self.lastOpenDir.split('/')[:-1])   

        output = os.path.join(self.lastOpenDir, 'det_label.json')  
        if not os.path.isfile(lbl_file): 
            QMessageBox.warning(self, "Information", "라벨파일(Label.txt)이 없습니다. 먼저 [라벨 내보내기]를 실행하세요.") 
            return 

        # 먼저 라벨파일 저장처리 
        savedfile = [self.getImglabelidx(i) for i in self.fileStatedict.keys()]
        with open(self.PPlabelpath, 'w', encoding='utf-8') as f:
            for key in self.PPlabel:
                if key in savedfile and self.PPlabel[key] != []:
                    f.write(key + '\t')
                    f.write(json.dumps(self.PPlabel[key], ensure_ascii=False) + '\n')
 
        output, result = self.make_det_dataset(lbl_file, img_dir, output, ocr_flag='M')
        
        if result:  
            QMessageBox.information(self, "Information", 
                                    'Detection Label 파일 저장 완료: \n- {}'.format(output))
        else: 
            QMessageBox.information(self, "Error", 
                                'Detection Label 파일 저장 실패: \n- {}'.format(output))

    def exportRecMM(self):
        if {} in [self.PPlabelpath, self.PPlabel, self.fileStatedict]:
            QMessageBox.information(self, "Information", "이미지를 확인하세요")
            return
 
        try:  
            rec_gt_dir = os.path.dirname(self.PPlabelpath) + '/rec_label.json'
            crop_img_dir = os.path.dirname(self.PPlabelpath) + '/crop_img/'
            ques_img = []
            if not os.path.exists(crop_img_dir):
                os.mkdir(crop_img_dir) 
            
            mm_parse = {
                "metainfo":
                    {
                    "dataset_type": "TextRecogDataset",
                    "task_name": "textrecog",
                    },
                "data_list": None 
            }
            
            inst_list = []  
            for key in self.fileStatedict:
                if not os.path.isfile(key): 
                    continue 
                idx = self.getImglabelidx(key)
                try: 
                    img = cv2.imread(key)
                    for i, label in enumerate(self.PPlabel[idx]):
                        if label['difficult']:
                            continue
                        img_crop = get_rotate_crop_image(img, np.array(label['points'], np.float32))
                        img_name = os.path.splitext(os.path.basename(idx))[0] + '_' + str(i) + '.jpg'
                        cv2.imwrite(os.path.join(crop_img_dir, img_name), img_crop) 
                        
                        inst_list.append({
                            "img_path": 'crop_img/' + img_name ,
                            "instances": [
                                {
                                "text": label['transcription'] 
                                }
                            ]
                        })  
                        
                        # f.write('crop_img/' + img_name + '\t')
                        # f.write(label['transcription'] + '\n')
                except Exception as ex:
                    ques_img.append(key)
                    print("Can not read image ", ex)
                
            mm_parse['data_list'] = inst_list 
            json_obj = json.dumps(mm_parse, ensure_ascii=False).replace("\\", "")    
            with open(rec_gt_dir, 'w') as f:  
                print(json_obj, file=f) 
                    
            if ques_img:
                QMessageBox.information(self, "Information",
                                        "아래의 이미지들은 저장될수 없습니다. 이미지 경로와 라벨을 확인해주세요.\n"
                                        + "".join(str(i) + '\n' for i in ques_img))
            QMessageBox.information(self, "Information", "Rec. 이미지들이 저장되었습니다(디렉토리): \n" + str(crop_img_dir))

        except Exception as ex: 
            print(repr(ex)) 

    def make_kie_dataset(self, label_file, img_dir, output1, output2): 
        # KIE 
        try: 
            if os.path.isfile(output1): os.remove(output1) 
            if os.path.isfile(output2): os.remove(output2)  
        
            with open(label_file) as f: 
                data = f.readlines() 
            
            results = []   
            for d in data:  
                bbox_info = []  
                
                # label = {'height':0, 'width': 0, 'ocr_info': []}  
                img_file = d.split('\t')[0] 
                label_list = d.split('\t')[1].strip()  

                fname, ext = img_file.split('.')[0], img_file.split('.')[1]  
                print(img_file) 

                if not os.path.isfile(os.path.join(img_dir, img_file)): 
                    print('Not found error {}: '.format(img_file))
                    continue 

                img = cv2.imread(os.path.join(img_dir, img_file))
 
                dics = ['{' + x for x in label_list.replace('[{', '').replace('}]', '}').split(', {')] 
                for seq, dic in enumerate(dics):  
                    item = json.loads(dic) 
                    exclude = item['difficult']  
                    text = item['transcription'] 
                    pt = item['points'] 
                    key = item['key_cls'].replace('None', 'other')
                     
                    if exclude: 
                        continue 
         
                    bbox_info.append({
                        'transcription': text, 
                        'label': key, 
                        'points': pt, 
                        'id': seq, 
                        'linking': []   
                    }) 
                    
                fname = fname.split('\\')[1] if os.name=='nt' else fname.split('/')[1] 
                # label['ocr_info'] = bbox_info 
                label_str = json.dumps(bbox_info, ensure_ascii=False) 
                results.append(fname + '.' + ext + '\t' + label_str)
        
            with open(output1, 'w') as f:  
                for d in results: 
                    f.write(d + '\n') 
            return output1, output2 
        except Exception as ex: 
            print(repr(ex)) 

    def make_det_dataset(self, label_file, img_dir, output, ocr_flag='P'): 
        # OCR Detection dataset 생성  
        try:  
            if os.path.isfile(output): os.remove(output)  
            with open(label_file) as f: 
                data = f.readlines()  
            mm_parse = {
                'metainfo': {
                    'dataset_type': 'TextDetDataset',   
                    'task_name': 'textdet', \
                    'category': [{'id': 0, 'name': 'text'}] 
                }, 
                'data_list': None 
            }     
            
            results = []  
            det_list = []   
            for d in data:  
                img_file = d.split('\t')[0] 
                label_list = d.split('\t')[1].strip()  
                if not os.path.isfile(os.path.join(img_dir, img_file)): 
                    continue 
                img = cv2.imread(os.path.join(img_dir, img_file))
                h, w, c = img.shape 

                fname, ext = img_file.split('.')[0], img_file.split('.')[1]  
                print(img_file) 

                if not os.path.isfile(os.path.join(img_dir, img_file)): 
                    print('Not found error {}: '.format(img_file))
                    continue  
                fname = fname.split('\\')[1] if os.name=='nt' else fname.split('/')[1]
                
                dicts = ['{' + x for x in label_list.replace('[{', '').replace('}]', '}').split(', {')]  
                if ocr_flag == 'P':   
                    ano = [] 
                    for seq, dic in enumerate(dicts):  
                        item = json.loads(dic) 
                        exclude = item['difficult']  
                        if not exclude:  
                            ano.append({
                                'transcription': item['transcription'] , 
                                'points': item['points'] 
                            })
                        else: 
                            ano.append({
                                'transcription': '###', 
                                'points': item['points'] 
                            })

                    label_str = json.dumps(ano, ensure_ascii=False) 
                    results.append(fname + '.' + ext + '\t' + label_str)
                else:   
                    ano = [] 
                    for seq, dic in enumerate(dicts):  
                        item = json.loads(dic) 
                        exclude = item['difficult']  
                        poly = item['points'][0] + item['points'][1] + item['points'][2] + item['points'][3]  
                        min_x1 = float((np.array(item['points'])[:,0]).min()) 
                        min_y1 = float((np.array(item['points'])[:,1]).min())  
                        max_x2 = float((np.array(item['points'])[:,0]).max()) 
                        max_y2 = float((np.array(item['points'])[:,1]).max())  
                        # "polygon": [0, 0, 0, 10, 10, 20, 20, 0], # [x1, y1, x2, y1, x2, y2, x1, y2] 
                        # "bbox": [0, 0, 10, 20], #  [x1, y1, x2, y2]  

                        if not exclude:  
                            ano.append({
                                'polygon': poly,  
                                'bbox':[min_x1, min_y1, max_x2, max_y2],  
                                'bbox_label': int(0), 
                                'ignore': False 
                            })
                        else: 
                            ano.append({
                                'polygon': poly,  
                                'bbox': [min_x1, min_y1, max_x2, max_y2], 
                                'bbox_label': int(0), 
                                'ignore': True
                            })

                    label_str = json.dumps(ano, ensure_ascii=False)  
                    det_list.append({
                        "instances": label_str,  
                        "img_path": fname + '.' + ext, 
                        "height": h,
                        "width": w , 
                        "seg_map": fname + '.txt'  
                    }) 
            
            if ocr_flag == 'P': 
                with open(output, 'w') as f:  
                    for d in results: 
                        f.write(d + '\n')
            else:  
                mm_parse['data_list'] = det_list  
                json_obj = json.dumps(mm_parse, ensure_ascii=False).replace("\\", "")    
                with open(output, 'w') as f: 
                    f.write(json_obj) 
                    # json.dump(mm_parse, f, indent=4)
    
            print('Created Detection Json file!')  

            return output, True  
        except Exception as ex: 
            print(repr(ex))
            return '', False  

    # ---------------------------------------------------------------- 
    def makeRecData(self):
        if os.name == 'nt': 
            img_dir = ("\\").join(self.lastOpenDir.split("\\"))  
        else: 
            img_dir = ('/').join(self.lastOpenDir.split('/'))    
        self.makeRecAugmentData(img_dir,  "rect_gt.txt")
        
    def makeRecAugmentData(self, img_dir, output): 
        try: 
            rec_images = [] 
            label_list = glob.glob(img_dir + "/rec_*.txt")
            for lbl in label_list: 
                with open(lbl, 'r') as f: 
                    labels = f.readlines() 
                for l in labels:  
                    rec_images.append(l)
                    
            rec_img_dir = os.path.join(img_dir, 'crop_img')  
            ret, output = self.rec_image_augment(rec_img_dir, rec_img_dir, rec_images)
            
            if ret:
                QMessageBox.information(self, "Information", "Rec 학습데이터가 생성되었습니다: \n{}".format(output)) 
            else: 
                QMessageBox.information(self, "Error", "Rec 학습데이터 생성 에러: \n{}".format(output) ) 
                
        except Exception as ex: 
            print(repr(ex))
            
    def rec_image_augment(self, img_dir=None, target_dir=None, label_list=None):   
        if not os.path.exists(target_dir): 
            os.mkdir(target_dir)
            
        try: 
            labels = []  
            for i, d in enumerate(label_list):  
                if len(d.split('\t')) < 2: 
                    continue 

                img_file = d.split('\t')[0].strip()
                label = d.split('\t')[1].strip()  
                labels.append(img_file + "\t" + label.replace('\n', '') )

                file_name = (img_file.split('.')[0])  
                file_ext = img_file.split('.')[1]  

                img = os.path.join(img_dir, img_file)
                if os.path.isfile(img):  
                    img = Image.open(img)
                    
                    img_aug = Contrast()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_contrast.' + file_ext))  
                    labels.append(file_name + '_contrast.' + file_ext + "\t" + label ) 

                    img_aug = Brightness()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_bright.' + file_ext))  
                    labels.append(file_name + '_bright.' + file_ext + "\t" + label )

                    img_aug = GaussianNoise()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_gnoise.' + file_ext))  
                    labels.append(file_name + '_gnoise.' + file_ext + "\t" + label )

                    img_aug = GlassBlur()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_glsblur.' + file_ext))  
                    labels.append(file_name + '_glsblur.' + file_ext + "\t" + label )
                    
                    img_aug = Sharpness()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_sharp.' + file_ext))  
                    labels.append(file_name + '_sharp.' + file_ext + "\t" + label )

                    img_aug = Color()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_color.' + file_ext))  
                    labels.append(file_name + '_color.' + file_ext + "\t" + label )

                    img_aug = Snow()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_snow.' + file_ext))  
                    labels.append(file_name + '_snow.' + file_ext + "\t" + label ) 
                    
                    # Fog, Frost, Snow, Rain, Shadow  
                    img_aug = Fog()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_fog.' + file_ext))  
                    labels.append(file_name + '_fog.' + file_ext + "\t" + label )  

                    img_aug = Shadow()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_shadow.' + file_ext))  
                    labels.append(file_name + '_shadow.' + file_ext + "\t" + label ) 

                    img_aug = Rain()(img)
                    img_aug.save(os.path.join(target_dir, file_name + '_rain.' + file_ext))  
                    labels.append(file_name + '_rain.' + file_ext + "\t" + label ) 

                    img_aug = Distortion_Sin(
                        img,
                        vertical=False,
                        horizontal=False
                    )
                    img_aug.save(os.path.join(target_dir, file_name + '_distort.' + file_ext))  
                    labels.append(file_name + '_distort.' + file_ext + "\t" + label )  

                else: 
                    print('No file: {}'.format(img))  

            output = os.path.join(target_dir, 'rec_label.txt') 
            with open(output, 'w') as f: 
                for lbl in labels: 
                    f.write(lbl + '\n')   
            return True, output  
        except Exception as ex: 
            print(repr(ex)) 
            return False, repr(ex)   
            
    # ------------------------------------------------------------------
    def autolcm(self):
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        self.panel = QLabel()
        self.panel.setText(self.stringBundle.getString('choseModelLg'))
        self.panel.setAlignment(Qt.AlignLeft)
        self.comboBox = QComboBox()
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(['Korean', 'English'])
        vbox.addWidget(self.panel)
        vbox.addWidget(self.comboBox)
        self.dialog = QDialog()
        self.dialog.resize(300, 100)
        self.okBtn = QPushButton(self.stringBundle.getString('ok'))
        self.cancelBtn = QPushButton(self.stringBundle.getString('cancel'))

        self.okBtn.clicked.connect(self.modelChoose)
        self.cancelBtn.clicked.connect(self.cancel)
        self.dialog.setWindowTitle(self.stringBundle.getString('choseModelLg'))

        hbox.addWidget(self.okBtn)
        hbox.addWidget(self.cancelBtn)

        vbox.addWidget(self.panel)
        vbox.addLayout(hbox)
        self.dialog.setLayout(vbox)
        self.dialog.setWindowModality(Qt.ApplicationModal)
        self.dialog.exec_()
        if self.filePath:
            self.AutoRecognition.setEnabled(True)
            self.actions.AutoRec.setEnabled(True) 

            self.AutoCurRecognition.setEnabled(True)
            self.actions.AutoCurRec.setEnabled(True)

    def modelChoose(self):
        print(self.comboBox.currentText())
        lg_idx = {'Korean': 'korean',             
            'English': 'en' }
        del self.ocr
        self.ocr = PaddleOCR(use_angle_cls=False, 
            det=True, 
            cls=False, 
            use_gpu=True, 
            lang=lg_idx[self.comboBox.currentText()])

        self.dialog.close()

    def cancel(self):
        self.dialog.close()

    def loadFilestate(self, saveDir):
        self.fileStatepath = saveDir + '/fileState.txt'
        self.fileStatedict = {}
        if not os.path.exists(self.fileStatepath):
            f = open(self.fileStatepath, 'w', encoding='utf-8')
        else:
            with open(self.fileStatepath, 'r', encoding='utf-8') as f:
                states = f.readlines()
                for each in states:
                    file, state = each.split('\t')
                    self.fileStatedict[file] = 1
                self.actions.saveLabel.setEnabled(True)  
                self.actions.exportDet.setEnabled(True) 
                self.actions.exportRec.setEnabled(True)
                self.actions.exportKie.setEnabled(True)

                self.actions.exportDetMM.setEnabled(True) 
                self.actions.exportRecMM.setEnabled(True)
                self.actions.exportKieMM.setEnabled(True)
                
                self.actions.makeRecData.setEnabled(True)  

    def saveFilestate(self):
        with open(self.fileStatepath, 'w', encoding='utf-8') as f:
            for key in self.fileStatedict:
                f.write(key + '\t')
                f.write(str(self.fileStatedict[key]) + '\n')

    def loadLabelFile(self, labelpath):
        labeldict = {}
        if not os.path.exists(labelpath):
            f = open(labelpath, 'w', encoding='utf-8')

        else:
            with open(labelpath, 'r', encoding='utf-8') as f:
                data = f.readlines()
                for each in data:
                    file, label = each.split('\t')
                    print(file)
                    if label:
                        label = label.replace('false', 'False')
                        label = label.replace('true', 'True')
                        labeldict[file] = eval(label)
                    else:
                        labeldict[file] = []
        return labeldict

    def savePPlabel(self, mode='Manual'):
        savedfile = [self.getImglabelidx(i) for i in self.fileStatedict.keys()]
        with open(self.PPlabelpath, 'w', encoding='utf-8') as f:
            for key in self.PPlabel:
                if key in savedfile and self.PPlabel[key] != []:
                    f.write(key + '\t')
                    f.write(json.dumps(self.PPlabel[key], ensure_ascii=False) + '\n')

        # 먼저 라벨파일 저장처리 
        savedfile = [self.getImglabelidx(i) for i in self.fileStatedict.keys()]
        with open(self.PPlabelpath, 'w', encoding='utf-8') as f:
            for key in self.PPlabel:
                if key in savedfile and self.PPlabel[key] != []:
                    f.write(key + '\t')
                    f.write(json.dumps(self.PPlabel[key], ensure_ascii=False) + '\n')

        if mode == 'Manual':
            msg = '체크된 이미지 저장: ' + self.PPlabelpath
            QMessageBox.information(self, "Information", msg)

    def saveCacheLabel(self):
        with open(self.Cachelabelpath, 'w', encoding='utf-8') as f:
            for key in self.Cachelabel:
                f.write(key + '\t')
                f.write(json.dumps(self.Cachelabel[key], ensure_ascii=False) + '\n')

    def saveLabelFile(self):
        self.saveFilestate()
        self.savePPlabel()

    
    def speedChoose(self):
        if self.labelDialogOption.isChecked():
            self.canvas.newShape.disconnect()
            self.canvas.newShape.connect(partial(self.newShape, True))

        else:
            self.canvas.newShape.disconnect()
            self.canvas.newShape.connect(partial(self.newShape, False))

    def autoSaveFunc(self):
        if self.autoSaveOption.isChecked():
            self.autoSaveNum = 1  # Real auto_Save
            try:
                self.saveLabelFile()
            except:
                pass
            print('The program will automatically save once after confirming an image')
        else:
            self.autoSaveNum = 5  # Used for backup
            print('The program will automatically save once after confirming 5 images (default)')

    def change_box_key(self):
        if not self.kie_mode:
            return
        key_text, _ = self.keyDialog.popUp(self.key_previous_text)
        if key_text is None:
            return
        self.key_previous_text = key_text
        for shape in self.canvas.selectedShapes:
            shape.key_cls = key_text
            if not self.keyList.findItemsByLabel(key_text):
                item = self.keyList.createItemFromLabel(key_text)
                self.keyList.addItem(item)
                rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                self.keyList.setItemLabel(item, key_text, rgb)

            self._update_shape_color(shape)
            self.keyDialog.addLabelHistory(key_text)

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.BoxList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)
        print("loadShapes")  # 1

    def lockSelectedShape(self):
        """lock the selected shapes.

        Add self.selectedShapes to lock self.canvas.lockedShapes, 
        which holds the ratio of the four coordinates of the locked shapes
        to the width and height of the image
        """
        width, height = self.image.width(), self.image.height()

        def format_shape(s):
            return dict(label=s.label,  # str
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        ratio=[[int(p.x()) / width, int(p.y()) / height] for p in s.points],  # QPonitF
                        difficult=s.difficult,  # bool
                        key_cls=s.key_cls,  # bool
                        )

        # lock
        if len(self.canvas.lockedShapes) == 0:
            for s in self.canvas.selectedShapes:
                s.line_color = DEFAULT_LOCK_COLOR
                s.locked = True
            shapes = [format_shape(shape) for shape in self.canvas.selectedShapes]
            trans_dic = []
            for box in shapes:
                trans_dict = {"transcription": box['label'], "ratio": box['ratio'], "difficult": box['difficult']}
                if self.kie_mode:
                    trans_dict.update({"key_cls": box["key_cls"]})
                trans_dic.append(trans_dict)
            self.canvas.lockedShapes = trans_dic
            self.actions.save.setEnabled(True)

        # unlock
        else:
            for s in self.canvas.shapes:
                s.line_color = DEFAULT_LINE_COLOR
            self.canvas.lockedShapes = []
            self.result_dic_locked = []
            self.setDirty()
            self.actions.save.setEnabled(True)


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra arguments to change predefined class file
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--lang", type=str, default='ko', nargs="?")
    arg_parser.add_argument("--gpu", type=str2bool, default=False, nargs="?")
    arg_parser.add_argument("--kie", type=str2bool, default=True, nargs="?")
    arg_parser.add_argument("--predefined_classes_file",
                            default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"),
                            nargs="?")
    args = arg_parser.parse_args(argv[1:])

    win = MainWindow(lang=args.lang,
                     gpu=args.gpu,
                     kie_mode=args.kie,
                     default_predefined_class_file=args.predefined_classes_file)
    win.show()
    return app, win


def main():
    """construct main app and run it"""
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':
    resource_file = './libs/resources[].py'
    if not os.path.exists(resource_file):
        output = os.system('pyrcc5 -o ./libs/resources.py resources.qrc')
        assert output == 0, "operate the cmd have some problems ,please check  whether there is a in the lib " \
                            "directory resources.py "
    import libs.resources
    sys.exit(main())
 