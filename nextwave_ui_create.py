from PyQt5.QtWidgets import (QWidget, QLabel, QScrollArea, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QGroupBox, QTabWidget, QPushButton, QCheckBox,
                             QLineEdit, QDoubleSpinBox, QSlider, QTextEdit, QShortcut)
from PyQt5.QtGui import QPixmap, QImage, QFont, QKeySequence
from PyQt5.QtCore import Qt
import PyQt5.QtGui as QtGui

import numpy as np

from pyqtgraph.parametertree import ParameterTree

from nextwave_widgets import ActuatorPlot, MyBarWidget

QIMAGE_HEIGHT = 1024
QIMAGE_WIDTH = 1024

MAIN_HEIGHT_WIN = 1024
MAIN_WIDTH_WIN = 1800

CAM_EXPO_MIN = 40. / 1000.0
CAM_EXPO_MAX = (100 * 1000) / 1000.00
CAM_GAIN_MIN = 0
CAM_GAIN_MAX = 9.83


def createUI(self):

    self.key_control = False

    self.setWindowIcon(QtGui.QIcon("./resources/wave_icon.png"))
    self.setWindowTitle('NextWave')
    #self.setWindowTitle("Icon")

    self.widget_centrals = QWidget()
    self.scroll_central = QScrollArea()
    layout=QVBoxLayout()
    pixmap_label = QLabel()
    #pixmap_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    #pixmap_label.resize(SPOTS_WIDTH_WIN,SPOTS_HEIGHT_WIN)
    pixmap_label.setAlignment(Qt.AlignCenter)
    self.pixmap_label=pixmap_label

    im_np = np.ones((QIMAGE_HEIGHT,QIMAGE_WIDTH),dtype='uint8')
    #im_np = np.transpose(im_np, (1,0,2))
    qimage = QImage(im_np, im_np.shape[1], im_np.shape[0],
                    QImage.Format_Mono)
    pixmap = QPixmap(qimage)
    #pixmap = pixmap.scaled(SPOTS_WIDTH_WIN,SPOTS_HEIGHT_WIN, Qt.KeepAspectRatio)
    pixmap_label.setPixmap(pixmap)
    pixmap_label.mousePressEvent = self.button_clicked

    #Scroll Area Properties
    self.scroll_central.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    self.scroll_central.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    self.scroll_central.setWidgetResizable(True)
    self.scroll_central.setWidget(self.pixmap_label)
    layout.addWidget(self.scroll_central,15)

    self.bar_plot = MyBarWidget()
    self.bar_plot.app = self
    layout.addWidget(self.bar_plot,5)
    self.widget_centrals.setLayout(layout)

    self.widget_displays = QWidget()
    layout=QVBoxLayout(self.widget_displays)
    layout.addWidget(QGroupBox('Pupil'))
    #layout.addWidget(QGroupBox('DM'))
    self.actuator_plot = ActuatorPlot(self)
    self.actuator_plot.resize(200,200)
    #layout.addWidget(QGroupBox('Wavefront'))
    #layout.addWidget(QGroupBox('PSF'))
    #self.ap = pg.plot()
    #self.ap.addItem( self.actuator_plot )
    layout.addWidget(self.actuator_plot)

    self.widget_controls = QWidget()
    layout=QVBoxLayout()
    tabs = QTabWidget()
    tabs.setTabPosition(QTabWidget.North)
    tabs.setMovable(True)
    self.tabs = tabs

    l1 = QHBoxLayout()

    self.widget_op = QWidget()
    layout_op = QVBoxLayout()
    self.ops_pupil = QGroupBox('Pupil')
    self.ops_pupil.setStyleSheet(":title {font-weight: bold}") # Doesn't work
    layout_op.addWidget(self.ops_pupil)
    self.ops_source = QGroupBox('Camera/Source')
    layout_op.addWidget(self.ops_source)
    self.ops_dm = QGroupBox('DM')
    layout_op.addWidget(self.ops_dm)
    self.widget_op.setLayout(layout_op)

    panel_names = ["Operation", "Settings", "Config", "Offline"]
    pages = [QWidget(tabs) for nam in panel_names]
    for n, tabnames in enumerate(panel_names):
        tabs.addTab(pages[n], tabnames)

    pages[0].setLayout(layout_op)
    ### Pupil ops
    layout1 = QGridLayout(self.ops_pupil)

    ### Arrows pad
    self.m = 1
    btnL = QPushButton("\u2190") # l
    layout1.addWidget(btnL,1,4)
    btnL.clicked.connect(lambda: self.move_center(-1,0) )
    btnU = QPushButton("\u2191") # u
    layout1.addWidget(btnU,0,5)
    btnU.clicked.connect(lambda: self.move_center(0,-1) )
    btnR = QPushButton("\u2192") # r
    layout1.addWidget(btnR,1,6)
    btnR.clicked.connect(lambda: self.move_center(1,0) )
    btnD = QPushButton("\u2193") # d
    layout1.addWidget(btnD,2,5)
    btnM = QPushButton() # d
    self.chkMove = QCheckBox("")
    layout1.addWidget(self.chkMove,1,5,alignment=Qt.AlignCenter)
    btnD.clicked.connect(lambda: self.move_center(0,1) )

    self.chkMove.clicked.connect(lambda: self.set_m(self.chkMove.isChecked()) )

    self.chkMoveTipTilt = QCheckBox("TT")
    layout1.addWidget(self.chkMoveTipTilt,0,6,alignment=Qt.AlignCenter)

    lbl = QLabel("Center X:")
    layout1.addWidget(lbl,0,0)
    lbl = QLabel("Center Y:")
    layout1.addWidget(lbl,1,0)
    lbl = QLabel("Diameter (mm):")
    layout1.addWidget(lbl,2,0)
    self.line_centerx = QLineEdit()
    self.line_centerx.setMaxLength(6)
    layout1.addWidget(self.line_centerx,0,1)
    self.line_centery = QLineEdit()
    self.line_centery.setMaxLength(6)
    layout1.addWidget(self.line_centery,1,1)
    self.line_pupil_diam = QLineEdit()
    self.line_pupil_diam.setMaxLength(6)
    layout1.addWidget(self.line_pupil_diam,2,1)

    self.line_pupil_diam.textChanged.connect(self.pupil_changed)
    #btnFind = QPushButton("Find center")
    #btnFind.setStyleSheet("color : orange")
    #layout1.addWidget(btnFind,2,1)

    self.chkRecord = QCheckBox("Rec")
    layout1.addWidget(self.chkRecord,0,4,alignment=Qt.AlignCenter)
    self.chkRecord.clicked.connect( self.toggle_record )

    self.it_start = QLineEdit("3.5")
    layout1.addWidget(self.it_start,4,0)
    self.it_step = QLineEdit("0.25")
    layout1.addWidget(self.it_step,4,1)
    self.it_stop = QLineEdit("6.4")
    layout1.addWidget(self.it_stop,4,2)

    btn = QPushButton("Start")
    layout1.addWidget(btn,4,3)
    btn.clicked.connect(lambda: self.engine.offline.offline_startbox() )

    btn = QPushButton("Run")
    layout1.addWidget(btn,4,4)
    btn.clicked.connect(lambda: self.iterative_run() )

    btn = QPushButton("Step")
    layout1.addWidget(btn,4,5)
    btn.clicked.connect(lambda: self.iterative_step() )

    btn = QPushButton("Reset")
    layout1.addWidget(btn,4,6)
    btn.clicked.connect(lambda: self.iterative_reset() )

    btn = QPushButton("\u2190") # left
    layout1.addWidget(btn,5,1)
    btn.clicked.connect(lambda: self.offline_move(-1,True) )

    self.lbl_frame_curr = QLabel()
    layout1.addWidget(self.lbl_frame_curr,5,0)

    btn = QPushButton("\u2192") # right
    layout1.addWidget(btn,5,2)
    btn.clicked.connect(lambda: self.offline_move (1,True) )

    btn = QPushButton("Autoall")
    layout1.addWidget(btn,5,5)
    btn.clicked.connect(lambda: self.engine.offline.offline_autoall() )

    btn = QPushButton("Save")
    layout1.addWidget(btn,5,6)
    btn.clicked.connect(lambda: self.engine.offline.offline_serialize() )

    btn = QPushButton("Dialog")
    layout1.addWidget(btn,5,4)
    btn.clicked.connect(lambda: self.engine.offline.show_dialog() )

    #btnIt1 = QPushButton("Step It+=0.5")
    #layout1.addWidget(btnIt1,3,1)
    #btnIt1.clicked.connect(self.run_iterative)

    #self.lblIt = QLabel("3.2")
    #layout1.addWidget(self.lblIt,4,5)

    ### Camera Ops
    layout1 = QGridLayout(self.ops_source)

    self.chkBackSubtract = QCheckBox("Subtract background")
    self.chkBackSubtract.stateChanged.connect(self.sub_background)
    layout1.addWidget(self.chkBackSubtract,0,0)
    btnBackSet = QPushButton("Set background")
    layout1.addWidget(btnBackSet,0,1)
    btnBackSet.clicked.connect(self.set_background)
    self.chkReplaceSubtract = QCheckBox("Replace subtracted")
    self.chkReplaceSubtract.stateChanged.connect(self.replace_background)
    layout1.addWidget(self.chkReplaceSubtract,0,2)
    btn1 = QPushButton("Background AUTO")
    layout1.addWidget(btn1,0,3)
    btn1.clicked.connect(self.set_background_auto)

    self.slider_threshold = QSlider(orientation=Qt.Horizontal)
    self.slider_threshold.setMinimum(0) # TODO: Get from camera
    self.slider_threshold.setMaximum(100) # TODO: Get from camera
    layout1.addWidget(self.slider_threshold,1,1)
    self.slider_threshold.valueChanged.connect(self.slider_threshold_changed) # TODO

    self.chkApplyThreshold = QCheckBox("Apply Thresholding")
    self.chkApplyThreshold.stateChanged.connect(self.click_apply_threshold)
    layout1.addWidget(self.chkApplyThreshold,1,0)

    self.threshold_val = QDoubleSpinBox()
    layout1.addWidget(self.threshold_val,1,2)
    self.threshold_val.setDecimals(2)

    lbl = QLabel("Exposure time (ms)")
    layout1.addWidget(lbl,2,0)

    self.slider_exposure = QSlider(orientation=Qt.Horizontal)
    self.slider_exposure.setMinimum(0) # TODO: Get from camera
    self.slider_exposure.setMaximum(100) # TODO: Get from camera
    layout1.addWidget(self.slider_exposure,2,1)
    self.slider_exposure.valueChanged.connect(self.slider_exposure_changed)

    self.exposure = QDoubleSpinBox()
    layout1.addWidget(self.exposure,2,2)
    self.exposure.setDecimals(4)
    self.exposure.setMinimum(CAM_EXPO_MIN)
    self.exposure.setMaximum(CAM_EXPO_MAX)

    lbl = QLabel("Gain (dB)")
    layout1.addWidget(lbl,3,0)

    self.slider_gain = QSlider(orientation=Qt.Horizontal)
    self.slider_gain.setMinimum(0) # TODO: Get from camera
    self.slider_gain.setMaximum(95) # TODO: Get from camera
    layout1.addWidget(self.slider_gain,3,1)
    self.slider_gain.valueChanged.connect(self.slider_gain_changed)

    self.gain = QDoubleSpinBox()
    layout1.addWidget(self.gain,3,2)
    self.gain.setMinimum(CAM_GAIN_MIN)
    self.gain.setMaximum(CAM_GAIN_MAX)

    ### DM Ops
    layout1 = QGridLayout(self.ops_dm)
    layout1.addWidget(self.chkLoop,0,0) # It's created above.. Need for order
    self.chkLoop.stateChanged.connect(self.loop_changed)

    btn = QPushButton("Save flat")
    btn.clicked.connect(self.flat_save)
    layout1.addWidget(btn, 0,3 )

    btn = QPushButton("Do flat")
    btn.clicked.connect(self.flat_do)
    layout1.addWidget(btn, 0,1 )

    btn = QPushButton("Do zero")
    btn.clicked.connect(self.zero_do)
    layout1.addWidget(btn, 0,2 )

    btn = QPushButton("Search box shift")
    btn.clicked.connect(lambda: self.show_zernike_dialog("Shift search boxes", self.engine.shift_search_boxes ) )
    layout1.addWidget(btn, 1,0 )

    btn = QPushButton("Reference shift")
    btn.clicked.connect(lambda: self.show_zernike_dialog("Shift references", self.engine.shift_references ) )
    layout1.addWidget(btn, 2,0 )

    self.chkActivateMetric = QCheckBox("Activate Metric")
    self.chkActivateMetric.stateChanged.connect(self.activate_metric)
    layout1.addWidget(self.chkActivateMetric, 8,2 )

    self.chkRemoveTipTilt = QCheckBox("Remove tiptilt")
    self.chkRemoveTipTilt.setChecked(True)
    self.chkRemoveTipTilt.stateChanged.connect(self.do_remove_tiptilt)
    layout1.addWidget(self.chkRemoveTipTilt, 8,3 )

    self.chkModalEdges = QCheckBox("Modal edges")
    self.chkModalEdges.setChecked(False)
    self.chkModalEdges.stateChanged.connect(self.do_modal_edges)
    layout1.addWidget(self.chkModalEdges, 9,3 )

#     self.chkBackSubtract = QCheckBox("Subtract background")
    #btn.clicked.connect(self.activate_metric)
#     self.chkBackSubtract.stateChanged.connect(self.sub_background)
#     layout1.addWidget(self.chkBackSubtract,0,0)


    self.chkFollow = QCheckBox("Boxes follow centroids")
    self.chkFollow.stateChanged.connect(lambda:self.set_follow(self.chkFollow.isChecked()))
    layout1.addWidget(self.chkFollow, 1,3 )

    btn = QPushButton("Load mirror file")
    btn.clicked.connect(self.load_mirror_file )
    layout1.addWidget(btn, 1,2 )

    btn = QPushButton("Search box RESET")
    btn.clicked.connect(self.engine.reset_search_boxes )
    layout1.addWidget(btn, 1,1 )

    btn = QPushButton("Reference RESET")
    btn.clicked.connect(self.engine.reset_references )
    layout1.addWidget(btn, 2,1 )

    btn = QPushButton("+")
    btn.clicked.connect(self.engine.defocus_plus )
    layout1.addWidget(btn, 2,3 )

    btn = QPushButton("-")
    btn.clicked.connect(self.engine.defocus_minus )
    layout1.addWidget(btn, 2,2 )

    self.label_defocus = QLabel("Defocus: ")
    layout1.addWidget(self.label_defocus, 3,3)
    self.slider_defocus = QSlider(orientation=Qt.Horizontal)
    self.slider_defocus.setMinimum(0)
    self.slider_defocus.setMaximum(1000)
    self.slider_defocus.setValue(500)
    layout1.addWidget(self.slider_defocus,3,2)
    self.slider_defocus.valueChanged.connect(self.slider_defocus_changed)

    self.label_aobleed = QLabel("AO Bleed: ")
    layout1.addWidget(self.label_aobleed, 4,0)
    self.slider_aobleed = QSlider(orientation=Qt.Horizontal)
    self.slider_aobleed.setMinimum(0)
    self.slider_aobleed.setMaximum(20)
    self.slider_aobleed.setValue(0)
    layout1.addWidget(self.slider_aobleed,4,1)
    self.slider_aobleed.valueChanged.connect(self.slider_aobleed_changed)


    self.label_aogain = QLabel("AO Gain: ")
    layout1.addWidget(self.label_aogain, 4,3)
    self.slider_aogain = QSlider(orientation=Qt.Horizontal)
    self.slider_aogain.setMinimum(0)
    self.slider_aogain.setMaximum(100)
    layout1.addWidget(self.slider_aogain,4,2)
    self.slider_aogain.valueChanged.connect(self.slider_aogain_changed)

    self.label_aorate = QLabel("AO Delay: ")
    layout1.addWidget(self.label_aorate, 3,0)
    self.slider_aorate = QSlider(orientation=Qt.Horizontal)
    self.slider_aorate.setMinimum(1)
    self.slider_aorate.setMaximum(100)
    layout1.addWidget(self.slider_aorate,3,1)
    self.slider_aorate.valueChanged.connect(self.slider_aorate_changed)

    self.label_dmfill = QLabel("DM Fill: ")
    layout1.addWidget(self.label_dmfill, 5,3)
    self.slider_dmfill = QSlider(orientation=Qt.Horizontal)
    self.slider_dmfill.setMinimum(0)
    self.slider_dmfill.setMaximum(1000)
    layout1.addWidget(self.slider_dmfill,5,2)
    self.slider_dmfill.valueChanged.connect(self.slider_dmfill_changed)

    self.modes = QDoubleSpinBox()
    layout1.addWidget(self.modes,5,0)
    self.modes.setMinimum(1)
    self.modes.setMaximum(97)
    self.modes.setValue(97)
    self.modes.valueChanged.connect(self.engine.modes_set)
    self.label_valid_acts = QLabel("Valid Acts:")
    layout1.addWidget(self.label_valid_acts, 5,1)
    self.label_condition = QLabel("Cond #:")
    layout1.addWidget(self.label_condition, 6,0)

    btn1 = QPushButton("Rebuild")
    layout1.addWidget(btn1,6,1)
    btn1.clicked.connect(self.engine.update_influence)

    # Not used
    #btn = QPushButton("Defocus start")
    #btn.clicked.connect(self.engine.defocus_start )
    #layout1.addWidget(btn, 3,2 )

    btn1 = QPushButton("Apply Zs.")
    layout1.addWidget(btn1,6,3)
#     btn1.clicked.connect(self.engine.apply_zernikes)
    btn1.clicked.connect(lambda: self.show_zernike_dialog("Apply Zs", self.engine.apply_zernikes ) )

    self.chkZonal = QCheckBox("Modal Correction")
    #self.chkFollow.stateChanged.connect(lambda:self.set_follow(self.chkFollow.isChecked()))
    layout1.addWidget(self.chkZonal, 7,3 )

    self.zs_for_apply = QDoubleSpinBox()
    self.zs_for_apply.setMinimum(1)
    self.zs_for_apply.setMaximum(65)
    self.zs_for_apply.setValue(25)
    layout1.addWidget(self.zs_for_apply,6,2)

    self.zonal_modes = QDoubleSpinBox()
    self.zonal_modes.setMinimum(1)
    self.zonal_modes.setMaximum(97)
    self.zonal_modes.setValue(85)
    layout1.addWidget(self.zonal_modes,7,2)

    self.label_metric = QLabel("Metric Threshold: ")
    layout1.addWidget(self.label_metric, 8,0)
    self.slider_metric = QSlider(orientation=Qt.Horizontal)
    self.slider_metric.setMinimum(1)
    self.slider_metric.setMaximum(100)
    self.slider_metric.setValue(50)
    layout1.addWidget(self.slider_metric,8,1)
    self.slider_metric.valueChanged.connect(self.slider_metric_changed)

    self.widget_mode_buttons = QWidget()
    layoutStatusButtons = QHBoxLayout(self.widget_mode_buttons)

    self.mode_btn1 = QPushButton("Init")
    layoutStatusButtons.addWidget(self.mode_btn1)
    self.mode_btn1.clicked.connect(self.mode_init)

    self.mode_btn2 = QPushButton("Snap")
    layoutStatusButtons.addWidget(self.mode_btn2)
    self.mode_btn2.clicked.connect(self.mode_snap)

    self.mode_btn3 = QPushButton("Run")
    layoutStatusButtons.addWidget(self.mode_btn3)
    self.mode_btn3.clicked.connect(self.mode_run)

    self.mode_btn_ao1 = QPushButton("Run1")
    layoutStatusButtons.addWidget(self.mode_btn_ao1)
    self.mode_btn_ao1.clicked.connect(self.engine.do_ao1)

    self.mode_btn4 = QPushButton("Stop")
    layoutStatusButtons.addWidget(self.mode_btn4)
    self.mode_btn4.clicked.connect(self.mode_stop)

    self.edit_num_runs = QLineEdit("999999")
    self.edit_num_runs.setMaxLength(6)
    layoutStatusButtons.addWidget(self.edit_num_runs)

    self.mode_btn2.setEnabled( True )
    self.mode_btn3.setEnabled( True )
    #self.mode_btn4.setEnabled( False )

    btn = QPushButton("Reset Counts")
    layoutStatusButtons.addWidget(btn)
    btn.clicked.connect(self.reset_counts)

    # Config
    layout1 = QGridLayout(pages[2])
    lbl = QLabel("XML Config: ")
    layout1.addWidget(lbl, 0,0)
    self.edit_xml_filename = QLineEdit(self.json_data["params"]["xml_file"])
    layout1.addWidget(self.edit_xml_filename, 0,1)
    btn = QPushButton("Select")
    layout1.addWidget(btn, 0,2)
    btn.clicked.connect(self.load_config)
    #btn = QPushButton("Edit")
    #layout1.addWidget(btn, 0,3)
    #btn.clicked.connect(self.reload_config)
    self.layout_config = layout1
    self.init_config_ui()

    # Settings
    layout1 = QGridLayout(pages[1])
    self.param_tree = ParameterTree()
    self.param_tree.setParameters(self.p, showTop=False)
    layout1.addWidget(self.param_tree,0,0)
    btn = QPushButton("Apply")
    btn.clicked.connect(self.params_apply_clicked)
    layout1.addWidget(btn,1,0)

    #self.widget_status_buttons.setLayout(layoutStatusButtons)
    layout.addWidget(self.widget_mode_buttons,1)

    self.label_status0 = QLabel("Status: ")
    layout.addWidget(self.label_status0, 1)

    self.text_status = QTextEdit()
    self.text_status.setReadOnly(True)

    #layout.addWidget(self.text_status, 1)
    layout.addWidget(tabs, 20)

    #self.widget_controls = QGroupBox('Controls')
    self.widget_controls.setLayout(layout)

    font=QFont("Courier",18,QFont.Bold);
    #font.setStyleHint(QFont::TypeWriter);

    #layout.addWidget(QGroupBox('Statistics'), 20)
    self.text_stats = QTextEdit()
    self.text_stats.setCurrentFont(font)
    self.text_stats.setReadOnly(True)
    layout.addWidget(self.text_stats)

    # OFFLINE
    layout1 = QGridLayout(pages[3])
    #btn1 = QPushButton("Load spot image")
    #btn1.setStyleSheet("color : orange")
    #layout1.addWidget(btn1,3,0,0,-1)
    #btn1.clicked.connect(self.offline_load_image)

    #lbl = QLabel("Spot image: ")
    #layout1.addWidget(lbl, 1,0)
    #self.offline_image_name = QLineEdit("spots.bin")
    #layout1.addWidget(self.offline_image_name, 1,0)
    self.btn_off = QPushButton("Load Offline Source")
    self.btn_off.clicked.connect(self.offline_load_image)
    layout1.addWidget(self.btn_off, 1,0)

    self.btn_off_back = QPushButton("Load Offline Background")
    self.btn_off_back.clicked.connect(self.offline_load_background)
    layout1.addWidget(self.btn_off_back, 2,0)

    # Offline scroll image:
    self.scroll_off = QScrollArea()
    self.layout_off = QGridLayout()

    #Scroll Area Properties
    self.scroll_off.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    self.scroll_off.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    self.scroll_off.setWidgetResizable(True)

    self.widget_off = QWidget()
    self.scroll_off.setWidget(self.widget_off)
    self.widget_off.setLayout(self.layout_off)

    layout1.addWidget(self.scroll_off,3,0) #,-1,-1)

    #btn = QPushButton("\u2190") # left
    #layout1.addWidget(btn,4,0)
    #btn.clicked.connect(lambda: self.offline_move(-1) )
    #btn = QPushButton("\u2192") # right
    #layout1.addWidget(btn,4,1)
    #btn.clicked.connect(lambda: self.offline_move (1) )

    self.chkOfflineAlgorithm = QCheckBox("Use offline algorithm")
    self.chkOfflineAlgorithm.stateChanged.connect(self.offline_algorithm)
    layout1.addWidget(self.chkOfflineAlgorithm,5,0)

    self.param_tree_offline = ParameterTree()
    self.param_tree_offline.setParameters(self.p_offline, showTop=False)

    # Main Widget
    self.widget_main = QWidget()
    layoutCentral = QHBoxLayout()
    layoutCentral.addWidget(self.widget_centrals, stretch=3)
    layoutCentral.addWidget(self.widget_displays, stretch=2)
    layoutCentral.addWidget(self.widget_controls, stretch=1)
    self.widget_main.setLayout(layoutCentral)

    self.setCentralWidget(self.widget_main)

    menu=self.menuBar().addMenu('&File')
    menu.addAction('&Export Centroids + Zernikes', self.export)
    menu.addAction('Export All &Zernikes', self.export_all)
    menu.addAction('Run &Calibration', self.do_calibration)
    menu.addAction('e&Xit', self.close)

    pixmap_label.setFocus()
    self.shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
    self.shortcut.activated.connect(self.mode_snap)
    self.shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
    self.shortcut.activated.connect(self.mode_stop)
    self.shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
    self.shortcut.activated.connect(self.mode_run)

    self.shortcut = QShortcut(QKeySequence("Ctrl+1"), self)
    self.shortcut.activated.connect(self.engine.defocus_minus10)
    self.shortcut = QShortcut(QKeySequence("Ctrl+2"), self)
    self.shortcut.activated.connect(self.engine.defocus_minus01)

    self.shortcut = QShortcut(QKeySequence("Ctrl+3"), self)
    self.shortcut.activated.connect(self.engine.defocus_plus01)
    self.shortcut = QShortcut(QKeySequence("Ctrl+4"), self)
    self.shortcut.activated.connect(self.engine.defocus_plus10)

    self.setGeometry(2,2,MAIN_WIDTH_WIN,MAIN_HEIGHT_WIN)
    self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
    self.show()
