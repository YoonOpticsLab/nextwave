from PyQt5.QtWidgets import (QWidget, QLabel, QScrollArea, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QGroupBox, QTabWidget, QPushButton, QCheckBox,
                             QLineEdit, QDoubleSpinBox, QSlider, QTextEdit, QShortcut)
from PyQt5.QtGui import QPixmap, QImage, QFont, QKeySequence
from PyQt5.QtCore import Qt
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

from PyQt5.QtCore import QTimer

import numpy as np

from pyqtgraph.parametertree import ParameterTree

from nextwave_widgets import ActuatorPlot, MyBarWidget, FrameGalleryModel, FrameSlider
import defaults

# From nextwave_defaults.py if it has them
QIMAGE_HEIGHT = getattr(defaults, 'QIMAGE_HEIGHT', 1024)
QIMAGE_WIDTH = getattr(defaults, 'QIMAGE_WIDTH', 1024)

MAIN_HEIGHT_WIN = getattr(defaults, 'MAIN_HEIGHT_WIN', 1024)
MAIN_WIDTH_WIN = getattr(defaults, 'MAIN_WIDTH_WIN', 1800)

CAM_EXPO_MIN = getattr(defaults, 'CAM_EXPO_MIN', 40. / 1000.0)
CAM_EXPO_MAX = getattr(defaults, 'CAM_EXPO_MAX', (100 * 1000) / 1000.00)
CAM_GAIN_MIN = getattr(defaults, 'CAM_GAIN_MIN', 0)
CAM_GAIN_MAX = getattr(defaults, 'CAM_GAIN_MAX', 9.83)


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
    pixmap_label.mousePressEvent = self.spot_window_clicked
    pixmap_label.setMouseTracking(True) # Move events without a button pressed: for the cross section
    pixmap_label.mouseMoveEvent = self.spot_window_mouse_moved
    pixmap_label.leaveEvent = self.spot_window_left

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

    # The DM display (realtime: it's shown only then, see "Main Widget")
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

    # Operation tab: Pupil and Boxes/Centroids; then Camera/Source and DM (realtime), or Offline (offline_only). In realtime the
    # Offline controls are in their own tab
    self.widget_op = QWidget()
    layout_op = QVBoxLayout()
    self.ops_pupil = QGroupBox('Pupil')
    self.ops_pupil.setStyleSheet(":title {font-weight: bold}") # Doesn't work
    layout_op.addWidget(self.ops_pupil)
    self.ops_source = QGroupBox('Camera/Source')
    self.ops_dm = QGroupBox('DM')
    self.ops_boxes = QGroupBox('Boxes/Centroids')
    self.ops_offline = QGroupBox('Offline')
    layout_op.addWidget(self.ops_boxes)
    if not self.offline_only:
        layout_op.addWidget(self.ops_source)
        layout_op.addWidget(self.ops_dm)
    else:
        layout_op.addWidget(self.ops_offline)
    self.widget_op.setLayout(layout_op)

    panel_names = ["Operation", "Settings", "Config"] + ([] if self.offline_only else ["Offline"])
    pages = [QWidget(tabs) for nam in panel_names]
    for n, tabnames in enumerate(panel_names):
        tabs.addTab(pages[n], tabnames)

    pages[0].setLayout(layout_op)
    ### Pupil ops
    layout1 = QGridLayout(self.ops_pupil)
    y_curr = 0

    ### Arrows pad + center labels (row 0)
    self.m = 1
    btnU = QPushButton("↑") # u
    btnU.clicked.connect(lambda: self.move_center(0,-1) )
    layout1.addWidget(btnU, y_curr,5)
    self.chkMoveTipTilt = QCheckBox("TT") # (Realtime) The arrows move by tip/tilt, in the centroiding plugin
    layout1.addWidget(self.chkMoveTipTilt, y_curr,6, alignment=Qt.AlignCenter)
    lbl = QLabel("Center X:")
    layout1.addWidget(lbl, y_curr,0)
    self.line_centerx = QLineEdit()
    self.line_centerx.setMaxLength(6)
    layout1.addWidget(self.line_centerx, y_curr,1)
    self.chkRecord = QCheckBox("Rec")
    self.chkRecord.clicked.connect(self.toggle_record)
    layout1.addWidget(self.chkRecord, y_curr,4, alignment=Qt.AlignCenter)
    if self.offline_only:
        self.chkMoveTipTilt.hide() # (Realtime only)
        self.chkRecord.hide()
    y_curr += 1

    ### Arrows pad + center labels (row 1)
    btnL = QPushButton("←") # l
    btnL.clicked.connect(lambda: self.move_center(-1,0) )
    layout1.addWidget(btnL, y_curr,4)
    self.chkMove = QCheckBox("")
    self.chkMove.clicked.connect(lambda: self.set_move_amount(self.chkMove.isChecked()) )
    layout1.addWidget(self.chkMove, y_curr,5, alignment=Qt.AlignCenter)
    btnR = QPushButton("→") # r
    btnR.clicked.connect(lambda: self.move_center(1,0) )
    layout1.addWidget(btnR, y_curr,6)
    lbl = QLabel("Center Y:")
    layout1.addWidget(lbl, y_curr,0)
    self.line_centery = QLineEdit()
    self.line_centery.setMaxLength(6)
    layout1.addWidget(self.line_centery, y_curr,1)
    y_curr += 1

    ### Arrows pad + center labels (row 2)
    btnD = QPushButton("↓") # d
    btnD.clicked.connect(lambda: self.move_center(0,1) )
    layout1.addWidget(btnD, y_curr,5)
    lbl = QLabel("Diameter (mm):")
    layout1.addWidget(lbl, y_curr,0)
    self.line_pupil_diam = QLineEdit()
    self.line_pupil_diam.setMaxLength(6)
    self.line_pupil_diam.textChanged.connect(self.pupil_changed)
    layout1.addWidget(self.line_pupil_diam, y_curr,1)
    y_curr += 1

    ### Iterative run controls
    self.it_start = QLineEdit(str(defaults.ITERATIVE_PUPIL_START))
    layout1.addWidget(self.it_start, y_curr,0)
    self.it_step = QLineEdit(str(defaults.ITERATIVE_PUPIL_STEP_SIZE))
    layout1.addWidget(self.it_step, y_curr,1)
    self.it_stop = QLineEdit(str(self.engine.offline.it_stop)) # (Forced value, if any, else the default)
    layout1.addWidget(self.it_stop, y_curr,2)
    self.it_start.textEdited.connect(self.it_start_edited)
    self.it_step.textEdited.connect(self.it_step_edited)
    self.it_stop.textEdited.connect(self.it_stop_edited)
    self.it_stop.editingFinished.connect(self.it_stop_changed)
    btn = QPushButton("Start")
    btn.clicked.connect(lambda: self.engine.offline.offline_startbox() )
    layout1.addWidget(btn, y_curr,3)
    btn = QPushButton("Run")
    btn.clicked.connect(lambda: self.iterative_run() )
    layout1.addWidget(btn, y_curr,4)
    btn = QPushButton("Step")
    btn.clicked.connect(lambda: self.iterative_step() )
    layout1.addWidget(btn, y_curr,5)
    btn = QPushButton("Reset")
    btn.clicked.connect(lambda: self.iterative_reset() )
    layout1.addWidget(btn, y_curr,6)
    y_curr += 1

    ### Camera Ops
    layout1 = QGridLayout(self.ops_source)
    y_curr = 0

    ### Background controls
    self.chkBackSubtract = QCheckBox("Subtract background")
    self.chkBackSubtract.stateChanged.connect(self.sub_background)
    layout1.addWidget(self.chkBackSubtract, y_curr,0)
    btnBackSet = QPushButton("Set background")
    btnBackSet.clicked.connect(self.set_background)
    layout1.addWidget(btnBackSet, y_curr,1)
    self.chkReplaceSubtract = QCheckBox("Replace subtracted")
    self.chkReplaceSubtract.stateChanged.connect(self.replace_background)
    layout1.addWidget(self.chkReplaceSubtract, y_curr,2)
    btn1 = QPushButton("Background AUTO")
    btn1.clicked.connect(self.set_background_auto)
    layout1.addWidget(btn1, y_curr,3)
    y_curr += 1

    ### Threshold controls
    self.chkApplyThreshold = QCheckBox("Apply Thresholding")
    self.chkApplyThreshold.stateChanged.connect(self.click_apply_threshold)
    layout1.addWidget(self.chkApplyThreshold, y_curr,0)
    self.slider_threshold = QSlider(orientation=Qt.Horizontal)
    self.slider_threshold.setMinimum(0) # TODO: Get from camera
    self.slider_threshold.setMaximum(100) # TODO: Get from camera
    self.slider_threshold.valueChanged.connect(self.slider_threshold_changed)
    layout1.addWidget(self.slider_threshold, y_curr,1)
    self.threshold_val = QDoubleSpinBox()
    self.threshold_val.setDecimals(2)
    layout1.addWidget(self.threshold_val, y_curr,2)
    y_curr += 1

    ### Exposure controls
    lbl = QLabel("Exposure time (ms)")
    layout1.addWidget(lbl, y_curr,0)
    self.slider_exposure = QSlider(orientation=Qt.Horizontal)
    self.slider_exposure.setMinimum(0) # TODO: Get from camera
    self.slider_exposure.setMaximum(100) # TODO: Get from camera
    self.slider_exposure.valueChanged.connect(self.slider_exposure_changed)
    layout1.addWidget(self.slider_exposure, y_curr,1)
    self.exposure = QDoubleSpinBox()
    self.exposure.setDecimals(4)
    self.exposure.setMinimum(CAM_EXPO_MIN)
    self.exposure.setMaximum(CAM_EXPO_MAX)
    layout1.addWidget(self.exposure, y_curr,2)
    y_curr += 1

    ### Gain controls
    lbl = QLabel("Gain (dB)")
    layout1.addWidget(lbl, y_curr,0)
    self.slider_gain = QSlider(orientation=Qt.Horizontal)
    self.slider_gain.setMinimum(0) # TODO: Get from camera
    self.slider_gain.setMaximum(95) # TODO: Get from camera
    self.slider_gain.valueChanged.connect(self.slider_gain_changed)
    layout1.addWidget(self.slider_gain, y_curr,1)
    self.gain = QDoubleSpinBox()
    self.gain.setMinimum(CAM_GAIN_MIN)
    self.gain.setMaximum(CAM_GAIN_MAX)
    layout1.addWidget(self.gain, y_curr,2)
    y_curr += 1

    ### DM Ops
    layout1 = QGridLayout(self.ops_dm)
    y_curr = 0

    ### Loop / flat controls
    layout1.addWidget(self.chkLoop, y_curr,0) # It's created above.. Need for order
    self.chkLoop.stateChanged.connect(self.loop_changed)
    btn = QPushButton("Do flat")
    btn.clicked.connect(self.flat_do)
    layout1.addWidget(btn, y_curr,1)
    btn = QPushButton("Do zero")
    btn.clicked.connect(self.zero_do)
    layout1.addWidget(btn, y_curr,2)
    btn = QPushButton("Save flat")
    btn.clicked.connect(self.flat_save)
    layout1.addWidget(btn, y_curr,3)
    y_curr += 1

    ### Mirror file / defocus (the search box and reference controls are in Boxes/Centroids)
    btn = QPushButton("Load mirror file")
    btn.clicked.connect(self.load_mirror_file)
    layout1.addWidget(btn, y_curr,0)
    btn = QPushButton("-")
    btn.clicked.connect(self.engine.defocus_minus)
    layout1.addWidget(btn, y_curr,2)
    btn = QPushButton("+")
    btn.clicked.connect(self.engine.defocus_plus)
    layout1.addWidget(btn, y_curr,3)
    y_curr += 1

    ### AO Delay / Defocus sliders
    self.label_aorate = QLabel("AO Delay: ")
    layout1.addWidget(self.label_aorate, y_curr,0)
    self.slider_aorate = QSlider(orientation=Qt.Horizontal)
    self.slider_aorate.setMinimum(1)
    self.slider_aorate.setMaximum(100)
    self.slider_aorate.valueChanged.connect(self.slider_aorate_changed)
    layout1.addWidget(self.slider_aorate, y_curr,1)
    self.slider_defocus = QSlider(orientation=Qt.Horizontal)
    self.slider_defocus.setMinimum(0)
    self.slider_defocus.setMaximum(2000)
    self.slider_defocus.setValue(1000)
    self.slider_defocus.valueChanged.connect(self.slider_defocus_changed)
    layout1.addWidget(self.slider_defocus, y_curr,2)
    self.label_defocus = QLabel("Defocus: ")
    layout1.addWidget(self.label_defocus, y_curr,3)
    y_curr += 1

    ### AO Bleed / Gain sliders
    self.label_aobleed = QLabel("AO Bleed: ")
    layout1.addWidget(self.label_aobleed, y_curr,0)
    self.slider_aobleed = QSlider(orientation=Qt.Horizontal)
    self.slider_aobleed.setMinimum(0)
    self.slider_aobleed.setMaximum(20)
    self.slider_aobleed.setValue(0)
    self.slider_aobleed.valueChanged.connect(self.slider_aobleed_changed)
    layout1.addWidget(self.slider_aobleed, y_curr,1)
    self.slider_aogain = QSlider(orientation=Qt.Horizontal)
    self.slider_aogain.setMinimum(0)
    self.slider_aogain.setMaximum(100)
    self.slider_aogain.valueChanged.connect(self.slider_aogain_changed)
    layout1.addWidget(self.slider_aogain, y_curr,2)
    self.label_aogain = QLabel("AO Gain: ")
    layout1.addWidget(self.label_aogain, y_curr,3)
    y_curr += 1

    ### Modes / DM Fill
    self.modes = QDoubleSpinBox()
    self.modes.setMinimum(1)
    self.modes.setMaximum(97)
    self.modes.setValue(97)
    self.modes.valueChanged.connect(self.engine.modes_set)
    layout1.addWidget(self.modes, y_curr,0)
    self.label_valid_acts = QLabel("Valid Acts:")
    layout1.addWidget(self.label_valid_acts, y_curr,1)
    self.slider_dmfill = QSlider(orientation=Qt.Horizontal)
    self.slider_dmfill.setMinimum(0)
    self.slider_dmfill.setMaximum(1000)
    self.slider_dmfill.valueChanged.connect(self.slider_dmfill_changed)
    layout1.addWidget(self.slider_dmfill, y_curr,2)
    self.label_dmfill = QLabel("DM Fill: ")
    layout1.addWidget(self.label_dmfill, y_curr,3)
    y_curr += 1

    ### Condition / rebuild / apply Zernikes
    self.label_condition = QLabel("Cond #:")
    layout1.addWidget(self.label_condition, y_curr,0)
    btn1 = QPushButton("Rebuild")
    btn1.clicked.connect(self.engine.update_influence)
    layout1.addWidget(btn1, y_curr,1)
    self.zs_for_apply = QDoubleSpinBox()
    self.zs_for_apply.setMinimum(1)
    self.zs_for_apply.setMaximum(65)
    self.zs_for_apply.setValue(25)
    layout1.addWidget(self.zs_for_apply, y_curr,2)
    btn1 = QPushButton("Apply Zs.")
    btn1.clicked.connect(lambda: self.show_zernike_dialog("Apply Zs", self.engine.apply_zernikes ) )
    layout1.addWidget(btn1, y_curr,3)
    y_curr += 1

    ### Zonal modes / modal correction
    self.zonal_modes = QDoubleSpinBox()
    self.zonal_modes.setMinimum(1)
    self.zonal_modes.setMaximum(97)
    self.zonal_modes.setValue(85)
    layout1.addWidget(self.zonal_modes, y_curr,2)
    self.chkZonal = QCheckBox("Modal Correction")
    layout1.addWidget(self.chkZonal, y_curr,3)
    y_curr += 1

    ### Metric threshold / tiptilt / metric activation
    self.label_metric = QLabel("Metric Threshold: ")
    layout1.addWidget(self.label_metric, y_curr,0)
    self.slider_metric = QSlider(orientation=Qt.Horizontal)
    self.slider_metric.setMinimum(1)
    self.slider_metric.setMaximum(1000)
    self.slider_metric.setValue(200)
    self.slider_metric.valueChanged.connect(self.slider_metric_changed)
    layout1.addWidget(self.slider_metric, y_curr,1)
    self.chkActivateMetric = QCheckBox("Activate Metric")
    self.chkActivateMetric.stateChanged.connect(self.activate_metric)
    layout1.addWidget(self.chkActivateMetric, y_curr,2)
    self.chkRemoveTipTilt = QCheckBox("Remove tiptilt")
    self.chkRemoveTipTilt.setChecked(True)
    self.chkRemoveTipTilt.stateChanged.connect(self.do_remove_tiptilt)
    layout1.addWidget(self.chkRemoveTipTilt, y_curr,3)
    y_curr += 1

    ### Modal edges
    self.chkModalEdges = QCheckBox("Modal edges")
    self.chkModalEdges.setChecked(False)
    self.chkModalEdges.stateChanged.connect(self.do_modal_edges)
    layout1.addWidget(self.chkModalEdges, y_curr,3)
    y_curr += 1

    ### Tikhanov correction
    self.chkAOPrecondition = QCheckBox("AO Tikhonov Precondition")
    self.chkAOPrecondition.setChecked(True)
    self.chkAOPrecondition.stateChanged.connect(self.do_ao_precondition)
    layout1.addWidget(self.chkAOPrecondition, y_curr,3)

    self.chkMidi = QCheckBox("MIDI")
    self.chkMidi.setChecked(True)
    self.chkMidi.stateChanged.connect(self.do_midi)
    layout1.addWidget(self.chkMidi, y_curr,2)

    y_curr += 1

    btn1 = QPushButton("Set defocus pos0")
    layout1.addWidget(btn1, y_curr,1)
    btn1.clicked.connect(self.set_defocus_pos0)

    btn1 = QPushButton("Set defocus pos1")
    layout1.addWidget(btn1, y_curr,2)
    btn1.clicked.connect(self.set_defocus_pos1)

    y_curr = 0

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
    if self.offline_only:
        self.mode_btn_ao1.hide() # (Realtime only: the C++ engine)
        btn.hide()

    # Config
    layout1 = QGridLayout(pages[2])
    y_curr = 0
    lbl = QLabel("XML Config: ")
    layout1.addWidget(lbl, y_curr,0)
    self.edit_xml_filename = QLineEdit(self.json_data["params"]["xml_file"])
    layout1.addWidget(self.edit_xml_filename, y_curr,1)
    btn = QPushButton("Select")
    btn.clicked.connect(self.load_config)
    layout1.addWidget(btn, y_curr,2)
    #btn = QPushButton("Edit")
    #layout1.addWidget(btn, y_curr,3)
    #btn.clicked.connect(self.reload_config)
    y_curr += 1
    self.layout_config = layout1
    self.init_config_ui()

    # Settings
    layout1 = QGridLayout(pages[1])
    y_curr = 0
    self.param_tree = ParameterTree()
    self.param_tree.setParameters(self.p, showTop=False)
    layout1.addWidget(self.param_tree, y_curr,0)
    y_curr += 1
    btn = QPushButton("Apply")
    btn.clicked.connect(self.params_apply_clicked)
    layout1.addWidget(btn, y_curr,0)
    y_curr += 1

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

    ### Boxes/Centroids
    layout1 = QGridLayout(self.ops_boxes)

    lbl = QLabel("Box size:")
    layout1.addWidget(lbl,0,0)
    self.widget_boxsize = QDoubleSpinBox()
    layout1.addWidget(self.widget_boxsize,0,1)
    self.widget_boxsize.setDecimals(2)
    self.widget_boxsize.valueChanged.connect(self.boxsize_changed)

    btn = QPushButton("Search box shift")
    btn.clicked.connect(lambda: self.show_zernike_dialog("Shift search boxes", self.engine.shift_search_boxes ) )
    layout1.addWidget(btn, 1,0 )

    btn = QPushButton("Reference shift")
    btn.clicked.connect(lambda: self.show_zernike_dialog("Shift references", self.engine.shift_references ) )
    layout1.addWidget(btn, 2,0 )

    self.chkFollow = QCheckBox("Boxes follow centroids")
    self.chkFollow.stateChanged.connect(lambda:self.set_follow(self.chkFollow.isChecked()))
    layout1.addWidget(self.chkFollow, 1,3 )

    btn = QPushButton("Search box RESET")
    btn.clicked.connect(self.engine.reset_search_boxes )
    layout1.addWidget(btn, 1,1 )

    btn = QPushButton("Reference RESET")
    btn.clicked.connect(self.engine.reset_references )
    layout1.addWidget(btn, 2,1 )

    btn = QPushButton("Shrink+Shift1")
    layout1.addWidget(btn,0,2)
    btn.clicked.connect(lambda: self.engine.offline.offline_auto_shrink1() )

    btn = QPushButton("Shrink+Autoshift")
    layout1.addWidget(btn,0,3)
    btn.clicked.connect(lambda: self.engine.offline.offline_auto_shrink() )

    ### Offline: in the Operation tab (offline_only), else in the Offline tab
    if not self.offline_only:
        QVBoxLayout(pages[3]).addWidget(self.ops_offline)
    layout1 = QGridLayout(self.ops_offline)

    self.btn_off = QPushButton("Load Offline Source")
    self.btn_off.clicked.connect(self.offline_load_image)
    layout1.addWidget(self.btn_off, 1,0)

    self.btn_off_back = QPushButton("Load Offline Background")
    self.btn_off_back.clicked.connect(self.offline_load_background)
    layout1.addWidget(self.btn_off_back, 2,0)

    # Choose the frame to show. Above the arrows, spanning their three columns
    self.frame_slider = FrameSlider()
    self.frame_slider.valueChanged.connect(self.offline_goto)
    layout1.addWidget(self.frame_slider, 2,1, 1,3)

    # OFfline iteration offline_next prev.
    btn = QPushButton("←") # left arrow
    layout1.addWidget(btn,3,1)
    btn.clicked.connect(lambda: self.offline_move(-1,True) )
    self.lbl_frame_curr = QLabel()
    layout1.addWidget(self.lbl_frame_curr,3,2)
    btn = QPushButton("→") # right arrow
    layout1.addWidget(btn,3,3)
    btn.clicked.connect(lambda: self.offline_move (1,True) )

    self.progress_bar = QtWidgets.QProgressBar(self)
    layout1.addWidget(self.progress_bar,3,0)

    self.btn_processing = QPushButton("Auto process all frames")
    layout1.addWidget(self.btn_processing,4,0)
    self.btn_processing.clicked.connect(lambda: self.offline_autoall() )

    btn = QPushButton("Rotate this frame")
    layout1.addWidget(btn,6,2)
    btn.clicked.connect(lambda: self.engine.offline.offline_rotation_fix() )

    btn = QPushButton("Process this frame")
    layout1.addWidget(btn,6,1)
    btn.clicked.connect(lambda: self.engine.offline.offline_manual1() )

    btn = QPushButton("Save this frame")
    layout1.addWidget(btn,4,2)
    btn.clicked.connect(lambda: self.engine.offline.offline_serialize() )

    btn = QPushButton("Dialog all frames")
    layout1.addWidget(btn,5,0)
    btn.clicked.connect(lambda: self.engine.offline.show_dialog() )

    # Dumb: Don't expand and shrink. Best for e.g. calibration images
    btn = QPushButton("Auto simple")
    layout1.addWidget(btn,6,0)
    btn.clicked.connect(lambda: self.engine.offline.offline_auto_dumb() )

    # Offline frame gallery: small thumbnails, wrapping across the panel's width. Virtual (a thumbnail is made when it is shown),
    # since movies can have thousands of frames
    self.frame_model = FrameGalleryModel(self)
    self.frame_gallery = QtWidgets.QListView()
    self.frame_gallery.setModel(self.frame_model)
    self.frame_gallery.setViewMode(QtWidgets.QListView.IconMode)
    self.frame_gallery.setResizeMode(QtWidgets.QListView.Adjust) # Re-wrap when the width changes
    self.frame_gallery.setMovement(QtWidgets.QListView.Static)
    self.frame_gallery.setUniformItemSizes(True) # (All the same size: much faster with thousands)
    self.frame_gallery.setIconSize(QtCore.QSize(FrameGalleryModel.THUMB_SIZE, FrameGalleryModel.THUMB_SIZE))
    self.frame_gallery.setSpacing(3)
    self.frame_gallery.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    self.frame_gallery.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    self.frame_gallery.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    self.frame_gallery.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    layout1.addWidget(self.frame_gallery,7,0, 1,4) # Across all the columns of the panel

    self.chkOfflineAlgorithm = QCheckBox("Use offline algorithm")
    self.chkOfflineAlgorithm.stateChanged.connect(self.offline_algorithm)
    layout1.addWidget(self.chkOfflineAlgorithm,8,0)

    self.param_tree_offline = ParameterTree()
    self.param_tree_offline.setParameters(self.p_offline, showTop=False)

    # Main Widget
    self.widget_main = QWidget()
    layoutCentral = QHBoxLayout()
    layoutCentral.addWidget(self.widget_centrals, stretch=3)
    if not self.offline_only:
        layoutCentral.addWidget(self.widget_displays, stretch=2) # (The DM display)
    layoutCentral.addWidget(self.widget_controls, stretch=1)
    self.widget_main.setLayout(layoutCentral)

    self.setCentralWidget(self.widget_main)

    menu=self.menuBar().addMenu('&File')
    menu.addAction('L&oad config file...', self.load_config_json)
    menu.addSeparator()
    menu.addAction('&Export Centroids + Zernikes', self.export)
    menu.addAction('Export All &Zernikes', self.export_all)
    menu.addAction('Process &directory of AVIs...', self.process_directory)
    menu.addAction('Make summary &plot from a directory of CSVs...', self.summary_plot_directory)
    menu.addAction('Show &Log', self.show_log)
    menu.addAction('Run &Calibration', self.do_calibration)
    menu.addAction('e&Xit', self.close)

    menu=self.menuBar().addMenu('&Help')
    menu.addAction('&About', self.show_about)

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

    self.connect_offline_signals()

    self.setGeometry(2,2,MAIN_WIDTH_WIN,MAIN_HEIGHT_WIN)
    self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
    self.show()
