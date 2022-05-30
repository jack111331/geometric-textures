from email.mime import image
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtWidgets import QPushButton, QTableWidgetItem
from torch import detach
from UI import Ui_MainWindow
import sys
import os
import glob
import inference

typeList = ['bed', 'trash can', 'cabinet', 'bookshelf', 'chair', 'clock', 'dishwasher', 'faucet', 'file cabinet', 'lamp', \
            'pillow', 'sofa', 'table']  #, 'bathtub']

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.typeSelecter.addItems(typeList)
        self.ui.typeSelecter.clicked.connect(self.getTypeSelected)
        currentRowCount = self.ui.tableWidget.rowCount() #necessary even when there are no rows in the table
        
        self.ui.tableWidget.setRowCount(5)
        self.ui.tableWidget.setColumnCount(6)
        
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(64)
        self.ui.tableWidget.verticalHeader().setVisible(False)
        self.ui.tableWidget.horizontalHeader().setDefaultSectionSize(64)
        self.ui.tableWidget.horizontalHeader().setVisible(False)
        self.ui.tableWidget.setShowGrid(False)
        self.ui.tableWidget.clicked.connect(self.getImageSelected)
        self.createButtons(typeList[0])

        self.ui.tableWidget_2.verticalHeader().setDefaultSectionSize(128)
        self.ui.tableWidget_2.verticalHeader().setVisible(False)
        self.ui.tableWidget_2.verticalHeader().setStretchLastSection(True)
        self.ui.tableWidget_2.horizontalHeader().setDefaultSectionSize(150)
        self.ui.tableWidget_2.horizontalHeader().setVisible(False)
        self.ui.tableWidget_2.setShowGrid(False)
        self.initGeometricTextures()
        self.ui.label.setText('Current selected image path: ')

        self.ui.pushButton.clicked.connect(self.run)
        #@t.imagePath = ''
        #t.sizeHint
        #t.setIcon(QtGui.QIcon('./images/bed/1178799.jpeg'))
        

    def initGeometricTextures(self):
        imagePaths = glob.glob(os.path.join("./examples/images/geometric-textures", "*"))
        self.ui.statusbar.showMessage("Loading geometric textures")
        self.ui.tableWidget_2.setRowCount(1)
        self.ui.tableWidget_2.setColumnCount(len(imagePaths))
        for idx, imagePath in enumerate(imagePaths):
            # print(imagePath)
            t = QTableWidgetItem()
            t.setData(QtCore.Qt.ItemDataRole.DecorationRole,
                      QtGui.QPixmap.fromImage(QtGui.QImage(imagePath)).scaled(150, 128))
            t.imagePath = imagePath
            self.ui.tableWidget_2.setItem(0, idx, t)
            self.ui.tableWidget_2.viewport().update()
            self.ui.statusProgressBar.setValue(int(idx * 100 / len(imagePaths)))

            # buttons.append(myCustomQWidget)
        self.ui.statusProgressBar.setValue(100)
        self.ui.statusbar.showMessage("Loaded geometric textures")

    def createButtons(self, typeName):
        cacheTypeName = globals().get('cacheTypeName')
        if cacheTypeName != None and typeName == cacheTypeName:
            return

        buttons = []
        self.ui.statusbar.showMessage("Loading category image")
        imagePaths  = glob.glob(os.path.join("./examples/images/" + typeName, "*"))
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.setColumnCount(0)

        self.ui.tableWidget.setRowCount(int(len(imagePaths) / 6) + 1)
        self.ui.tableWidget.setColumnCount(6)
        currentRowCount = 0
        currentColCount = 0
        for idx, imagePath in enumerate(imagePaths):
            #print(imagePath)
            t = QTableWidgetItem()
            t.setData(QtCore.Qt.ItemDataRole.DecorationRole, QtGui.QPixmap.fromImage(QtGui.QImage(imagePath)).scaled(64,64))
            t.imagePath = imagePath
            self.ui.tableWidget.setItem(currentRowCount, currentColCount, t)
            currentColCount += 1
            if currentColCount == 6:
                # Refresh every row
                self.ui.tableWidget.viewport().update()
                self.ui.statusProgressBar.setValue(int(idx * 100 / len(imagePaths)))
                currentColCount = 0
                currentRowCount += 1
            
            #buttons.append(myCustomQWidget)
        self.ui.statusProgressBar.setValue(100)
        self.ui.statusbar.showMessage("Loaded category image")

        globals()['cacheTypeName'] = typeName
        return buttons

    def run(self, qIndex):
        if self.ui.tableWidget.selectedItems()[0].imagePath == '':
            self.ui.statusbar.showMessage("imagePath doesnot exist")
            return

        if self.ui.tableWidget_2.selectedItems()[0].imagePath == '':
            self.ui.statusbar.showMessage("geometric texture Path doesnot exist")
            return 
        
        imagePath = self.ui.tableWidget.selectedItems()[0].imagePath
        textureName = self.ui.tableWidget_2.selectedItems()[0].imagePath.split('/')[-1][:-4]
        
        output = inference.run(imagePath = imagePath, status_bar=self.ui.statusbar,progress_bar=self.ui.statusProgressBar, textureName=textureName)
        vertices = output.vs
        faces = output.faces

        vertices = vertices.detach().cpu().numpy()
        faces  = faces.long().detach().cpu().numpy()
        
        self.ui.openGLWidget.flushBufferFromMesh(vertices=vertices, faces=faces)
        
    def getImageSelected(self, qIndex):
        self.ui.label.setText('Current selected image path: ' + self.ui.tableWidget.selectedItems()[0].imagePath)

    def getTypeSelected(self, qIndex):
       
       
        self.createButtons(typeList[qIndex.row()])
        self.ui.label.setText('Current selected image path: ')
       

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())