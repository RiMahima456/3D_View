import sys
import cv2
import numpy as np
import mediapipe as mp
import trimesh
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

class Page(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Dino + Hand Detection Feed")
        self.setGeometry(100, 100, 1200, 800)

        # 3D Display
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter.interactor)

        # Laod 3D Model
        dino = trimesh.load("12978_tulip_flower_l3.obj") #tried on glb and obj
        if isinstance(dino, trimesh.Scene):
            dino = dino.dump(concatenate=True)
        faces = np.hstack([np.full((len(dino.faces), 1), 3), dino.faces])
        pvdino = pv.PolyData(dino.vertices, faces)

        self.plotter.add_mesh(pvdino, color=True, show_edges=False)
        self.plotter.set_background("black")
        self.plotter.enable_3_lights()
        self.plotter.add_axes()
        #uncomment if u need grid
        # self.plotter.show_grid(color="white")  

        # overlay just so u know what is happening
        self.feed_label = QLabel(self)
        self.feed_label.setFixedSize(400, 300)
        self.feed_label.setStyleSheet("border: 2px solid white;")
        self.feed_label.move(self.width() - 420, self.height() - 340)
        self.feed_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        # hand ges processing
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8,
                                         min_tracking_confidence=0.8,
                                         max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

        
        self.cap = cv2.VideoCapture(0)

        # Update the timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.updater)
        self.timer.start(30)

        # have used pinch mode for rotation and distance mode for zoom in and zoom out
        self.lastcen = None
        self.lasdis=None

    def palm_status(hand_landmarks, handedness="Right"):
        fingers = []
        if handedness == "Right":
            fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
        else:
            fingers.append(hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x)

        """Camera Dependent. For both mine and my collaborato's webcam this fitted. If the origin for your Webcam lies on
        other corner flip the comparison operators"""

        fingers.append(hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y)
        fingers.append(hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y)
        fingers.append(hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y)
        fingers.append(hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y)

        return all(fingers)



    def updater(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape
                flag=Page.palm_status(hand_landmarks)
                if flag:
                    continue

                #the pinch between the index finger tip and thumb. have uploaded hand.png in the folder for reff.
                x1, y1,z1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h), hand_landmarks.landmark[4].z
                x2, y2,z2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h), hand_landmarks.landmark[4].z

                
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                print(distance)

                if distance < 40:                  #This is the pinch code. manually calculated.kept a little buffer for loosely pinched finger. Reduce it as you want
                    xa, ya, za = (x1 + x2) // 2, (y1 + y2) // 2, (z1+z2)//2
                    cv2.circle(frame, (xa, ya), 10, (0, 0, 255), -1)

                    if self.lastcen is not None:
                        dx, dy, dz = xa - self.lastcen[0], ya - self.lastcen[1], za -self.lastcen[2]

                        if abs(dx) > 2 or abs(dy) > 2 or abs(dz)>2:
                            self.plotter.camera.azimuth -= dx * 0.8
                            self.plotter.camera.elevation -= dy * 0.8  
                            self.plotter.render()

                    self.lastcen = (xa, ya,za)
                else:
                    self.lastcen = None
                    if self.lasdis is None:
                        self.lasdis=distance
                    else:
                        factor=distance/self.lasdis
                        if factor>1.05: 
                            self.plotter.camera.zoom(factor*1.25)
                            self.plotter.render()
                        elif factor<0.95:
                            self.plotter.camera.zoom(factor*0.75)
                            self.plotter.render()
                        self.lasdis=distance

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.feed_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.feed_label.width(), self.feed_label.height(), Qt.KeepAspectRatio
        ))

    def resizeEvent(self, event):
        self.feed_label.move(self.width() - 420, self.height() - 340)

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Page()
    win.show()
    sys.exit(app.exec())
