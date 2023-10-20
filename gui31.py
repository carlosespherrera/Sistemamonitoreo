from __future__ import absolute_import, division, print_function


from typing import Dict, List, Any
from collections import OrderedDict
from functools import partial
from matplotlib.font_manager import json_dump
from numba import jit



from PyQt5.QtWidgets import (
    QWidget,
    QDialog,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QStyle,
    QGridLayout,
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QApplication,
)
from PyQt5.QtCore import (
    QThread,
    pyqtSignal,
    QPoint,
    pyqtSlot,
    QSize,
    Qt,
    QTimer,
    QTime,
    QDate,
    QObject,
    QEvent,
)
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QPalette,
    QResizeEvent,
    QMouseEvent,
    QFont,
    QIcon,
)

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from decouple import config
from dataclasses import dataclass
from typing import Any, Dict, Optional, Text, Literal, Union, TypeVar
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL, MySQLdb
from ttkwidgets.frames import ScrolledFrame
#----------------------------LIBRERIAS AUTENTICACION---------------------------------------------------------------
from flask import Flask
from flask_jwt import JWT, jwt_required, current_identity
from flask import Flask
from re import split
from flask import Blueprint, request, jsonify
#from routes.auth import routes_auth
#from routes.users_github import users_github
from dotenv import load_dotenv

from jwt import encode, decode
from jwt import exceptions
from os import getenv
from datetime import datetime, timedelta
from flask import jsonify

from requests import get

#-----------------------------------------------------------------------------------------------


import tkinter as tk
import cv2
import imutils
import os
import tkinter.messagebox as MessageBox
import mysql.connector as mysql
import mysql.connector as mysqll
import pymysql
import numpy as np
import sys
import shutil
import time
import smtplib
import pyautogui as pg
import time
import webbrowser as web
import numba
import rtsp
import json
import pickle
import nmap









njit= numba.jit

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

phone_no="+523317022428"




#web.open('https://web.whatsapp.com/send?phone='+phone_no)
#correo_emisor= 'carlosespherrera@gmail.com'
#contraseña='ironman42..'


#server= smtplib.SMTP('smtp.gmail.com', 587)
#correo= 'carlosespherrera@gmail.com'
#contraseña='ironman42..'

#server.starttls()
#server.login(correo_emisor,contraseña)

os.environ['OPENCV_VIDEOIO_DEBUG']='1'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF']='O'

dataPath = 'C:/Users/zero_/OneDrive/Documentos/reconocimiento/cio' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Leyendo el modelo
#face_recognizer.read('modeloEigenFace.xml')
#face_recognizer.read('modeloFisherFace.xml')
face_recognizer.read('modeloLBPHFace.xml')


#cap = cv2.VideoCapture('Video.mp4')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


width, height= 480*6, 270*6
w= 1920//2
h= 1080//2
capture_delay= 80

datos= {}
datos['historial']=[]
#------------------------------Api Rest----------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------   
#-----------------------------------------------------CAMARAS---------------------------------------------------------------------------------
def camaras_registro():
    root3= Toplevel()
    root3.iconbitmap("rost.ico")






    root3.title("REGISTRO DE CAMARA")

    miImagen3= PhotoImage(file="how-to-develop-machine-learning-applications-for-business-featured.gif")
    my3= Label(root3, image = miImagen3).place(x=0, y=0, relwidth=1, relheight=1)
    #root3.config(bg="green")



    def pulsar():



#------------------------------------------------DETECCION Y CONTROL DE DATOS-----------------------------------------

        @dataclass
        class KnownFace:
            label: Text
    
            position: Optional[Dict[Literal["x", "y", "w", "h"], int]] = None
            was_found: bool = False
            was_notified: bool = False
            timestamp: Optional[datetime] = None
            was_located: bool= False



        def acquire_frame(cap):
            # Aquí se hace la captura del frame y envío con cv2
            has, frame = cap.read()
            return frame


        def deteccion_facial(frame, known_faces: Dict[Text, KnownFace]):
            # ! TODO: Esta función probablemente debería segmentarse en al menos
            # dos pasos.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aux_frame = gray.copy()
            faces = faceClassif.detectMultiScale(
                gray, 1.3, 10
            )  # Idealmente, face_classif debería ser un parámetro de la función

            k_faces_labels = set()
            for (x, y, w, h) in faces:
                face_frame = aux_frame[y : y + h, x : x + w]
                face_frame = cv2.resize(face_frame, (150, 150), interpolation=cv2.INTER_CUBIC)
                recognized = face_recognizer.predict(
                    face_frame
                )
                if recognized[1] < 70: 
                    cv2.putText(frame,'{}'.format(imagePaths[recognized[0 ]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        

                # Este también debería ser un parámetro
                if not (recognized[1] < 70):
                    continue

                label = str(format(imagePaths[recognized[0]]))
                known = known_faces.get(label)
        

                if not known:
                    continue

                k_faces_labels.add(label)
                known_faces = {
                    **known_faces,
                    label: KnownFace(
                        label=label,
                        position={"x": x, "y": y, "w": w, "h": h},
                        was_found=True,
                        was_located=True,
                        was_notified=known.was_found,
                        timestamp=datetime.utcnow(),
                    ),
            }


    

            unrecognized_faces = set(known_faces.keys()) - k_faces_labels- k_faces_labels

            known_faces = {
                label: KnownFace(
                    label=face.label,
                    position=face.position if label not in unrecognized_faces else None,
                    was_found=face.was_found if label not in unrecognized_faces else False,
                    was_located= face.was_located if (label not in unrecognized_faces) in cams.items() else False,
                    was_notified=face.was_notified
                    if label not in unrecognized_faces
                    else False,
                    timestamp=face.timestamp if label not in unrecognized_faces else None,
                )
                for label, face in known_faces.items()
                }

    

    

            return known_faces



        def draw(frame, label, position):
            # Aquí debería regresar el frame con el cuadro dibujado.
            frame = cv2.rectangle(
                frame,
                (position["x"], position["y"]),
                (position["x"] + position["w"], position["y"] + position["h"]),
                (255, 0, 0),
                5,
            )
            frame = cv2.putText(
                frame,
                f"{label}",
                (position["x"], position["y"] - 25),
                2,
                1.1,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            return frame


        def face_painting(frame, known_faces: Dict[Text, KnownFace]):
            """Esta función pinta el frame con el rostro y el label"""
            new_frame = frame
            for label, face in known_faces.items():
                if face.was_found:
                    new_frame = draw(frame, label, face.position)
            return new_frame

        def show_frame(title, frame2):
            cv2.imshow(title, frame2)

        def ubicacion():
                known_faces = {
                v: KnownFace(label=v)

                for v in imagePaths}

                camara_ubic= {

                cams[5]:{
                "capture_device":cv2.VideoCapture(0),
       

                "locacion": "  En casa"
                },

                cams[11]:{

                "capture_device": cv2.VideoCapture(2),
    

                "locacion": '  en comedor'

                }
                }
                ko = cams.items()
        
                for cam_id, link in cams.items():
              
                        if  link is not None :
                   
            
                            frame= acquire_frame(camara_ubic[cams[cam_id]][cams[cam_id]])
                            if cv2.waitKey(1) == ord("q"):
                                break 
                            known_faces = deteccion_facial(frame, known_faces)
                            frame = face_painting(frame, known_faces)
                            known_faces = {
                            label: send_notification(face)
                            if face.was_found and not face.was_notified 

                            else face
                            for label, face in known_faces.items()
                            }
                    
                            locacion=(camara_ubic[cams[cam_id]]["locacion"])
                            show_frame("frame", frame)

                    
                
                
      
                    
                            print(cam_id)
                    


        def send_notification(face: KnownFace):
            global message
            response={"success" : False, "message" : "" , "data" : {}   }
    
    
            face.was_notified = True
            face.was_located= True
    
    
            cara=face.label
            nombrecara=str(cara)

    

    
            #print(locacion)
    
            registro=(nombrecara + " DETECTADO " + locacion +"  El dia :  " + horario)
            print(registro)

            correo_receptor= 'zero_marvel_64@hotmail.com'
    

            
            message= (nombrecara + " DETECTADO " + locacion )
            
            #------------------------------------correo gmail---------------------------------------

            #server.sendmail(correo_emisor, correo_receptor, message)

            #server.quit()
            #print("correo enviado con exito") 

            #-------------------------------WHATSAPP--------------------------------------------------
            #web.open('https://web.whatsapp.com/send?phone='+phone_no)
            #time.sleep(8)
            #pg.write(message)
            #pg.press('enter')
            #print("mensaje de whatsapp enviado con exito")


            #-----------------------------------HISTORIAL EXCELL---------------------------------------------
            #historial= open("historial.csv", "a")
            #historial.write(message)
            #historial.write("\n")
            #print("Registrado en el historial")

            datos['historial'].append(registro)
            with open('C:/Users/zero_/OneDrive/Documentos/reconocimiento/historial.json', 'w') as f:
                json.dump(datos, f)

            response["success"]= True

    
            return face
            #return json.dumps(registro)






        #---------------------------------------------------------------------------------------------------------------------------------

                    
        #----------------------------------------
        class Slot(QThread):
            signal= pyqtSignal(np.ndarray, int, int , bool)

            def __init__(self, parent: QWidget, index: int, cam_id: int, link: str)-> None:
                QThread.__init__(self, parent)
                self.parent= parent
                self.index= index
                self.cam_id= cam_id
                self.link= link

            def run(self)-> None:
                global locacion

                known_faces = {
                v: KnownFace(label=v)

                for v in imagePaths}

        

                camara_ubic= {

                cams[5]:{
                "capture_device":cv2.VideoCapture(self.link),
       

                "locacion": "  En comedor"
                },

                cams[11]:{

                "capture_device": cv2.VideoCapture(self.link),
    

                "locacion": "En casa"

                }
                }

                cap=cv2.VideoCapture(self.link)

                while cap.isOpened() and cv2.waitKey(1) != ord ("q") :

                    for cam_id, link in cams.items():
                            if  self.link  is not None :
                                #print ("Camara   "+  str(self.cam_id))
                    
                                frame= acquire_frame(camara_ubic[cams[self.cam_id]]["capture_device"])
                        
                    
                                known_faces = deteccion_facial(frame, known_faces)
                                frame = face_painting(frame, known_faces)
                                known_faces = {
                                label: send_notification(face)
                                if face.was_found and not face.was_notified 

                                else face
                                for label, face in known_faces.items()
                                }
                    
                                locacion=(camara_ubic[cams[self.cam_id]]["locacion"])

                        
                                #print (locacion)

                                known_faces = {
                                locacion: send_notification(face)
                                if face.was_found and not face.was_notified 

                                else face
                                for locacion, face in known_faces.items()
                                }

                        

                                frame= cv2.resize(frame, (w,h))
                                self.signal.emit(frame, self.index, self.cam_id, True)
                                cv2.waitKey(capture_delay) & 0xFF

                frame= np.zeros((h,w,3), dtype= np.uint8)
                self.signal.emit(frame, self.index, self.cam_id, False)
                cv2.waitKey(capture_delay) & 0xFF


        def clickable(widget):
            class Filter(QObject):
                clicked= pyqtSignal()
                def eventFilter(self, obj, event):
                    if obj == widget:
                        if event.type() == QEvent.MouseButtonRelease:
                            self.clicked.emit()
                            return True 
                    return False

            filter= Filter(widget)
            widget.installEventFilter(filter)
            return filter.clicked


        #--------------------------------------------------------NUEVA VENTANA DE CAMARA SIN SEÑAL----------------------------------


        class NewWindow(QDialog):
            def __init__(self, parent: QWidget)-> None:
                QDialog.__init__(self, parent)
                self.parent= parent
                self.index: int= 0

                self.label=QLabel()
                self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                self.label.setScaledContents(True)
                self.label.setFont(QFont("Times", 30))
                self. label. setStyleSheet(
                    "color: rgb(255,255,255);"
                    "background-image : url(videonot.jpg);"
                    "qproperty-alignment: AlingCenter;")

                layout= QVBoxLayout()
                layout.setContentsMargins(0,0,0,0)

                layout.addWidget(self.label)
                self.setLayout(layout)
                self.setWindowTitle('Camara {}'.format(self.index))

            def sizeHint(self)-> QSize:
                return QSize(width//3, height//3)

            def resizeEvent(self, event)-> None:
                self.update()

            def KeyPressEvent(self, event) -> None:
                if event.key()== Qt.key_Escape:
                    self.accept()


        #-----------------------------------------------------------------------------------------------------------------------------------------------
            


        #locacion= "en casa"


        class Window(QWidget):

            def __init__(self, cams: Dict[int, str]) -> None:
        

                super(Window, self).__init__()
        
                #--------------iniciar camara con valores vacios------------

                self.cameras: Dict[int, List[Any]]= OrderedDict()
                index: int 
                for index in range(len(cams.keys())):

                    self.cameras[index]= [None, None, False]


                index= 0
                for cam_id, link in cams.items():

                    self.cameras[index]=[cam_id,link, False]
                    index +=1


                layout= QGridLayout()
                layout.setContentsMargins(0,0,0,0)
                layout.setSpacing(2)

                self.labels: List[QLabel]= []
                self.threads: List[Slot]=[]
                for index, value in self.cameras.items():
                    cam_id, link, active= value

                    slot= Slot(self, index, cam_id, link)
                    slot.signal.connect(self.ReadImage)
                    self.threads.append(slot)


                    label= QLabel()
                    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                    label.setScaledContents(True)
                    label.setFont(QFont("Times", 30))
                    label.setStyleSheet(
                        "color: rgb(255,255,255); background-image : url(imagen2.jpg);"
                        "qproperty-alignment: AlingCenter;")

                    clickable(label).connect(partial(self.showCam, index))
                    self.labels.append(label)

                    if index ==0:
                        layout.addWidget(label, 0,0)
                    elif index ==1:
                        layout.addWidget(label, 0,1)
                    elif index ==2:
                        layout.addWidget(label, 0,2)
                    elif index ==3:
                        layout.addWidget(label, 0,3)
                    elif index ==4:
                        layout.addWidget(label, 1,0)
                    elif index ==5:
                        layout.addWidget(label, 1,1)
                    elif index ==6:
                        layout.addWidget(label, 1,2)
                    elif index ==7:
                        layout.addWidget(label, 1,3)
                    elif index ==8:
                        layout.addWidget(label, 2,0)
                    elif index ==9:
                        layout.addWidget(label, 2,1)
                    elif index ==10:
                        layout.addWidget(label, 2,2)
                    elif index ==11:
                        layout.addWidget(label, 2,3)
                    elif index ==12:
                        layout.addWidget(label, 3,0)
                    elif index ==13:
                        layout.addWidget(label, 3,1)
                    elif index ==14:
                        layout.addWidget(label, 3,2)
                    elif index ==15:
                        layout.addWidget(label, 3,3)
                    elif index ==16:
                        layout.addWidget(label, 4,0)
                    elif index ==17:
                        layout.addWidget(label, 4,1)
                    elif index ==18:
                        layout.addWidget(label, 4,2)
                    elif index ==19:
                        layout.addWidget(label, 4,3)
                    else:
                        raise ValueError("n Camera != rows/cols ")

        

                timer= QTimer(self)
                timer.timeout.connect(self.showTime)
                timer.start(1000)
                self.showTime()


                timer_th= QTimer(self)
                timer_th.timeout.connect(self.refresh)
                timer_th.start(60000*60*3)

                self.setLayout(layout)
                self.setWindowTitle('FACIAL RECOGNITION PROJECT')
                self.setWindowIcon(QIcon('voblakye.jpg'))

                self.newWindow= NewWindow(self)

                self.refresh()

            def sizeHint(self)-> QSize:
                return QSize(width, height)

            def resizeEvent (self, event)-> None:
                self.update()

            def KeyPressEvent(self, event)-> None:
                if event.key()== Qt.key_Escape:
                    self.close()
            def closeEvent(self, event): pass

            def showCam(self, index: Any)-> None:
                self.newWindow.index= index
        

        
                if not self.cameras[index][2]:
                        text_= "Camara {}\n!Sin señal!".format(self.cameras[index][0])
                        self.newWindow.label.setText(text_)
        
                        print("SIN SEÑAL!")
        

        
                elif (self.cameras[index][2]) is True:         
       
                    print("SEÑAL ")


                self.newWindow.setWindowTitle('Camara {}'.format(self.cameras[index][0]))
                self.newWindow.show()

            def showTime(self)-> None:
                global horario

                time= QTime.currentTime()
                textTime= time.toString('hh:mm:ss')
        

                date= QDate.currentDate()
                textDate= date.toString('dddd d, MMMM yyyy')
                horario=( str(textDate) + " " + str(textTime))
                #print(horario)

                text= "{}\n{}".format(textTime, textDate)
                #print(str(text))

                for index, value in self.cameras.items():
                    cam_id, link, active= value 
                    if not active:
                        text_="Camara {}\n".format(cam_id)+ text
                        self.labels[index].setText(text_)

        
            



            @pyqtSlot(np.ndarray, int, int , bool)
            def ReadImage(self, im: np.ndarray, index:int, cam_id:int,active:bool)->None:
                self.cameras[index][2]= active
                cam_id,link, active= self.cameras[index]

                im= QImage(im.data, im.shape[1], im.shape[0], QImage.Format_RGB888).rgbSwapped()
        
                self.labels[index].setPixmap(QPixmap.fromImage(im))

                if index == self.newWindow.index:
                    self.newWindow.label.setPixmap(QPixmap.fromImage(im))


            def refresh(self)-> None:
                for slot in self.threads:

                    slot.start()






                


        if __name__== '__main__':
            dataPath = 'C:/Users/zero_/OneDrive/Documentos/reconocimiento/cio' #Cambia a la ruta donde hayas almacenado Data
            imagePaths = os.listdir(dataPath)
    

            import sys
#-----------------------------DIRECCION DE CAMARA-------------------------------------------
            cams: Dict[int, Any]= OrderedDict()

    
    

            cams[1] = None
            cams[2] = None
            cams[3] = None
            cams[4] = None
            cams[5] = 0
            cams[6] = None
            cams[7] = None
            cams[8] = None
            cams[9] = None
            cams[10] = None
            cams[11] = 2
            cams[12] = None
            cams[13] = None
            cams[14] = None
            cams[15] = None
            cams[16] = None
            cams[17] = None
            cams[18] = None
            cams[19] = None
            cams[20] = None

    
            camara_ubic= {

            cams[5]:{
                "capture_device":cv2.VideoCapture(0),
       

                "locacion": "  En casa"
                },

            cams[11]:{

            "capture_device": cv2.VideoCapture(2),
    

            "locacion": '  en comedor'

                }
                }

    


            print('imagePaths=',imagePaths)


    
    


            ko= cams.items()

            app= QApplication(sys.argv)
            win= Window(cams= cams)
    
            win.show()
   
    
    


            sys.exit(app.exec_())
    
    

    


    def registro_de_camaras():
        global e_locacion
        root4= Toplevel()
        root4.iconbitmap("rost.ico")

        miImagen4= PhotoImage(file="how-to-develop-machine-learning-applications-for-business-featured.gif")
        my4= Label(root4, image = miImagen4).place(x=0, y=0, relwidth=1, relheight=1)

        def scann_ip():
            print("ESCANEANDO...")
            




            #ip=input("[+] IP Objetivo ==> ")
            ip=e_ip.get()

            nm = nmap.PortScanner()
            puertos_abiertos="-p "
            results = nm.scan(hosts=ip,arguments="-sT -n -Pn -T4")
            count=0
            print("aqui va")
            print (results)
            print("\nHost : %s" % ip)
            print("State : %s" % nm[ip].state())

            for proto in nm[ip].all_protocols():
                print("Protocol : %s" % proto)
                print()
                lport = nm[ip][proto].keys()
                sorted(lport)
                for port in lport:
                    print ("port : %s\tstate : %s" % (port, nm[ip][proto][port]["state"]))
                    if count==0:
                        puertos_abiertos=puertos_abiertos+str(port)
                        count=1
                else:
                    puertos_abiertos=puertos_abiertos+","+str(port)

            print("\nPuertos abiertos: "+ puertos_abiertos +" "+str(ip))
            barp3['value']+=(300)
            print(ip)
            e_ipv4.delete(0, 'end')


        def listar_cams():
            con= MySQLdb.connect(host="localhost", user="root", password="12345678", database="camaras",cursorclass=MySQLdb.cursors.DictCursor)

    
            cursor= con.cursor()
    
            cursor.execute('SELECT * FROM register')
            data = cursor.fetchall()
            register=[]
            content={}
            for result in data:
                content={'Id': result['Id'], 'Ip':result['Ip'],'Protocolo':result['Protocolo'],'Usuario':result['Usuario'], 'Contraseña':result['Contraseña'], 'Canal':result['Canal']}
                register.append(content)
                content={}
            con.close()
            print (register)



        root4.title("ESCANEAR IP")
        root4.config(bg="orange")

        root4.geometry("550x600+1300+10")

        root4.resizable(width=False,height=False)


        
        e_ip= StringVar()
        e_ipv4=Entry(root4,textvariable=e_ip)
        e_ipv4.place(x=170, y=300)

        scannip= Button(root4, text="Escanear", command= scann_ip)
        scannip.place(x=170, y=390)

        inip= Label(root4, text='Ingrese IP', font=('bold', 10))
        inip.place(x=20, y=300)

        barp3=ttk.Progressbar(root4,style="TProgressbar", orient=HORIZONTAL, length=300, mode='determinate')
        barp3.pack(padx=10, pady=10, expand= True)
        barp3.place(x=170, y=435)

        Editcams= Button(root4, text="Listar Camaras", command=listar_cams)
        Editcams.place(x=170, y=340)

        root4.mainloop()



    def inster_cams_registro():
        #response={"success" : False, "message" : "" , "data" : {}   }
        #print(request.form)
       
        #id =e_id.get()
        #nombre =e_name.get()
        #telefono =e_num.get()

        #if request.method == 'POST':
     
        #Ip = request.form.get('Ip')
        #Protocolo = request.form.get('Protocolo')
        #Usuario = request.form.get('Usuario')
        #Contraseña = request.form.get('Contraseña')
        #Canal = request.form.get('Canal')

        Ip = eip.get()
        Protocolo = e_protocolo.get()
        Usuario = e_usuario.get()
        Contraseña = e_contraseña.get()
        Canal = e_canal.get()
        #print(id , nombre, telefono)

        if(Protocolo=="" or Usuario=="" or Contraseña=="" or Canal==""):
            #response["message"]= "Todos los datos son requeridos"
           MessageBox.showinfo("Inserta estatus", "Todos los datos son requeridos")

    
        else:
            #print("jj")
            con= pymysql.connect(host="localhost", user="root", password="12345678", database="camaras")
            cursor= con.cursor()

            cursor.execute("INSERT INTO  register (Ip, Protocolo, Usuario, Contraseña, Canal) VALUES('"+ str(Ip) +"','"+ str(Protocolo) +"','"+ Usuario +"', '"+ str(Contraseña) +"', '"+ str(Canal) +"' )")
            cursor.execute("commit")
            #print(Ip, Usuario, Protocolo)

            eip.delete(0, 'end')
            e_protocolo.delete(0, 'end')
            e_usuario.delete(0, 'end')
            e_contraseña.delete(0, 'end')
            e_canal.delete(0, 'end')
            #show()

            #response["success"]= True
            #response["message"]= "Registro camara exitosa"
            MessageBox.showinfo("Inserta estatus", "Registro exitoso")
            #barp.stop()
            #bar.stop()


            con.close();

    def delete_cam_registro():
        #response={"success" : False, "message" : "" , "data" : {}   }
        #if request.method == 'DELETE':
            #Ip = eip.get()
        Ip=eip.get()
        #dataPath = 'C:/Users/zero_/OneDrive/Documentos/reconocimiento/cio'#Cambia a la ruta donde hayas almacenado Data


        #eliminar = e_name.get()
        #personPath = dataPath + '/' + eliminar
    

        if(Ip=="" ):
            #response["message"]= "Ingrese el dato correcto"

            MessageBox.showinfo("Borra estatus", "Ingrese el dato correcto")

        else:
            con= pymysql.connect(host="localhost", user="root", password="12345678", database="camaras")

    
            cursor= con.cursor()
            #print (nombre)

            cursor.execute("DELETE FROM  register WHERE Ip='"+str(Ip)+"'")
           
        

            cursor.execute("commit")
            #response["success"]= True
            #response["message"]= "Dato Eliminado"
        

            eip.delete(0, 'end')
            #e_nombre.delete(0, 'end')
            #e_num.delete(0, 'end')

            #show()
            MessageBox.showinfo("Borrar estatus", "Borrado Exitoso")

        #folder_path = '/path/to/folder'
        #for eliminar in os.listdir(dataPath): 
            #file_object_path = os.path.join( personPath) 

        #if os.path.isfile(file_object_path): 
            #os.unlink(file_object_path) 
        #else: shutil.rmtree(file_object_path)

     
        #return response
        con.close();

    


    root3.geometry("500x600+100+10")

    root3.resizable(width=False,height=False)
    

    Re_camaras= Button(root3, text="ESCANEAR", command= registro_de_camaras)
    Re_camaras.place(x=40, y=500)


    Sis_cams= Button(root3, text="SISTEMA DE CAMARAS", command= pulsar)
    Sis_cams.place(x=40, y=550)

    e_camname= StringVar()
    e_camnombr=Entry(root3,textvariable=e_camname)
    e_camnombr.place(x=170, y=80)


    e_access= Entry(root3)
    e_access.place(x=170, y=140)

    e_locacion= StringVar()
    e_localizacion=Entry(root3,textvariable=e_locacion)
    e_localizacion.place(x=170, y=110)

    camaranombre= Label(root3, text='Nombre de camara', font=('bold', 10))
    camaranombre.place(x=20, y=80)

    acceso= Label(root3, text='Ruta de acceso', font=('bold', 10))
    acceso.place(x=20, y=140)

    ubicacioncam= Label(root3, text='Ingrese ubicacion', font=('bold', 10))
    ubicacioncam.place(x=20, y=110)

    eip= Entry(root3)
    eip.place(x=170, y=170)

    e_protocolo= Entry(root3)
    e_protocolo.place(x=170, y=200)

    e_usuario= Entry(root3)
    e_usuario.place(x=170, y=230)

    e_contraseña= Entry(root3)
    e_contraseña.place(x=170, y=260)

    e_canal= Entry(root3)
    e_canal.place(x=170, y=290)

    Ipin= Label(root3, text='Registre IP', font=('bold', 10))
    Ipin.place(x=20, y=170)

    Protocoloin= Label(root3, text='Protocolo', font=('bold', 10))
    Protocoloin.place(x=20, y=200)

    Usuarioin= Label(root3, text='Usuario', font=('bold', 10))
    Usuarioin.place(x=20, y=230)

    Contraseñain= Label(root3, text='Contraseña', font=('bold', 10))
    Contraseñain.place(x=20, y=260)

    Canalin= Label(root3, text='Ingrese Canal', font=('bold', 10))
    Canalin.place(x=20, y=290)

    Registracams= Button(root3, text="Registrar camara", command= inster_cams_registro)
    Registracams.place(x=40, y=330)

    Deletecams= Button(root3, text="Borrar camara", command= delete_cam_registro)
    Deletecams.place(x=180, y=330)

    root3.mainloop()

    


    




    

#------------------------------------------------------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------PERSONAS-------------------------------------------------------------------
def tabla():
    root1= Toplevel()

    root1.iconbitmap("rost.ico")



    root1.title("TABLA")
    root1.config(bg="white")
    #root1.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    root1.geometry("550x600+100+10")

    root1.resizable(width=False,height=False)


    miImagen= PhotoImage(file="inteligencia-artificial-manos.gif")
    my= Label(root1, image = miImagen).place(x=0, y=0, relwidth=1, relheight=1)

    def insert():
        Nombre =""
        Edad =""
        Apellido= ""

        if(Nombre=="" or Apellido=="" or Edad==""):
           MessageBox.showinfo("Inserta estatus", "Todos los datos son requeridos")

    
        else:
            con= pymysql.connect(host="localhost", user="root", password="12345678", database="personas")
            cursor= con.cursor()

            cursor.execute("INSERT INTO  rostros_agregados (Nombre, Apellido, Edad) VALUES('"+ str(Nombre) +"','"+ str(Apellido) +"','"+ Edad +"' )")
            cursor.execute("commit")

            #e_lastname.delete(0, 'end')
            #e_nombre.delete(0, 'end')
            #e_edad.delete(0, 'end')
            show()


            MessageBox.showinfo("Inserta estatus", "Registro exitoso")
            #barp.stop()
            #bar.stop()


            con.close();




    def delete(Id,Nombre, Apellido):
        print("va ", Id)


        dataPath = 'C:/Users/zero_/OneDrive/Documentos/reconocimiento/cio'#Cambia a la ruta donde hayas almacenado Data

 

        eliminar1 = Nombre 
        eliminar2= Apellido
        eliminar=eliminar1 +" "+ eliminar2
        print (eliminar)
        personPath = dataPath + '/' + eliminar
    

        if(Id=="" ):

            MessageBox.showinfo("Borra estatus", "Ingrese el dato correcto")

        else:
            con= pymysql.connect(host="localhost", user="root", password="12345678", database="personas")

    
            cursor= con.cursor()

            cursor.execute("DELETE FROM  rostros_agregados WHERE Id='"+str(Id)+"'")
        

            cursor.execute("commit")
        

            #e_lastname.delete(0, 'end')
            #e_nombre.delete(0, 'end')
            

            show()
            MessageBox.showinfo("Borrar estatus", "Borrado Exitoso")

        folder_path = '/path/to/folder'
        for eliminar in os.listdir(dataPath): 
            file_object_path = os.path.join( personPath) 

        if os.path.isfile(file_object_path): 
            os.unlink(file_object_path) 
        else: shutil.rmtree(file_object_path)

     

        



        con.close();

    def update():
        Nombre =""
        Apellido =""
        Edad= ""
        Id= " "

        if(Id=="" or Nombre=="" or Apellido=="" or Edad==""):
            MessageBox.showinfo("Actualiza estatus", "Todos los datos son requeridos")

    
        else:
            con= pymysql.connect(host="localhost", user="root", password="12345678", database="faccion")
            cursor= con.cursor()

            cursor.execute("UPDATE  tablamas SET Apellido='"+ Nombre +"', Apellido='"+Apellido+"', , Edad='"+Edad+"'where id= '"+id+"'")
            cursor.execute("commit")

            #e_apellido.delete(0, 'end')
            #e_nombre.delete(0, 'end')
            #e_edad.delete(0, 'end')
            show()


            MessageBox.showinfo("Actializa", "Actualizado Exitoso")


            con.close();


    def get():
        Id=""

        if(Id=="" ):

            MessageBox.showinfo("Buscar estatus", "ID esta listo para borrarse")

        else:
            con= pymysql.connect(host="localhost", user="root", password="12345678", database="personas")

    
            cursor= con.cursor()
            cursor.execute("SELECT* from  rostros_agregados WHERE Id='"+str(Id)+"'")

            rows= cursor.fetchall()
        
            #for row in rows:
                #e_nombre.insert(0, row[1])
                #e_num.insert(0, row[2])

        

        

            con.close();
    
    




    def show():
        con= pymysql.connect(host="localhost", user="root", password="12345678", database="personas")

    
        cursor= con.cursor()
        cursor.execute("SELECT   Id,Nombre, Apellido from  rostros_agregados ")
        #cursor.execute("SELECT* from  tablamas ")
        rows= cursor.fetchall()
        
        #list.delete(0, list.size())
        row=1
        sd=sorted(rows)
        #print(sd)
        
       

        frame = ScrolledFrame(root1, compound=tk.RIGHT, canvasheight=520)
        frame.place(x=10, y= 50)

        gh=Label(root1,text="NOMBRE", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        gj=Label(root1,text="APELLIDO", anchor="w").grid(row=0, column=1, sticky="ew", padx=10, pady=10)

        for ( Id,Nombre, Apellido ) in rows:
            #nr_id=tk.Label(root1 ,text=str(Id), anchor="w")
            nr_label = tk.Label(frame.interior ,text=str(Nombre), anchor="w",)
            name_label = tk.Label(frame.interior, text=str(Apellido), anchor="w")
            #inserData= str(row[0]) + " "+ str(row[1])
            #list.insert(tk.END, "Elemento {}".format(row))
            borrar= Button(frame.interior, text="Borrar", font=("italic", 10),bg="blue",command=lambda Nombre=Nombre, Apellido=Apellido: delete(Id, Nombre, Apellido))
            
            #nr_id.grid(row=row, column=0, sticky="ew")
            nr_label.grid(row=row, column=0, sticky="ew", padx=10, pady=10)
            name_label.grid(row=row, column=1, sticky="ew", padx=10, pady=10)

            
            borrar.grid(row=row, column=7, sticky="ew",padx=10, pady=10)
            

            row+=1
           




            
            #list.insert(list.size()+1, inserData)

        
        con.close()

    
    

    #list=Listbox(root1, bg= "#F5F5DC" ,yscrollcommand = scrollbar.set)
    
    #scrollbar.config( command = list.yview )
    #list.place(x= 20, y= 70)
    show()

    
    boton2= Button(root1, text="REGISTRAR USUARIO", command= registro_user)
    boton2.place(x=370, y=10)

    root1.mainloop()



    #borrar= Button(root1, text="Borrar", font=("italic", 10),bg="blue", command= delete)
    #borrar.place(x=460, y=500)


     





#----------------------------------------------CREAR->>REGISTRAR USUARIO---------------------------------
def registro_user():

    root2= Toplevel()
    root2.iconbitmap("rost.ico")






    root2.title("Registro de usuario")
    root2.config(bg="white")

    root2.geometry("550x600+1300+10")

    root2.resizable(width=False,height=False)


    miImagen2= PhotoImage(file="inteligencia-artificial-manos.gif")
    my2= Label(root2, image = miImagen2).place(x=0, y=0, relwidth=1, relheight=1)

    def insert():
        Nombre =e_name.get()
        Edad =e_edad.get()
        Apellido= e_apellido.get()

        if(Nombre=="" or Apellido=="" or Edad==""):
           MessageBox.showinfo("Inserta estatus", "Todos los datos son requeridos")

    
        else:
            con= pymysql.connect(host="localhost", user="root", password="12345678", database="personas")
            cursor= con.cursor()

            cursor.execute("INSERT INTO  rostros_agregados (Nombre, Apellido, Edad) VALUES('"+ str(Nombre) +"','"+ str(Apellido) +"','"+ Edad +"' )")
            cursor.execute("commit")

            e_lastname.delete(0, 'end')
            e_nombre.delete(0, 'end')
            e_edad.delete(0, 'end')
            show()


            MessageBox.showinfo("Inserta estatus", "Registro exitoso")
            barp.stop()
            bar.stop()


            con.close();


    def delete():


        dataPath = 'C:/Users/zero_/OneDrive/Documentos/reconocimiento/cio'#Cambia a la ruta donde hayas almacenado Data

 

        eliminar1 = e_name.get() + " "
        eliminar2= e_apellido.get()
        eliminar=eliminar1 + eliminar2
        print (eliminar)
        personPath = dataPath + '/' + eliminar
    

        if(e_name.get()=="" ):

            MessageBox.showinfo("Borra estatus", "Ingrese el dato correcto")

        else:
            con= pymysql.connect(host="localhost", user="root", password="12345678", database="personas")

    
            cursor= con.cursor()

            cursor.execute("DELETE FROM  rostros_agregados WHERE Nombre='"+e_name.get()+"'")
        

            cursor.execute("commit")
        

            e_lastname.delete(0, 'end')
            e_nombre.delete(0, 'end')
            

            show()
            MessageBox.showinfo("Borrar estatus", "Borrado Exitoso")

        folder_path = '/path/to/folder'
        for eliminar in os.listdir(dataPath): 
            file_object_path = os.path.join( personPath) 

        if os.path.isfile(file_object_path): 
            os.unlink(file_object_path) 
        else: shutil.rmtree(file_object_path)

     

        



        con.close();

    def update():
        Nombre =e_name.get()
        Apellido =e_apellido.get()
        Edad= e_edad.get()
        Id= " "

        if(Id=="" or Nombre=="" or Apellido=="" or Edad==""):
            MessageBox.showinfo("Actualiza estatus", "Todos los datos son requeridos")

    
        else:
            con= pymysql.connect(host="localhost", user="root", password="12345678", database="faccion")
            cursor= con.cursor()

            cursor.execute("UPDATE  tablamas SET Apellido='"+ Nombre +"', Apellido='"+Apellido+"', , Edad='"+Edad+"'where id= '"+id+"'")
            cursor.execute("commit")

            e_apellido.delete(0, 'end')
            e_nombre.delete(0, 'end')
            e_edad.delete(0, 'end')
            show()


            MessageBox.showinfo("Actializa", "Actualizado Exitoso")


            con.close();


    def get():
        Id=" "

        if(Id=="" ):
            #response["message"]= "Todos los datos son requeridos"

            MessageBox.showinfo("Buscar estatus", "ID esta listo para borrarse")

        else:
            con= MySQLdb.connect(host="localhost", user="root", password="12345678", database="faccion",cursorclass=MySQLdb.cursors.DictCursor)

    
            cursor= con.cursor()
            print(id)
            cursor.execute("SELECT Id, nombre, Apellido, Edad from  rostros_agregados where Id= '"+str(Id)+"'")

            rows= cursor.fetchall()
        
            for row in rows:
                e_nombre.insert(0, row[1])
                e_edad.insert(0, row[2])
                e_lastname.insert(0, row[3])

        

        

            con.close();
    
    def capturar():
        personName1 = e_name.get() + " "
        personName2= e_apellido.get()
        personName=personName1 + personName2
        print (personName)
        
        dataPath = 'C:/Users/zero_/OneDrive/Documentos/reconocimiento/cio'#Cambia a la ruta donde hayas almacenado Data
        personPath = dataPath + '/' + personName
        if not os.path.exists(personPath):
            print('Carpeta creada: ',personPath)
            os.makedirs(personPath)
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        #cap = cv2.VideoCapture('Video.mp4')
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        count = 0
        while True:
    
            ret, frame = cap.read()
            if ret == False: break
            rame =  imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()
            faces = faceClassif.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w,y+h),(89,44,211),2)
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
                #bs64
                count = count + 1
            cv2.imshow('frame',frame)

    
            k =  cv2.waitKey(1)
            if k == 27 or count >= 300:
                barp['value']+=(count)
                break
        cv2.destroyAllWindows()

   



    def validar():
            dataPath = 'C:/Users/zero_/OneDrive/Documentos/reconocimiento/cio'
            peopleList = os.listdir(dataPath)
            print('Lista de personas: ', peopleList)

            labels = []
            facesData = []
            label = 0

            for nameDir in peopleList:
                personPath = dataPath + '/' + nameDir
                print('Leyendo las imágenes')

                for fileName in os.listdir(personPath):
                    print('Rostros: ', nameDir + '/' + fileName)
                    labels.append(label)
                    facesData.append(cv2.imread(personPath+'/'+fileName,0))
                    image = cv2.imread(personPath+'/'+fileName,0)
                    cv2.imshow('image',image)
                    cv2.waitKey(10)
                label = label + 1
            print('labels= ',labels)
            print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
            print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

            #ace_recognizer = cv2.face.EigenFaceRecognizer_create()
            #face_recognizer = cv2.FisherFaceRecognizer_create()
            face_recognizer =cv2.face.LBPHFaceRecognizer_create()
            print("Entrenando...")
            face_recognizer.train(facesData, np.array(labels))


            #face_recognizer.write('modeloEigenFace.xml')
            #face_recognizer.write('modeloFisherFace.xml')
            face_recognizer.write('modeloLBPHFace.xml')
            print("Modelo almacenado...")
            bar['value']+=(300)
        
    cv2.destroyAllWindows()






    def show():
        con= pymysql.connect(host="localhost", user="root", password="12345678", database="faccion")

    
        cursor= con.cursor()
        cursor.execute("SELECT* from  tablamas ")
        rows= cursor.fetchall()
        #list.delete(0, list.size())

        #for row in rows:
            #inserData= str(row[0])+'    '+ row[1]
            #list.insert(list.size()+1, inserData)

        con.close()


    

    textoregis= Label(root2, text='REGISTRAR USUARIO', font=('bold', 10))
    textoregis.place(x=80, y=20)


    nombre= Label(root2, text='Ingrese nombre', font=('bold', 10))
    nombre.place(x=20, y=80)

    apellido= Label(root2, text='Ingrese apellido', font=('bold', 10))
    apellido.place(x=20, y=110)

    edad= Label(root2, text='Ingrese edad', font=('bold', 10))
    edad.place(x=20, y=140)

    


    e_name= StringVar()
    e_nombre=Entry(root2,textvariable=e_name)
    e_nombre.place(x=170, y=80)


    e_edad= Entry(root2)
    e_edad.place(x=170, y=140)

    e_apellido= StringVar()
    e_lastname=Entry(root2,textvariable=e_apellido)
    e_lastname.place(x=170, y=110)




    insert= Button(root2, text="Registrar", font=("italic", 10),bg="blue", command= insert)
    insert.place(x=30, y=190)

    borrar= Button(root2, text="Borrar", font=("italic", 10),bg="blue", command= delete)
    borrar.place(x=100, y=190)

    actual= Button(root2, text="Actualizar", font=("italic", 10),bg="blue", command= update)
    actual.place(x=160, y=190)


    get= Button(root2, text="Obtener", font=("italic", 10),bg="blue", command= get)
    get.place(x=235, y=190)

    #list=Listbox(root2, bg= "#F5F5DC" )
    #list.place(x= 400, y= 20)
    #show()


    boton3= Button(root2, text="Guardar foto", command= validar)
    boton3.place(x=40, y=370)


    boton4= Button(root2, text="Capturar foto", command= capturar)
    boton4.place(x=40, y=250)



    barp=ttk.Progressbar(root2,style="TProgressbar", orient=HORIZONTAL, length=300, mode='determinate')
    barp.pack(padx=10, pady=10, expand= True)
    barp.place(x=40, y=300)

    estilo = ttk.Style()
    estilo.configure("TProgressbar", bg="darkblue")

    bar=ttk.Progressbar(root2, style="TProgressbar",orient=HORIZONTAL, length=300, mode='determinate')
    bar.pack(padx=10, pady=10, expand= True)
    bar.place(x=40, y=420)

    root2.mainloop()
#-----------------------------------------------------------------------------------------------------------

#--------------------------------INTERFAZ DE VISUALIZACION Y REGISTRO DE CAMARAS------------------------------------

    

#---------------------------------------------------------------------------------------------------------------


#----------------------INTERFAZ PRINCIPAL----------------------------------------------------


root= Tk()
root.iconbitmap("rost.ico")



root.title("RECONOCIMIENTO FACIAL")
root.config(bg="yellow")

root.geometry("550x600+700+10")

root.resizable(width=False,height=False)


miImagen= PhotoImage(file="articulo.gif")
my= Label(root, image = miImagen).place(x=0, y=0, relwidth=1, relheight=1)
    
    
boton= Button(root, text="Camaras", command= camaras_registro)
boton.place(x=40, y=320)


boton1= Button(root, text="Personas", command= tabla)
boton1.place(x=40, y=150)



root.mainloop()


    
    

    

 
    