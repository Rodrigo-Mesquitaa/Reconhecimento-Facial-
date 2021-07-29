import cv2
import os
import numpy as np
from PIL import Image


eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000)
fisherface = cv2.face.FisherFaceRecognizer_create(3, 2000)
lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 50)

def getImagemComId():
    caminhos = [os.path.join('deivifaces/treinamento', f) for f in os.listdir('deivifaces/treinamento')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
       imagemFace = Image.open(caminhoImagem).convert('L')
       imagemNP = np.array(imagemFace, 'uint8')
       id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
       ids.append(id)
       faces.append(imagemNP)

    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando...")
eigenface.train(faces, ids)
eigenface.write('classificadorEigenDeivi.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisherDeiviyml')

lbph.train(faces, ids)
lbph.write('classificadorLBPHDeivi.yml')

print("Treinamento realizado")