import face_recognition as fr

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if(len(rostos) > 0):
        return True, rostos
    
    return False, []

def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []

    roger1 = reconhece_face("./img/roger1.jpg")
    if(roger1[0]):
        rostos_conhecidos.append(roger1[1][0])
        nomes_dos_rostos.append("Roger")

    romulo1 = reconhece_face("./img/romulo1.jpg")
    if(romulo1[0]):
        rostos_conhecidos.append(romulo1[1][0])
        nomes_dos_rostos.append("Romulo")
    
    return rostos_conhecidos, nomes_dos_rostos