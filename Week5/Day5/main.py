import cv2
import numpy as np
import os

def load_image(image_path):
    """Charge une image et vérifie son existence."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Erreur : le fichier '{image_path}' est introuvable.")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Erreur : format d'image non supporté ou fichier corrompu.")
    return image

def detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    """Détecte tous les visages dans l'image avec Haar Cascade et retourne l'array des bboxes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )
    if len(faces) == 0:
        print("Aucun visage détecté.")
        return None
    print(f"{len(faces)} visage(s) détecté(s).")
    return faces

def extract_face(image, bbox, margin=0):
    """Extrait le visage à partir des coordonnées (x,y,w,h).
       margin : pour ajouter un padding autour du visage (en pixels)."""
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]

    # appliquer margin en restant dans l'image
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        raise ValueError("Erreur : extraction de visage vide.")
    return face

def save_face_numpy(face, filename):
    """Sauvegarde la matrice NumPy."""
    np.save(filename, face)
    print(f"Fichier NumPy sauvegardé : {filename}")

def save_face_image(face, filename):
    """Sauvegarde le visage en image (png/jpg)."""
    cv2.imwrite(filename, face)
    print(f"Image sauvegardée : {filename}")

def display_results_all(image, faces, show_each_face=True):
    """Affiche l'image originale avec tous les rectangles, et optionnellement chaque visage."""
    display_img = image.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_img, f"{i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Tous les visages détectés", display_img)

    if show_each_face:
        # Afficher chaque visage dans sa propre fenêtre
        for i, (x, y, w, h) in enumerate(faces):
            face = extract_face(image, (x, y, w, h))
            cv2.imshow(f"Face_{i}", face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    image_path = "groupe.png"  # <-- à modifier si besoin

    try:
        image = load_image(image_path)
        faces = detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))

        if faces is None:
            return

        # Préparer noms dynamiques et dossier
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = f"{base_name}_faces"
        ensure_dir(out_dir)

        # Trier les faces par coordonnée x  
        faces = sorted(faces, key=lambda b: b[0])

        for i, bbox in enumerate(faces):
            face = extract_face(image, bbox, margin=10)  
            npy_name = os.path.join(out_dir, f"{base_name}_{i}.npy")
            img_name = os.path.join(out_dir, f"{base_name}_{i}.png")

            save_face_numpy(face, npy_name)
            save_face_image(face, img_name)

        # Affichage final
        display_results_all(image, faces, show_each_face=True)

    except Exception as e:
        print("Erreur :", e)

if __name__ == "__main__":
    main()
