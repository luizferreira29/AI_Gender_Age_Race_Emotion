pip install deepface

from deepface import DeepFace
from IPython.display import Image

path = "/content/testyoda2.png"
Image(filename=path)

## when a real human cut enforce_detection
obj = DeepFace.analyze(
    img_path=path,
    actions=['age', 'gender', 'race', 'emotion'],
    enforce_detection=False,
    prog_bar=False
)

emotions = obj['emotion']
max(emotions, key=emotions.get)

obj['gender']

races = obj['race']
max(races, key=races.get)

obj['age']


