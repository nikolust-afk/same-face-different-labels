# Same Face, Different Labels

A subject poses for the camera. After a 3, 2, 1 countdown a photo is taken, and fourteen face-analysis models read it. The page shows each model's label in six categories: Is it human?, Age, Gender, Race, Expression and BMI. Beside each label it gives the score and the dataset the model learned from.

All the models run in the browser. The photo never leaves the device and is not stored.

## Run it

- **Online:** GitHub Pages serves this repository as it is. Open the Pages URL and allow camera access.
- **Locally:** run `python -m http.server 8000` in this folder and open `http://localhost:8000`. The camera only works over https or on localhost.
- **Controls:** press Take photo, Space or Enter to start the countdown. Press again for the next subject.

## Models and where they learned

| Category | Model | Learned from |
|---|---|---|
| Human | Tiny Face Detector (face-api.js) | Not disclosed |
| Human | BlazeFace (Human) | Not stated |
| Human | CenterNet (Human) | COCO |
| Human | Anti-spoofing (Human) | Kaggle real-or-fake faces |
| Human | Liveness (Human) | Not stated |
| Age, gender | face-api.js age/gender net | UTKFace, FG-NET, ChaLearn, Wiki, IMDB, CACD, MegaAge, MegaAge-Asian |
| Age, gender | SSR-Net (Human) | IMDB-WIKI |
| Age, gender | FaceRes (Human) | Not stated |
| Age, gender, race | GEAR (Human) | Not stated |
| Age, gender | InsightFace genderage | Not stated |
| Age, gender, race | FairFace | FairFace |
| Expression | face-api.js expression net | Not disclosed |
| Expression | oarriaga emotion (Human) | FER-2013 |
| Age, sex, BMI | face2bmi (VGG16) | Polk County Jail mugshots, VGG Face |

The BMI category comes from the WHO cut-offs applied to the model's number: under 18.5 is underweight, under 25 normal weight, under 30 overweight, under 40 obese, and 40 or more morbidly obese.

## Credits

- [face-api.js](https://github.com/vladmandic/face-api) and [Human](https://github.com/vladmandic/human) by Vladimir Mandic.
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime).
- [FairFace](https://github.com/joojs/fair-face-recognition) by Kärkkäinen and Joo.
- [InsightFace](https://github.com/deepinsight/insightface) genderage. Its models are licensed for non-commercial research only.
- [Face-Says-Bmi](https://github.com/thebitanpaul/Face-Says-Bmi), converted to ONNX.
- IBM Plex Mono typeface, under the SIL Open Font License.

`sample.jpg` shows a generated face of a person who does not exist. It is used only to warm up the models.
