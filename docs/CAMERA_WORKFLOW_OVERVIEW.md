# Camera Workflow Overview

Denne fil forklarer de vigtigste kamera- og visionfiler i repoet fra start til slut, så det er nemt at se, hvad der bruges til hvad.

## Overblik

Det nuvaerende workflow er:

1. Kalibrer kameraet med et checkerboard.
2. Gem kalibreringskonstanter i `calibration_data.npz`.
3. Brug konstanterne til at undistorte billeder eller livefeed.
4. Vaelg top-down metode: ArUco-markers, HSV-baseret baneramme eller manuel 4-punkts selection.
5. Find banens geometri.
6. Warp billedet til et top-down view.
7. Brug manuel 4-punkts selection som fallback hvis den automatiske hjoernefinding er ustabil.

## Centrale filer

### [tools/calibrate_camera.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/calibrate_camera.py)

Dette er scriptet til selve kamerakalibreringen.

Det gør følgende:

- aabner livefeed fra kameraet
- finder checkerboard-hjoerner
- vurderer live om et billede er godt nok til kalibrering
- gemmer gode checkerboard-billeder internt som punktpar
- koerer `cv2.fisheye.calibrate(...)`
- gemmer resultatet i `calibration_data.npz`

Det er her vi finder:

- `K`: kameramatrix
- `D`: distortion-koefficienter
- `image_size`: den oploesning kalibreringen er lavet til

Bruges saadan:

```bash
python3 tools/calibrate_camera.py
```

### [calibration_data.npz](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/calibration_data.npz)

Dette er outputfilen fra kalibreringen.

Den indeholder:

- `K`
- `D`
- `image_size`

Den bruges af de andre scripts, saa de kan undistorte billeder uden at kalibrere igen.

## Undistortion

### [camera/imageprocessing.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/camera/imageprocessing.py)

Denne fil indeholder hjaelpefunktioner til billedbehandling.

Lige nu er de vigtigste funktioner:

- `imageprocessing(img, colorspace)`
- `undistort_with_calibration(img, calibration_file, balance=0.0)`

`undistort_with_calibration(...)`:

- loader `K` og `D` fra `calibration_data.npz`
- checker at billedets oploesning matcher kalibreringsopluesningen
- bruger OpenCV fisheye-funktionerne til at undistorte billedet

Det er denne funktion, de andre scripts genbruger.

### [tools/undistort_bane.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/undistort_bane.py)

Dette er et lille tests script til stillbilleder.

Det:

- loader `Bane.png`
- anvender `undistort_with_calibration(...)`
- gemmer resultatet som `Bane_undistorted.png`

Bruges saadan:

```bash
python3 tools/undistort_bane.py
```

### [tools/live_undistort.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/live_undistort.py)

Dette script bruges til hurtigt at teste kalibreringen paa livefeedet.

Det:

- loader `calibration_data.npz`
- aabner kameraet
- bygger fisheye-remap
- viser originalt og undistortet billede side om side

Bruges saadan:

```bash
python3 tools/live_undistort.py
```

Hvis det undistortede billede ser helt sort ud, er kalibreringen typisk ustabil eller lavet paa for faa / for daarlige checkerboard-billeder.

## Top-down workflow

### [tools/auto_topdown_aruco.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/auto_topdown_aruco.py)

Dette script laver et automatisk top-down view baseret paa 4 ArUco-markers i stedet for HSV-segmentering.

Det:

1. loader `calibration_data.npz`
2. aabner livefeed fra kameraet
3. undistorter hvert frame foerst med `undistort_with_calibration(...)`
4. detekterer ArUco markers med `cv2.aruco.DICT_4X4_50`
5. forventer ID `0`, `1`, `2`, `3` som henholdsvis top-left, top-right, bottom-right og bottom-left
6. bygger `src_points` i fast clockwise orden fra marker-centrene
7. beregner en homography mod et padded top-down output, saa arenaens yderkanter kommer med
8. cacher transformationsmatrixen, saa top-down viewet stadig virker, hvis en marker kortvarigt bliver daekket

Vinduer:

- `Live Feed (Debug)`: undistortet livefeed med marker-highlights, centerpunkter og status
- `Top-Down View`: warpet top-down output eller placeholder, indtil alle 4 markers er laast

Konfiguration:

- `MARKER_DIST_X_CM` og `MARKER_DIST_Y_CM` skal opdateres til de faktiske center-til-center afstande, naar jig-placeringen er endelig
- `EDGE_OFFSET_CM` bruges til at tage de sidste centimeter fra marker-center til fysisk banevaeg med i warp'en
- `PIXELS_PER_CM` styrer output-oploesningen

Bruges saadan:

```bash
python3 tools/auto_topdown_aruco.py
```

Tryk `q` for at afslutte.

### [tools/live_topdown_view.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/live_topdown_view.py)

Dette er det mest komplette live-debug-script i den nuvaerende gren.

Det koerer hele den relevante testkæde:

1. loader `calibration_data.npz`
2. undistorter livefeedet
3. konverterer billedet til HSV
4. segmenterer den orange baneramme med trackbars
5. finder stoerste kontur
6. approksimerer konturen med `cv2.approxPolyDP`
7. checker om polygonen har 4 hjoerner
8. warper billedet til et top-down view hvis geometrien er gyldig

Vinduer:

- `LiveFeed`: undistortet billede med konturer, status og HSV-sliders
- `HSV Mask`: viser hvad HSV-filteret faktisk udvaelger
- `TopDownView`: viser warp eller placeholder hvis der ikke er 4 hjoerner

Bruges saadan:

```bash
python3 tools/live_topdown_view.py
```

Tryk `q` for at afslutte.

### [tools/manual_topdown_view.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/manual_topdown_view.py)

Dette script er den manuelle fallback til top-down view, naar den automatiske HSV-baserede hjoernefinding ikke er stabil nok.

Det:

1. aabner livefeed fra kameraet
2. undistorter hvert frame med `undistort_with_calibration(...)`
3. viser en loupe / forstorrelsesvisning af omraadet under musen
4. lader brugeren vaelge 4 punkter med venstreklik
5. nulstiller valgte punkter med hoejreklik eller `r`
6. bygger en perspektivtransformation, saa snart der er valgt praecist 4 punkter
7. viser et live `Top-Down View`, som opdateres med den gemte transformationsmatrix

Vinduer:

- `Manual Top-Down Selector`: undistortet livefeed med punkter, linjer og loupe
- `Top-Down View`: live warp baseret paa de 4 manuelt valgte punkter

Bruges saadan:

```bash
python3 tools/manual_topdown_view.py
```

Tryk `q` for at afslutte.

### [tools/robot_origin_calibration.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/robot_origin_calibration.py)

Dette script kalibrerer robotens lokale nulpunkt / center of rotation ud fra en eller to ArUco-markers monteret paa robotten.

Det:

1. loader `calibration_data.npz`
2. undistorter livefeedet
3. laver eller genbruger en top-down homography
4. detekterer robot-markerne i top-down billedet
5. samler marker-positioner mens robotten drejer paa stedet
6. fitter en cirkel med `cv2.minEnclosingCircle(...)`
7. checker om spin-banen ligner en ellipse med `cv2.fitEllipse(...)`
8. gemmer `dx`, `dy` og `alpha` i `robot_calibration.json`
9. viser live robot-origin som debug-overlay efter kalibrering

Standardvaerdierne for kamera, marker-ID og parallax-parametre ligger som variabler oeverst i scriptet:

- `CAMERA_INDEX`
- `ROBOT_MARKER_IDS`
- `MARKER_HEIGHT_CM`
- `CAMERA_HEIGHT_CM`
- `CALIBRATION_PLANE_HEIGHT_CM`

Bruges normalt saadan:

```bash
python3 tools/robot_origin_calibration.py
```

Man kan stadig midlertidigt overskrive variablerne fra terminalen, f.eks. med to markers:

```bash
python3 tools/robot_origin_calibration.py --camera-index 0 --marker-ids 10 11 --marker-height-cm 9 --camera-height-cm 179 --calib-z-cm 7
```

Foerste gang scriptet startes, skal der vaelges fire arena-hjoerner i det undistortede billede:

- venstreklik: tilfoej punkt
- hoejreklik eller `r`: nulstil punkter
- `q`: afslut

Naar top-down viewet er aktivt:

- `c`: start dataopsamling mens robotten spinnes 360 grader paa stedet
- `s`: stop dataopsamling og beregn robot-origin
- `Enter`: gem offsets efter robotten er rettet fremad langs top-down billedets positive Y-retning
- `r`: vaelg ny homography
- `q`: afslut

Scriptet aabner ogsaa et `Robot Origin Calibration - Geometry` vindue med samme type parallax-parametre som `topdown_object_detector.py`:

- `Cam h cm`: kameraets hoejde over gulvet
- `Marker h cm`: ArUco-markerens hoejde over gulvet
- `Calib z cm`: hoejden paa det plan, som top-down homographyen er kalibreret paa
- `Cam C X` / `Cam C Y`: kameraets optiske centrum i top-down billedet

Hvis `robot_calibration.json` allerede findes, bruges de gemte parallax-vaerdier som startvaerdier for sliders. Hvis man vil koere uden sliders, kan man bruge `--no-geometry-trackbars`.

Outputfiler:

- `robot_topdown_homography.npz`: gemt manuel top-down transform, saa de fire hjoerner ikke skal klikkes hver gang
- `robot_calibration.json`: robot-marker offsets `dx`, `dy`, `alpha_rad` og `alpha_deg`

Hvis ellipse-ratioen bliver for hoej, viser scriptet en advarsel. Det betyder typisk, at robotten glider under spin, eller at top-down homographyen skal laves om.

`topdown_object_detector.py --drive` kan ogsaa lave en begrænset robot-origin
opdatering som del af `k` drive calibration. Den bruger 360-graders spin-testen
til at fitte et nyt pivot/origin og foreslaar kun nye `dx`/`dy` vaerdier i
`robot_calibration.json`; `alpha_rad` bevares, saa robot-heading ikke aendres
uden en separat forward-alignment kalibrering.

## Main flow og ældre filer

### [Main.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/Main.py)

Dette er et tidligt hovedscript for projektet.

Det:

- tager et billede fra kameraet
- konverterer det til et andet colorspace
- gemmer et debug-billede

Det bruger endnu ikke den nye kalibrering eller top-down pipeline direkte.

### [camera/image.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/camera/image.py)

Indeholder `imagecapture(CameraID)`, som tager et enkelt billede fra et kamera.

### [camera/pictocord.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/camera/pictocord.py)

En tidlig skitse til at oversaette et billede til et koordinatsystem.

Filen virker mere som en placeholder end som en faerdig del af pipeline'en lige nu.

### [camera/calibration.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/camera/calibration.py)

Denne fil indeholder generelle checkerboard- og warp-hjaelpere til tests.

Den bruges ikke som det primære live-kalibreringsscript. Den er mere en samling utility-funktioner til smoke tests og eksperimenter.

## Test- og hjælpefiler

### [tools/checkerboard_smoke_test.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/checkerboard_smoke_test.py)

Testscript til checkerboard-detektion paa et fast testbillede.

### [tools/perspective_warp_test.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/perspective_warp_test.py)

Testscript til perspektivtransformation paa et fast billede.

### [test/test_checkerboard_smoke.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/test/test_checkerboard_smoke.py)

Automatisk test for checkerboard-relateret funktionalitet.

### [test/test_perspective_warp.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/test/test_perspective_warp.py)

Automatisk test for perspektiv-warp.

## Typisk brug i praksis

Hvis man vil arbejde fra start til slut, er den normale rækkefølge:

1. Koer [tools/calibrate_camera.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/calibrate_camera.py)
2. Kontroller at `calibration_data.npz` bliver oprettet
3. Test undistortion med [tools/live_undistort.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/live_undistort.py) eller [tools/undistort_bane.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/undistort_bane.py)
4. Hvis arenaen er sat op med ArUco-jig, koer [tools/auto_topdown_aruco.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/auto_topdown_aruco.py) for marker-baseret top-down warp
5. Ellers koer [tools/live_topdown_view.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/live_topdown_view.py) for at tune HSV og teste top-down warp
6. Hvis den automatiske hjoernefinding er ustabil, koer [tools/manual_topdown_view.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/manual_topdown_view.py) som manuel fallback
7. Koer [tools/robot_origin_calibration.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/robot_origin_calibration.py), naar robot-marker offsets og center of rotation skal kalibreres
8. Brug [tools/topdown_object_detector.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/topdown_object_detector.py) hvis du vil detektere roede zoner, hvide bolde og orange bolde i et top-down billede og se baade annoteret kameravisning og 2D-skema. Koer f.eks. `python3 tools/topdown_object_detector.py --image test_topdown.png`, `python3 tools/topdown_object_detector.py --live`, eller `python3 tools/topdown_object_detector.py --video videos/run-001.mp4`. Hvis en optagelse fra samme kamera bliver laest i en anden oploesning, kan den afspilles med `--resize-video-to-calibration`. Lokale videooptagelser laegges i `videos/`, som er ignoreret af git.
9. Integrer de dele, der virker, ind i den endelige vision-pipeline

## Kort opsummering

Hvis man kun skal huske tre filer, er det:

- [tools/calibrate_camera.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/calibrate_camera.py): laver kalibreringen
- [camera/imageprocessing.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/camera/imageprocessing.py): anvender kalibreringen
- [tools/auto_topdown_aruco.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/auto_topdown_aruco.py): laver top-down warp ud fra 4 ArUco-markers med cached homography
- [tools/live_topdown_view.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/live_topdown_view.py): viser den samlede live-debugkæde
- [tools/topdown_object_detector.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/topdown_object_detector.py): detekterer bolde og roede zoner i top-down view med trackbars og 2D-skema

Hvis den automatiske top-down detection fejler, er den vigtigste fallback:

- [tools/manual_topdown_view.py](/Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo/tools/manual_topdown_view.py): manuel 4-punkts selection med loupe og live top-down warp
