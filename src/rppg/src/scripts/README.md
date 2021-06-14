The pretrianed models are excluded from the git due to maximum size of the repo. Download the models and put the in the correct dirs:
- put the landmark detector (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) in the following directory: models/landmark_detector/
- put the face detector files (https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) in : models/retinface/weights

to run FLIR camera with face detection + landmark tracking: 
> python3 main.py


IMPORTANT: 
- always be careful which resolution you are choosing!
- result.csv is structured as [subject_nr, trial, hr_gt, hr, spo2_gt, spo2, ror]
  (ror only added if mode == train)
- jetson: if fps drops after some time (and memory not full): CPU overheat. make sure switch fan on ???
TODOs:
- implement multithreading
- test jpg compression
- write preprocessing pipelien for bwh_small
  
secondary todos:
- combine main and main_video
- check if skipping frame works if face is not detected  
- find optimal parameters to optimze for pos and chrom
- (calibrate pbv vector )
- implement dynamic roi
- build a setup with fixed lighting condition (<-> constant lighting on roi)
- rotate forhead roi
- make subwindows with live plotting of bvp signal
- implement stabilizer


DONE:
- implement skin selection (+skip preprocessing when skin selection)
- make plot: increasing resolution, rppg processing time and general fps
- validate on more subjects (test best results on test set)
- test pos with temporal normalization (pos_new_3_train)
- rewrite rppg class
- find a way to overcome "holes" in rppg signal (i.e will get skipped)
- increasing bbox size
- calculate rppg signal in real time with fifo buffer (s.t if buffer is full -> calculate hr online)
- clipping hr change
- fix slicing in preprocessing
- save results in json or np.array ie and write evaluatoion algo (mae, rmse)
- in post processing: split in batches of n = 500 ie and make estimate
- record rppg singal avg and pulse and save it
- detect Sp02
- splitting forehead is slow when n is too small
- evaluate rppg from saved video input
- split forehead in n equal parts
- calculate rppg signal from roi
- detect HR from rppg signal

DIFF: main_video & main:
- buffer live evaluation
- skip frames if face not detected
- write results in results csv


SECONDARYS:
- fix: FLIR camera has low fps sometimes. maybe throttled when getting too hot? 
- make system portable to other devices (x86): isntead of using jetson.display use cv2 show, disable tensorRT
- make multi person rppg possible
