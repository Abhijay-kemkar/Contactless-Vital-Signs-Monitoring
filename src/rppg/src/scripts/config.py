#Face Detector parameters
NETWORK_TYPE = 'mobile0.25'
FACE_DETECTION_RESOLUTION = (300,300)
CPU = False
TRAINED_MODEL_RES = 'models/retinaface/weights/Resnet50_Final.pth'
TRAINED_MODEL_MOBILE = './models/retinaface/weights/mobilenet0.25_Final.pth'
TRAINED_MODEL_MOBILE_PRETRAIN = "./models/retinaface/weights/mobilenetV1X0.25_pretrain.tar"
TRAINED_MODEL_LANDMARKS = 'models/landmark_detector/shape_predictor_68_face_landmarks.dat'
SAVE_FOLDER = './results/'
ORIGIN_SIZE = False
CONFIDENCE_THRESHOLD = 0.02
TOP_K = 5000
NMS_THRESHOLD =0.4
KEEP_TOP_K = 750
SAVE_IMAGE=False
SHOW_IMAGE=True
VIS_THRESH =0.5

#Light weight Face Dector parameters
CONFIDENCE_THRESHOLD_LW = 0.6
CANDIDATE_SIZE_LW = 1500
INPUT_SIZE_LW = 640
TRAINED_MODEL_SLIM_320 ="./models/ulfg_fd//models/pretrained/version-slim-320.pth"
TRAINED_MODEL_SLIM_640 =   "./models/ulfg_fd/models/pretrained/version-slim-640.pth"
TRAINED_MODEL_RFB_320 = "./models/ulfg_fd/models/pretrained/version-RFB-320.pth"
TRAINED_MODEL_RFB_640 = "./models/ulfg_fd/models/pretrained/version-RFB-640.pth"

#Dataset paths
UBFC_PATH = "/media/7F9C-1517/dataset/"
BWH_SMALL_PATH = "/media/BAB1-9078/data/"
BWH_SMALL_TRIALS = 5

#display paramaeters
MODE_OFFSETX = 0
MODE_OFFSETY = 0
MODE_SETWIDTH = 3840    #max screen resolution
#MODE_SETWIDTH =  4096  #max blackfly resolution
MODE_SETHEIGHT = 2160   #max screen resoltion
#MODE_SETHEIGHT = 3000  #max blackfly resolution

#max roll angle for face
MAX_ROLL_ANGLE = 5

#minium face bounding box for cropping image
MIN_BBOX_SIZE = 50


#FOCAL LENGTH OF CAMERAS
FOCAL_LENGTH_BLACKFLY = 0.012

#FOCAL LENGTH OPEN MV CAMERA
FOCAL_LENGTH_OPENMV = 0.0028

#bad subjects ubfc dataset
NO_DATA = [2,28, 29]
BAD_HR_READING = [11, 24, 20]
HAIR_ON_FOREHEAD = [5,10,11,15,30, 38]
#HAIR_ON_FOREHEAD_PARTIALLY = [4,13, 22, 23, 32, 43, 48, 49]
BAD_SUBJECTS_UBFC = NO_DATA + BAD_HR_READING + HAIR_ON_FOREHEAD#+HAIR_ON_FOREHEAD_PARTIALLY

all_subjects  = list(range(1,50))
GOOD_SUBJECTS_UBFC = list(set(all_subjects) - set(BAD_SUBJECTS_UBFC))
#TEST_SUBJECTS = GOOD_SUBJECTS_UBFC[0:len(GOOD_SUBJECTS_UBFC) // 2]
#TRAIN_SUBJECTS = GOOD_SUBJECTS_UBFC[len(GOOD_SUBJECTS_UBFC) // 2:-1]
#GOOD_SUBJECTS = [25] # 46, 34, 35, 27

#so far all subjects good
GOOD_SUBJECTS_BWH_SMALL = list(range(1,100))

#MODE_SETWIDTH =  2048  #half blackfly resolution
#MODE_SETHEIGHT = 1500  #half blackfly resolution

#MODE_OFFSETX = 2048//2
#MODE_OFFSETY = 1500//2


# argumetns for input
import argparse
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('-save', action='store_true', default=False, help='save video or not')
parser.add_argument('-show', action='store_true', default=False, help='show video or not')
parser.add_argument('--detection_rate', default=10, type=int, help='use only every n-th frame for face detection')
parser.add_argument('--resolution', default="bwh_small", help='change general resolution')
parser.add_argument('--dataset', default='bwh_small', help ='choose from [ubfc, bwh_small]')
parser.add_argument('--fd_resolution', default=FACE_DETECTION_RESOLUTION,
                    help='resoultion for images for face detection')
parser.add_argument('--length', default=float('inf'), help='maximum length of stream')
parser.add_argument('--skin_segm', default=None,
                    help='skin segmenation method in ROI. Choose form [pitas, conaire, none]. If skin selection is chosen, roi area will be changed to skin')
parser.add_argument('-static_face_bb', action='store_true', default=False,
                    help='determines if face bounding box rotates with roll angle of face')
parser.add_argument('-blacken', action='store_true', default=False,
                    help='determines if area outside face bounding box is blackened or not')
parser.add_argument('-useRT', action='store_true', default=False, help='enables tensorRT porting for face detector')
parser.add_argument('--network', default='mobile',
                    help='choose from face detection network backbone [resnet, mobile].Default is mobile')
parser.add_argument('--buffer_size', default=600, help='Buffer size for rppg rignal estimation')
parser.add_argument('--roi_slices', default=3, help='the roi is split into n² equal subregions for spo2 estimation.  ')
parser.add_argument('--fs', default=30, help='sampling rate of recorded video')
parser.add_argument('--stride', default=150, help='sliding window stride')
parser.add_argument('--hr_reset', default=0, help='max ðHR/s change. if equal to zero, hr is disabled')
parser.add_argument('-save_plots', default=False, action='store_true')
parser.add_argument('--mode', default = 'all' ,help=' [train, test, all]')
parser.add_argument('-filter_cn', default=False, action='store_true')
parser.add_argument('-filter_patch', default=False, action='store_true')
parser.add_argument('--method', default ='pos', help = 'choose from [pbv, pos, chrom, abpv]')
parser.add_argument('--spo2_method', default ='pos', help = 'choose from [pbv, pos, chrom, abpv]')
parser.add_argument('-compr', default = False, action ='store_true')
parser.add_argument('-whole_face', default =False, action ='store_true')
parser.add_argument('-temp_norm', default =False, action ='store_true')
parser.add_argument('-crop_face', default = False, action ='store_true', help = 'use cropped face instead of whole face')
parser.add_argument('--roi', default="forehead", help='choose from [forehead, skin]')
parser.add_argument('-rppg_show', action='store_true', default=False, help='show video which will be passed to rppg algo (without bbox or landmark drawings)')

args = parser.parse_args()