import numpy as np
import cv2
import super_gradients
import matplotlib.pyplot as plt
import time
import argparse
import sys
sys.path.append('./SMILEtrack_Official')
sys.path.append('./BoT-SORT/')
from utils_tracker import *
from utils import *
import reid_SOLIDER as Re_ID
from SMILEtrack_Official.tracker.mc_SMILEtrack import SMILEtrack

def main():
    # Load a COCO-pretrained YOLO-NAS-s model
    model_name=opt.model_name
    weight=opt.model_weight

    # Get names and colors
    yolo_nas = super_gradients.training.models.get(model_name, pretrained_weights=weight).cuda()
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    # defind tracker
    tracker = SMILEtrack(opt,opt.frame_rate)

    # reid
    reid = Re_ID.REID(config_file=opt.soldier_config,test_weight_path=opt.soldier_weights)

    # tracking parameters
    conf=opt.conf_thres
    classes_ids_fillter=opt.classes_ids_fillter
    min_box_area=opt.min_box_area
    reid_thred=opt.soldier_reid_thred
    frame_thred= opt.frame_thred #  how many frame dose the object be tracker
    pose_conf=opt.pose_conf
    pose_point_conf=opt.pose_point_conf

    # read video from source
    cap = cv2.VideoCapture(opt.source)

    # reid process
    reid_process=re_id_process(aspect_ratio_thresh=opt.aspect_ratio_thresh,min_box_area=min_box_area,reid=reid,frame_thred=frame_thred,reid_thred=reid_thred)

    # Run inference
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame=cv2.resize(frame,dsize=tuple(opt.frame_size))
        # object detection
        detect = next(iter(yolo_nas.predict(frame, iou=opt.iou_thres,conf=conf,fuse_model=False)))

        # prepare detection results
        filter_results=preprocessing_results(model_name)
        results=filter_results(detect=detect,pose_conf=pose_conf,pose_point_conf=pose_point_conf,conf=conf,classes_ids_fillter=classes_ids_fillter)
        # Tracking
        online_targets = tracker.update(np.array(results), frame)
        for t in online_targets:
            # reid
            t,tracker=reid_process(t,traker=tracker,frame=frame)
            # plot the update tracking
            if "pose" in model_name:
                label = f'{t.track_id}, person'
            else:
                label = f'{t.track_id}, {detect.class_names[int(t.cls)]}'
            plot_one_box(t.tlbr, frame, label=label, color=colors[int(t.track_id) % len(colors)], line_thickness=2)

        # Calculating the fps
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('tracking',frame)
        vis_results=detect.draw()
        cv2.imshow('frame',vis_results)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ReID SOLIDER
    parser.add_argument('--soldier-weights', nargs='+', type=str, default='./SOLIDER-REID/swin_base_market.pth', help='path for solder model path')
    parser.add_argument('--soldier-config', nargs='+', type=str, default='./SOLIDER-REID/configs/market/swin_base.yml', help='path for solder config path')
    parser.add_argument('--soldier-reid-thred', nargs='+', type=int, default=700, help='threshold for reid matching')
    # custom
    parser.add_argument("--frame-rate", type=int, default=25, help="the frame of video of camera")
    parser.add_argument("--frame-thred", type=int, default=10, help="how many frame dose the object be tracker")
    parser.add_argument("--pose-conf", type=float, default=0.6, help="conf for pose")
    parser.add_argument("--pose-point-conf", type=float, default=0.4, help="conf for each 17 point")
    parser.add_argument("--frame-size", type=tuple, default=[1080,720], help="the size frame of video of camera")

    # model
    parser.add_argument('--model-name', nargs='+', type=str, default='yolo_nas_pose_l', help='model')
    parser.add_argument('--model-weight', nargs='+', type=str, default='coco_pose', help='mode weight')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')


    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes-ids-fillter', default=0, type=int, help='filter the needed class')
    parser.add_argument('--name', default='exp', help='save results to project/name')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.5, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=20, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min-box-area', type=float, default=1e-3, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    ## ReID for tracker is off in this project
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()
    opt.ablation = False

    # run the project
    main()
