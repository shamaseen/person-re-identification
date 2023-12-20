from typing import Any
import numpy as np

class re_id_process:
    def __init__(self,aspect_ratio_thresh:float,min_box_area:float,reid,frame_thred:int,reid_thred:int) -> None:
        self.first_time=True
        self.aspect_ratio_thresh=aspect_ratio_thresh # aspect ratio thresh for bbox
        self.min_box_area=min_box_area  # min value for bbox
        self.feats = dict()
        self.reid=reid
        self.frame_thred= frame_thred  #  how many frame dose the object be tracked
        self.reid_thred=reid_thred
    def defind_values(self,t,tracker) -> None:
        self.tracker=tracker
        self.t=t
        self.tlwh = t.tlwh
        self.tlbr = abs(t.tlbr)
        self.tid = t.track_id
        self.tcls = t.cls
        self.vertical = self.tlwh[2] / self.tlwh[3] >self.aspect_ratio_thresh
        self.horizontal= self.tlwh[3] / self.tlwh[2] >self.aspect_ratio_thresh*2.5
    def process(self,frame):
        if self.first_time:
            self.feats[self.tid] = self.reid._features([frame[int(self.tlbr[1]):int(self.tlbr[3]), int(self.tlbr[0]):int(self.tlbr[2])]])  # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])
            self.first_time=False
        elif (self.t.end_frame-self.t.start_frame)<self.frame_thred :
            new_person=self.reid._features([frame[int(self.tlbr[1]):int(self.tlbr[3]), int(self.tlbr[0]):int(self.tlbr[2])]])
            min_distance=[np.mean(self.reid.compute_distance(new_person, self.feats[i])) for i in self.feats.copy().keys()]

            feat_match_index=list(self.feats)[min_distance.index(min(min_distance))]
            print(f'compute_distance is {min_distance} for id {self.t.track_id}')

            # check if the object is tracked
            # check_if_object_already_tracked=[count for count,i in enumerate(set(self.tracker.tracked_stracks)) if i.track_id==feat_match_index]
            # if (min(min_distance) <reid_thred) and (id_index !=[]) and (t.end_frame - t.start_frame)<frame_thred  and check_if_object_already_tracked ==[]:
            if (min(min_distance) <self.reid_thred) :
                # get the id for match person
                id__remove_index=[count for count,i in enumerate(set(self.tracker.removed_stracks)) if i.track_id==feat_match_index]
                id__lost_index=[count for count,i in enumerate(set(self.tracker.lost_stracks)) if i.track_id==feat_match_index]
                if id__remove_index !=[] :
                    index=id__remove_index[0]
                    self.tracker.removed_stracks=list(set(self.tracker.removed_stracks))
                    # remove the new trackid id because we found match
                    self.tracker.tracked_stracks.remove(self.t)
                    # get the strack for the match persin
                    self.tracker.tracked_stracks.append(self.tracker.removed_stracks[index])
                    self.tracker.tracked_stracks[-1].track_id=feat_match_index
                    self.tracker.tracked_stracks[-1].re_activate(self.t, self.tracker.frame_id)
                    self.tracker.removed_stracks[0].sub_id()
                    # remove it from removed stracks list
                    self.tracker.removed_stracks.pop(index)
                elif  id__lost_index !=[]:
                    index=id__lost_index[0]
                    self.tracker.lost_stracks=list(set(self.tracker.lost_stracks))
                    # remove the new trackid id because we found match
                    self.tracker.tracked_stracks.remove(self.t)
                    # get the strack for the match persin
                    self.tracker.tracked_stracks.append(self.tracker.lost_stracks[index])
                    self.tracker.tracked_stracks[-1].track_id=feat_match_index
                    self.tracker.tracked_stracks[-1].re_activate(self.t, self.tracker.frame_id)
                    # remove the new trackid id because we found match
                    self.tracker.lost_stracks.pop(index)
                    self.t.sub_id()
                # # remove the track id from match dict
                # if t.track_id in feats.keys() :
                #     del feats[t.track_id]
                else:
                    self.t.sub_id() if feat_match_index != self.t.track_id else None
                    self.t.re_activate(self.t, self.tracker.frame_id)
                    self.t.track_id=(feat_match_index)

            # elif (t.track_id not in feats.keys()) and int(frame_thred//3)<(t.end_frame-t.start_frame)<frame_thred:
            elif (self.t.track_id not in self.feats.keys()) and int(self.frame_thred//3)<(self.t.end_frame-self.t.start_frame)<self.frame_thred:
                self.feats[self.tid] = self.reid._features([frame[int(self.tlbr[1]):int(self.tlbr[3]), int(self.tlbr[0]):int(self.tlbr[2])]])  # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])

        else:
            if self.t.end_frame-self.t.start_frame < 5:
                self.t.sub_id()
                self.tracker.tracked_stracks.remove(self.t)
            # for removed in list(set(tracker.removed_stracks)):
            #     if removed.tracklet_len < 5:
            #         t.sub_id()
            #         self.tracker.removed_stracks.remove(t)
        return self.t,self.tracker
    def __call__(self,t,traker,frame) -> Any:
        self.defind_values(t,traker)
        return self.process(frame)
# if __name__=='__main__':
#     reid_process=re_id_process(aspect_ratio_thresh=1.6,min_box_area=1e-3,reid=reid,frame_thred=10,reid_thred=700)
#     t,tracker=reid_process(frame)