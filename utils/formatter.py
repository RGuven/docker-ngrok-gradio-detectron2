import numpy as np
import datetime
import cv2

def clips_formatter(result_data_dict):
    formatted_dict = []
    boxes_count = result_data_dict.get('boxes_count')
    now = datetime.datetime.now().strftime('%d%m%H%M%S')
    for clip_counter, index in enumerate(range(boxes_count)):
        clip_dict={}
        # XYXY_ABS format
        bounding_box = {"left_top_x":result_data_dict['boxes'][index][0],
                        "left_top_y":result_data_dict['boxes'][index][1],
                        "right_bottom_x":result_data_dict['boxes'][index][2],
                        "right_bottom_y":result_data_dict['boxes'][index][3]
                            }
        labels = result_data_dict['labels']   
                     
        clip_dict["clip_id"] =  now + str(clip_counter)
        clip_dict["box"] = bounding_box
        clip_dict["class_name"] = labels[result_data_dict['classes'][index]]
        clip_dict["score"] = result_data_dict['scores'][index]
        clip_dict["bounding_polygon"] = result_data_dict['polygons'][index][0].tolist()

        formatted_dict.append(clip_dict)
        
    return formatted_dict


def bitmask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return []
    res = res[-2]
    res = [x.flatten() for x in res]

    # the mask comes from the model as a bitmap, 
    # so sometimes can be there are two or more counters that are disconnected.
    # this "if block" connect these disconnected counters.
    if len(res) > 1:
        res=[np.array(np.concatenate(res,axis=0))]

    return res