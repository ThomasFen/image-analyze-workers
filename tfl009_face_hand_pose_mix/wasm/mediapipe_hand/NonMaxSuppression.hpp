#ifndef __MEDIAPIPE_HAND_NON_MAX_SUPPRESSION_HPP__
#define __MEDIAPIPE_HAND_NON_MAX_SUPPRESSION_HPP__

#include <list>
#include "../hand.hpp"

int non_max_suppression(std::list<palm_t> &hand_list, std::list<palm_t> &hand_sel_list, float iou_thresh, int max_palm_num);

#endif // __MEDIAPIPE_HAND_NON_MAX_SUPPRESSION_HPP__