"""
This is a dummy file for HW5 of CSE353 Machine Learning, Fall 2020
You need to provide implementation for this file

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 26-Oct-2020
Last modified: 26-Oct-2020
"""

import random
import numpy as np
from hw5 import k_means, assignment
from tqdm import tqdm

class TrackletClustering(object):
    """
    You need to implement the methods of this class.
    Do not change the signatures of the methods
    """

    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.feature1 = []

    def add_tracklet(self, tracklet):

        first = tracklet['tracks'][0][1:]
        last = tracklet['tracks'][-1][1:]
        mid1_x = (first[0] + first[2]) / 2
        mid1_y = (first[1] + first[3]) / 2
        mid2_x = (last[0] + last[2]) / 2
        mid2_y = (last[1] + last[3]) / 2
        self.feature1.append([mid1_x, mid1_y, mid2_x, mid2_y])

        "Add a new tracklet into the database"
        '''
        for tracklets in tracklet:
            print(tracklets['tracks'][0])
            first = tracklets['tracks'][0][1:]
            last = tracklets['tracks'][-1][1:]
            mid1_x = (first[0] + first[2])/2
            mid1_y = (first[1] + first[3])/2
            mid2_x = (last[0] + last[2]) / 2
            mid2_y = (last[1] + last[3]) / 2
            feature.append([mid1_x,mid1_y,mid2_x,mid2_y])
        '''


    def build_clustering_model(self):
        "Perform clustering algorithm"
        self.centroid = k_means(np.array(self.feature1),self.num_cluster )


    def get_cluster_id(self, tracklet):
        """
        Assign the cluster ID for a tracklet. This funciton must return a non-negative integer <= num_cluster
        It is possible to return value 0, but it is reserved for special category of abnormal behavior (for Question 2.3)
        """
        feature1 = []
        first = tracklet['tracks'][0][1:]
        last = tracklet['tracks'][-1][1:]
        mid1_x = (first[0] + first[2]) / 2
        mid1_y = (first[1] + first[3]) / 2
        mid2_x = (last[0] + last[2]) / 2
        mid2_y = (last[1] + last[3]) / 2
        feature1.append([mid1_x, mid1_y, mid2_x, mid2_y])

        idx = assignment(feature1, self.centroid).item() + 1
        return idx

        #return random.randint(0, self.num_cluster)
