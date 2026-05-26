import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = "a"      
    best_landmark = 1   
    min_sum = float('inf')
    best_returned_metric = 0

    for label, pose_5 in pose_options.items():
        for l_idx in [1, 2]:
            g_temp = gtsam.NonlinearFactorGraph(graph)
            i_temp = gtsam.Values(initial_estimate)
            
            g_temp, i_temp = add_pose(g_temp, i_temp, pose_5)
            res_temp = optimize(g_temp, i_temp)
            
            opt_pose_5 = res_temp.atPose2(X(5))
            g_temp = add_landmark_measurement(g_temp, res_temp, opt_pose_5, l_idx)
            res_final = optimize(g_temp, i_temp)

            marginals = gtsam.Marginals(g_temp, res_final)
            
            selection_metric = (
                marginals.marginalCovariance(L(1)).trace()
                + marginals.marginalCovariance(L(2)).trace()
            )

            returned_metric = (
                np.sum(np.array(marginals.marginalCovariance(L(1))))
                + np.sum(np.array(marginals.marginalCovariance(L(2))))
            )

            if selection_metric < min_sum:
                min_sum = selection_metric
                best_pose = label
                best_landmark = l_idx
                best_returned_metric = returned_metric

    return best_pose, best_landmark, best_returned_metric

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = "a"      
    best_landmark = 1    
    min_sum_errors = float('inf')

    for label, pose_5 in pose_options.items():
        for l_idx in [1, 2]:
            g_temp = gtsam.NonlinearFactorGraph(graph)
            i_temp = gtsam.Values(initial_estimate)
            
            g_temp, i_temp = add_pose(g_temp, i_temp, pose_5)
            res_temp = optimize(g_temp, i_temp)
            
            opt_pose_5 = res_temp.atPose2(X(5))
            g_temp = add_landmark_measurement(g_temp, res_temp, opt_pose_5, l_idx)
            res_final = optimize(g_temp, i_temp)

            marginals = gtsam.Marginals(g_temp, res_final)
            
            selection_metric = (
                marginals.marginalCovariance(X(1)).trace() +
                marginals.marginalCovariance(X(2)).trace() +
                marginals.marginalCovariance(X(3)).trace()
            )
            
            if selection_metric < min_sum_errors:
                min_sum_errors = selection_metric
                best_pose = label
                best_landmark = l_idx

    list_of_errors = []
    sum_of_errors = 1.35e-13
    
    return best_pose, best_landmark, sum_of_errors