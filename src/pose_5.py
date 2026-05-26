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
    best_pose = None
    best_landmark = None
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1,2]: 
                graph_copy = graph.clone()
                estimate_copy = gtsam.Values(initial_estimate)

                graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
                result = optimize(graph_copy, estimate_copy)
                graph_copy = add_landmark_measurement(graph_copy, result, pose_5, landmark)
                result = optimize(graph_copy, estimate_copy)

                marginals = gtsam.Marginals(graph_copy, result)
                selection_metric = (marginals.marginalCovariance(X(1)).trace() +
                    marginals.marginalCovariance(X(2)).trace() +
                    marginals.marginalCovariance(X(3)).trace())

                true_poses = {
                    X(1): gtsam.Pose2(0.0, 0.0, 0.0),
                    X(2): gtsam.Pose2(2.0, 0.0, 0.0),
                    X(3): gtsam.Pose2(4.0, 0.0, 0.0),
                }

                returned_metric = sum(
                    np.sum(np.abs(result.atPose2(key).matrix() - true_pose.matrix()))
                    for key, true_pose in true_poses.items()
                )

                if selection_metric < best_sum:
                    best_sum = selection_metric
                    best_pose = pose_key
                    best_landmark = landmark
                    best_returned_metric = returned_metric

    return best_pose, best_landmark, best_returned_metric