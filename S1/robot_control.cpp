// ROS include files
#include <ros/package.h>
#include <ros/ros.h>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>


// Inverse Dynamics Solver
#include <whc/control/configuration.hpp>
#include <whc/control/feedback.hpp>
#include <whc/dynamics/constraint/constraints.hpp>
#include <whc/dynamics/solver/id_solver.hpp>
#include <whc/dynamics/task/tasks.hpp>
#include <whc/qp_solver/qp_oases.hpp>
#include <whc/utils/math.hpp>

// Robotiq Gripper includes
#include <robotiq_2f_gripper_control/Robotiq2FGripper_robot_output.h>


// robot_dart includes
// this is mainly to allow easy creation of DART model from URDF string
#include <robot_dart/robot.hpp>


// Minimal GMR implementation
bool gluInvertMatrix(double m[4], double* invOut, double& deter)
{

    double inv[4];
    int i;
    deter = m[0] * m[3] - m[1] * m[2];

    inv[0] = m[3] / deter;
    inv[1] = -m[1] / deter;
    inv[2] = -m[2] / deter;
    inv[3] = m[0] / deter;

    if (deter == 0)
        return false;

    for (i = 0; i < 4; i++)
        invOut[i] = inv[i];

    return true;
}

double GaussianPDF(std::vector<double> input, std::vector<double> Mean, std::vector<std::vector<double>> covariance)
{
    std::vector<double> dif;
    int dim = input.size();
    double inputD[4];
    int k = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            inputD[k] = covariance[i][j];
            //cout << "covariance " << covariance[i][j] << "  " << endl;
            //cout << "inputD " << inputD[k] << "  " << endl;
            k++;
        }
    }

    bool ok;
    double invOut[4];
    double deter = 0;
    ok = gluInvertMatrix(inputD, &invOut[0], deter);
    std::vector<std::vector<double>> invMatrix(2, std::vector<double>(2));

    int cRow = 0;
    int cCol = 0;
    for (int i = 0; i < 4; i++) {
        if (cCol < 2) {
            invMatrix[cRow][cCol] = invOut[i];
        }
        else {
            cCol = 0;
            cRow++;
            invMatrix[cRow][cCol] = invOut[i];
        }
        //cout << "invMatrix " << invMatrix[cRow][cCol] << "  " << endl;
        cCol++;
    }

    if (ok) {

        std::vector<double> dif(2, 1);
        dif[0] = input[0] - Mean[0];
        dif[1] = input[1] - Mean[1];

        std::vector<double> cA(2, 1);
        // invMatrix*dif
        for (int i = 0; i < 2; i++) {
            cA[i] = 0;
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                cA[i] += (dif[j] * invMatrix[i][j]);
            }
        }

        double p = 0.;
        for (int i = 0; i < 2; i++) {
            p += (cA[i] * dif[i]);
        }

        p = exp(-0.5 * p) / sqrt(pow(2.0 * M_PI, dim) * (fabs(deter) + 2.2251e-308));
        return p;
    }
    else {
        std::cout << "Error inverting sigma" << std::endl;
        return 0;
    }
}

std::vector<double> GMR(std::vector<double> Priors, std::vector<std::vector<double>> Mean, std::vector<std::vector<std::vector<double>>> covariance, std::vector<double> input, std::vector<int> in, std::vector<int> out)
{

    int nbStates = 3;
    std::vector<double> Pxi(3, 1);
    std::vector<std::vector<double>> y_tmp(3, std::vector<double>(2));
    for (int i = 0; i < nbStates; i++) {
        std::vector<double> newMean(2, 1);
        newMean[0] = Mean[in[0] - 1][i];
        newMean[1] = Mean[in[1] - 1][i];

        std::vector<std::vector<double>> newCovariance(2, std::vector<double>(2));
        newCovariance[0][0] = covariance[i][in[0] - 1][in[0] - 1];
        newCovariance[0][1] = covariance[i][in[0] - 1][in[1] - 1];
        newCovariance[1][0] = covariance[i][in[1] - 1][in[0] - 1];
        newCovariance[1][1] = covariance[i][in[1] - 1][in[1] - 1];

        Pxi[i] = Priors[i] * GaussianPDF(input, newMean, newCovariance);
    }

    // calculate the sum of Priors
    double sum = 0.;
    for (int i = 0; i < nbStates; i++)
        sum += Pxi[i];
    // compute beta
    std::vector<double> beta(3, 1);
    for (int i = 0; i < nbStates; i++)
        beta[i] = Pxi[i] / (sum + 2.2251e-308);

    //getchar();
    // compute means y, given input x
    // y_tmp(:,:,j) = repmat(Mu(out,j),1,nbData) + Sigma(out,in,j)/(Sigma(in,in,j)) * (x-repmat(Mu(in,j),1,nbData));
    for (int i = 0; i < nbStates; i++) {

        std::vector<double> newMean(2, 1);
        newMean[0] = Mean[out[0] - 1][i];
        newMean[1] = Mean[out[1] - 1][i];

        std::vector<double> newMean1(2, 1);
        newMean1[0] = Mean[in[0] - 1][i];
        newMean1[1] = Mean[in[1] - 1][i];

        std::vector<double> newMean2(2, 1);
        newMean2[0] = Mean[out[0] - 1][i];
        newMean2[1] = Mean[out[1] - 1][i];

        std::vector<std::vector<double>> newCovariance1(2, std::vector<double>(2));
        newCovariance1[0][0] = covariance[i][out[0] - 1][in[0] - 1];
        newCovariance1[0][1] = covariance[i][out[0] - 1][in[1] - 1];
        newCovariance1[1][0] = covariance[i][out[1] - 1][in[0] - 1];
        newCovariance1[1][1] = covariance[i][out[1] - 1][in[1] - 1];

        std::vector<std::vector<double>> newCovariance2(2, std::vector<double>(2));
        newCovariance2[0][0] = covariance[i][in[0] - 1][in[0] - 1];
        newCovariance2[0][1] = covariance[i][in[0] - 1][in[1] - 1];
        newCovariance2[1][0] = covariance[i][in[1] - 1][in[0] - 1];
        newCovariance2[1][1] = covariance[i][in[1] - 1][in[1] - 1];

        // invert newCovariance2
        double inputD[4];
        int k = 0;
        for (int l = 0; l < 2; l++) {
            for (int j = 0; j < 2; j++) {
                inputD[k] = newCovariance2[l][j];
                k++;
            }
        }

        double invOut[4];
        double deter = 0.;
        gluInvertMatrix(inputD, &invOut[0], deter);
        std::vector<std::vector<double>> invCovariance2(2, std::vector<double>(2));

        // multiple newCovariance1*invCovariance2
        int cRow = 0;
        int cCol = 0;
        for (int j = 0; j < 4; j++) {
            if (cCol < 2) {
                invCovariance2[cRow][cCol] = invOut[j];
            }
            else {
                cCol = 0;
                cRow++;
                invCovariance2[cRow][cCol] = invOut[j];
            }
            cCol++;
        }

        std::vector<std::vector<double>> cA(2, std::vector<double>(2));
        // invMatrix*dif
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                cA[j][k] = 0;
            }
        }

        // multiply 2x2 * 2x2
        cA[0][0] = newCovariance1[0][0] * invCovariance2[0][0] + newCovariance1[0][1] * invCovariance2[1][0];
        cA[0][1] = newCovariance1[0][0] * invCovariance2[0][1] + newCovariance1[0][1] * invCovariance2[1][1];
        cA[1][0] = newCovariance1[1][0] * invCovariance2[0][0] + newCovariance1[1][1] * invCovariance2[1][0];
        cA[1][1] = newCovariance1[1][0] * invCovariance2[0][1] + newCovariance1[1][1] * invCovariance2[1][1];

        // compute (x-repmat(Mu(in,j),1,nbData))
        std::vector<double> dif(2, 1);
        dif[0] = input[0] - newMean1[0];
        dif[1] = input[1] - newMean1[1];

        // Sigma(out,in,j)/(Sigma(in,in,j)) * (x-repmat(Mu(in,j),1,nbData));
        std::vector<double> cB(2, 1);
        // cA*dif
        for (int j = 0; j < 2; j++) {
            cB[j] = 0;
        }

        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                cB[j] += (cA[j][k] * dif[k]);
            }
        }

        // repmat(Mu(out,j),1,nbData) + ans
        // Mean2 + cB
        for (int j = 0; j < 2; j++) {
            y_tmp[i][j] = 0;
        }

        for (int j = 0; j < 2; j++) {
            y_tmp[i][j] = newMean2[j] + cB[j];
        }
    }

    std::vector<double> beta_tmp(3, 1); // same as beta
    beta_tmp = beta;

    // compute y_tmp2
    std::vector<std::vector<double>> y_tmp2(3, std::vector<double>(2));
    // invMatrix*dif
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 2; k++) {
            y_tmp2[j][k] = 0;
        }
    }

    for (int k = 0; k < 2; k++) {
        for (int j = 0; j < 3; j++) {
            y_tmp2[j][k] = beta_tmp[j] * y_tmp[j][k];
        }
    }

    std::vector<double> y(2, 1);
    // invMatrix*dif
    for (int i = 0; i < 2; i++) {
        y[i] = 0;
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            y[i] += y_tmp2[j][i];
        }
    }

    // y is xd (the derivative of x)
    return y;
}
// End of minimal GMR implementation


// Controller class
class MyControl {
public:
    MyControl(ros::NodeHandle& nh, const dart::dynamics::SkeletonPtr& skel)
    {
        _initial_state = true;
        _target_config.resize(7);
        // Initial configuration of the joints
        _target_config << -0.20539679176653725, -0.3890446121543803, 0.24360951157515925, -1.6223705584636583, -0.08254772463511181, 0.5571544055476394, 1.689246188659511;

        _curr_pos = Eigen::VectorXd::Zero(7);
        _curr_vel = Eigen::VectorXd::Zero(7);
        _curr_tau = Eigen::VectorXd::Zero(7);
        _curr_object_pose = Eigen::VectorXd::Zero(6);

        // Set-up inverse dynamics solver
        _solver = std::make_shared<whc::dyn::solver::IDSolver>(skel);
        _solver->set_qp_solver<whc::qp_solver::QPOases>();
        _config = whc::control::Configuration(skel);
        _config.add_eef("body_name_of_your_end_effector", false); // no need for contacts

        // Publisher to command topic (assumes that the robot accepts torque-commands)
        _cmd_pub = nh.advertise<std_msgs::Float64MultiArray>("/path/to/topic/TorqueController/command", 1000);
        // Publisher for gripper commands --  we give a sample for the Robotiq2F 85mm
        _gripper_pub = nh.advertise<robotiq_2f_gripper_control::Robotiq2FGripper_robot_output>("Robotiq2FGripperRobotOutput", 1000);
        _gripper_msg.rACT = 1;
        _gripper_msg.rGTO = 1;
        _gripper_msg.rATR = 0;
        _gripper_msg.rPR = 0;
        _gripper_msg.rSP = 255;
        _gripper_msg.rFR = 150;

        _cmd_msg.data.resize(7);

        _first = true;

        // Subscriber: estimated object position from computer vision
        _object_sub = nh.subscribe("/estimatedObject", 1000, &MyControl::object_cb, this);
        // Subscriber: estimated object dimensions from computer vision
        _dim_sub = nh.subscribe("/estimatedDimensions", 1000, &MyControl::dim_cb, this);

        // Get board transformation for correct commands
        bool got_board = false;
        tf::StampedTransform transform;
        ROS_INFO_STREAM("Trying to get board transformation!");
        while (!got_board) {
            try {
                _tf_listener.lookupTransform("your_robot_base_link", "board",
                    ros::Time(0), transform);
                got_board = true;
            }
            catch (tf::TransformException ex) {
                ROS_ERROR("%s", ex.what());
                ros::Duration(1.0).sleep();
            }
        }

        Eigen::Isometry3d trans;
        tf::transformTFToEigen(transform, trans);
        _board_transform = trans.matrix();
        ROS_INFO_STREAM("Got board transformation: " << _board_transform);
    }

    ~MyControl()
    {
        _cmd_msg.data = {0, 0, 0, 0, 0, 0, 0};
        _cmd_pub.publish(_cmd_msg);
    }

    void joint_state_cb(const sensor_msgs::JointState::ConstPtr& msg)
    {
        for (size_t i = 0; i < 7; i++) {
            _curr_pos[i] = msg->position[i];
            _curr_vel[i] = msg->velocity[i];
            _curr_tau[i] = msg->effort[i];
        }

        _joints_received = true;
    }

    void object_cb(const geometry_msgs::PointStamped::ConstPtr& msg)
    {
        _curr_object_pose[3] = msg->point.x;
        _curr_object_pose[4] = msg->point.y;
        _curr_object_pose[5] = msg->point.z;

        _got_object = true;
    }

    void dim_cb(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        _width = msg->data[0];
        _height_up = msg->data[3];

        _got_dim = true;
    }

    void control()
    {
        if (!_joints_received)
            return;
        // First publish the gripper command
        _gripper_pub.publish(_gripper_msg);

        // Update the model-based controller
        Eigen::VectorXd pos = _curr_pos;
        Eigen::VectorXd vel = _curr_vel;
        Eigen::VectorXd effort = _curr_tau;

        _config.skeleton()->setPositions(pos);
        _config.skeleton()->setVelocities(vel);
        _config.update(false); // no need for contacts
        size_t dofs = _config.skeleton()->getNumDofs();

        // In initial state (or object is not yet available from CV), we go back to the initial configuration
        // we add a small delay to start the motion to give time to the human to manipulate the glass
        if (_initial_state || !_got_object || (_got_object && _delay_counter < _delay_max)) {
            if (_got_object)
                _delay_counter++;
            Eigen::VectorXd target = Eigen::VectorXd::Zero(dofs * 2);
            if (_initial_state && _first) {
                // In the first time, we create the tasks for the inverse dynamics solver
                _solver->clear_all();
                Eigen::VectorXd gweights = Eigen::VectorXd::Constant(target.size(), 0.01);
                gweights.head(dofs) = Eigen::VectorXd::Constant(dofs, 1.);

                _solver->add_task(whc::utils::make_unique<whc::dyn::task::DirectTrackingTask>(_config.skeleton(), target, gweights));

                // Dynamics constraints
                _solver->add_constraint(whc::utils::make_unique<whc::dyn::constraint::DynamicsConstraint>(_config.skeleton(), false));
                _first = false;
            }

            double Kp = 60., Kd = 1.;
            target.head(dofs) = Kp * (_target_config - _curr_pos) - Kd * _curr_vel;

            _solver->tasks()[0]->set_desired(target);
            if ((_target_config - _curr_pos).norm() < 5e-1) {
                _initial_state = false;
                _first = true;
            }
        }
        else {
            if (_first) {
                _solver->clear_all();
                _prev_tau = Eigen::VectorXd::Zero(7);

                _config.eef(0)->desired = _config.eef(0)->state;
                _config.eef(0)->desired.vel = Eigen::VectorXd::Zero(6);
                _config.eef(0)->desired.acc = Eigen::VectorXd::Zero(6);
                _first = false;
            }

            // if the object is being tracked and we haven't yet grasped it
            // set the desired end-effector target to the object
            if (_got_object && !_grasped) {
                Eigen::VectorXd tmp_pose(4);
                tmp_pose.head(3) = _curr_object_pose.tail(3);
                tmp_pose[3] = 1.;

                _config.eef(0)->desired.pose.tail(3) = (_board_transform * tmp_pose).head(3);
            }

            // Task-space PD gains
            whc::control::PDGains gains;
            gains.kp = Eigen::VectorXd::Constant(6, 400.);
            gains.kd = Eigen::VectorXd::Constant(6, 20.);
            gains.kp.head(3) = Eigen::VectorXd::Constant(3, 40.);
            gains.kd.head(3) = Eigen::VectorXd::Constant(3, 2.);

            auto eef = _config.eef(0);
            // Calculate error from desired end-effector location
            Eigen::VectorXd pos_error = eef->desired.pose - eef->state.pose;
            pos_error.head(3) = whc::utils::rotation_error(dart::math::expMapRot(eef->desired.pose.head(3)), dart::math::expMapRot(eef->state.pose.head(3)));

            if (!_grasped) {
                // If we haven't grasped it yet, we use the human-inspired distance-based trajectory (see the paper)
                std::vector<double> priors = {0.24669, 0.33064, 0.42267};
                std::vector<std::vector<double>> mean = {{-0.76027, -0.070237, -0.38292},
                    {-0.058227, -0.010945, -0.1216},
                    {0.060199, 0.029963, 0.047317},
                    {-0.0095265, 0.006203, 0.0078715}};

                std::vector<std::vector<std::vector<double>>> covariance = {{{0.0095135, -0.0019258, 0.00077335, -0.0013122}, {-0.0019258, 0.00095126, -0.0001556, 0.00018462}, {0.00077335, -0.0001556, 0.00017587, -0.00025051}, {-0.0013122, 0.00018462, -0.00025051, 0.00081786}},
                    {{0.0039812, 0.00053782, -0.001004, -0.00034992}, {0.00053782, 0.00014615, -0.00013021, -7.037e-05}, {-0.001004, -0.00013021, 0.00032428, 8.0298e-05}, {-0.00034992, -7.037e-05, 8.0298e-05, 8.3045e-05}},
                    {{0.013152, 0.0019672, -0.00093656, 0.0019493}, {0.0019672, 0.0017942, 9.4689e-05, 0.00032847}, {-0.00093656, 9.4689e-05, 0.00025576, -0.00012418}, {0.0019493, 0.00032847, -0.00012418, 0.00038843}}};

                std::vector<int> in = {1, 2};
                std::vector<int> out = {3, 4};

                double x = std::abs(pos_error[3]);
                double z = std::abs(pos_error[5]);

                std::vector<double> curr = {-x, -z};
                auto out2 = GMR(priors, mean, covariance, curr, in, out);

                double s_x = 1.;
                if (pos_error[3] < 0.)
                    s_x = -1.;
                double s_z = 1.;
                if (pos_error[5] < 0.)
                    s_z = -1.;
                pos_error[3] = s_x * (x + out2[0] * 0.00001);
                pos_error[5] = s_z * (z + out2[1] * 0.00001);
            }

            // Compute final desired end-effector acceleration to pass the inverse dynamics solver
            Eigen::VectorXd acc = eef->desired.acc + gains.kd.cwiseProduct(eef->desired.vel - eef->state.vel) + gains.kp.cwiseProduct(pos_error);

            // Calculate distance to desired end-effector position (usually the object)
            double dist = (eef->desired.pose.tail(3) - eef->state.pose.tail(3)).norm();

            // If we have grasped the object and moved to the final location, open the gripper and update the end-effector desired location
            if (_got_object && !_finished && _grasped && _grasp_counter >= _grasp_max && dist < 0.02) {
                _gripper_msg.rPR = 0;
                _gripper_pub.publish(_gripper_msg);

                _config.eef(0)->desired.pose.tail(3)[0] -= 0.25;
                _config.eef(0)->desired.pose.tail(3)[2] -= 0.02;

                _finished = true;
            }
            // Close the gripper to 80% of the width of the object (to grasp it steadily)
            double w = 0.8 * _width;
            if (w > 0.085)
                w = 0.085;
            if (w < 0)
                w = 0.;
            int gripper_val = 255 - 255 * (w / 0.085);
            double dist_allowed = (0.05 + _width);

            // if we haven't grasped the object yet and the distance from the object is small
            // close the gripper!
            if ((dist - dist_allowed) < 1e-3 && !_grasped) {
                _grasped = true;
                _gripper_msg.rPR = static_cast<uint8_t>(gripper_val);
                _gripper_pub.publish(_gripper_msg);
            }

            // If we just grapsed the object, go to desired delivery location
            if (_got_object && _grasped && !_finished) {
                _grasp_counter++;
                if (_grasp_counter >= _grasp_max) {
                    Eigen::VectorXd tmp_pose(4);
                    tmp_pose.head(3) << 0.195, 0.265, 0.085;
                    tmp_pose[3] = 1.;

                    _config.eef(0)->desired.pose.tail(3) = (_board_transform * tmp_pose).head(3);

                    _config.eef(0)->desired.pose.head(3) << 1.29357, 1.21571, 1.17118;
                }
            }

            // In the first time, we create the tasks for the inverse dynamics solver
            if (_solver->tasks().size() == 0) {
                Eigen::VectorXd weights = Eigen::VectorXd::Constant(6, 2.);
                _solver->add_task(whc::utils::make_unique<whc::dyn::task::AccelerationTask>(_config.skeleton(), "body_name_of_your_end_effector", acc, weights));

                // Regularization
                Eigen::VectorXd target = Eigen::VectorXd::Zero(dofs * 2);
                Eigen::VectorXd gweights = Eigen::VectorXd::Constant(target.size(), 0.01);
                gweights.head(dofs) = Eigen::VectorXd::Constant(dofs, 0.3);

                _solver->add_task(whc::utils::make_unique<whc::dyn::task::DirectTrackingTask>(_config.skeleton(), target, gweights));

                // Dynamics constraints
                _solver->add_constraint(whc::utils::make_unique<whc::dyn::constraint::DynamicsConstraint>(_config.skeleton(), false));
            }
            _solver->tasks()[0]->set_desired(acc);
        }

        _solver->solve();

        // We substract the gravity and coriolis forces from our controller, because our robot already has them
        Eigen::VectorXd Cg = _config.skeleton()->getCoriolisAndGravityForces();
        Eigen::VectorXd commands = _solver->solution().tail(dofs) - Cg;
        _prev_tau = commands + Cg;
        Eigen::VectorXd::Map(_cmd_msg.data.data(), _cmd_msg.data.size()) = commands;

        // Publish the torque commands
        _cmd_pub.publish(_cmd_msg);
    }

protected:
    std::shared_ptr<whc::dyn::solver::IDSolver> _solver;
    whc::control::Configuration _config;
    Eigen::VectorXd _curr_pos, _curr_vel, _curr_object_pose, _prev_tau, _curr_tau;
    Eigen::MatrixXd _board_transform;
    ros::Publisher _cmd_pub, _gripper_pub;
    ros::Subscriber _object_sub, _dim_sub;
    tf::TransformListener _tf_listener;
    std_msgs::Float64MultiArray _cmd_msg;
    robotiq_2f_gripper_control::Robotiq2FGripper_robot_output _gripper_msg;
    bool _first = true, _got_object = false;
    double _width = 0.1;
    bool _got_dim = false;
    bool _joints_received = false;
    bool _grasped = false, _finished = false;
    double _height_up = 0.;
    size_t _grasp_counter = 0, _grasp_max = 40;
    bool _initial_state = true;
    Eigen::VectorXd _target_config;
    size_t _delay_counter = 0, _delay_max = 300;
};

int main(int argc, char** argv)
{
    // Initialize ROS node
    ros::init(argc, argv, "name_of_your_node");

    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // Get description of our robot
    std::string package_path = ros::package::getPath("package_of_robot_description");

    // Get the URDF XML from the parameter server
    std::string urdf_string, full_param;
    std::string robot_description = "robot_description";
    std::string end_effector;

    // gets the location of the robot description on the parameter server
    if (!nh.searchParam(robot_description, full_param)) {
        ROS_ERROR("Could not find parameter %s on parameter server", robot_description.c_str());
        return false;
    }

    // search and wait for robot_description on param server
    while (urdf_string.empty()) {
        ROS_INFO_ONCE_NAMED("name_of_your_node", "name_of_your_node is waiting for model"
                                                 " URDF in parameter [%s] on the ROS param server.",
            robot_description.c_str());

        nh.getParam(full_param, urdf_string);

        usleep(100000);
    }
    ROS_INFO("URDF Loaded");

    // Load the URDF in robot_dart/DART/whc
    std::vector<std::pair<std::string, std::string>> packages = {{"package_of_robot_description", package_path}};
    auto arm = std::make_shared<robot_dart::Robot>(urdf_string, packages, "name_of_your_robot", true);
    arm->set_position_enforced(true);
    arm->fix_to_world();

    MyControl control(nh, arm->skeleton());

    ros::Subscriber sub = nh.subscribe("/path/to/topic/of/robot/joint_states", 1000, &MyControl::joint_state_cb, &control);

    ros::Rate loop_rate(200);

    while (ros::ok()) {
        control.control();
        loop_rate.sleep();
    }

    spinner.stop();

    return 0;
}