cmake_minimum_required(VERSION 3.8)
project(physics_amm_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AnalyticModel.msg"
  "srv/StartSession.srv"
  "srv/EndSession.srv"
  "srv/GetModel.srv"
  "srv/SetSessionData.srv"
  "action/LearnModel.action"
)

ament_package()
