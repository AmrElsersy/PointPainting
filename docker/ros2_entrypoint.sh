#!/bin/bash
set -e

# setup ros environment

if [ ${ROS_DISTRO} == "foxy" ] ; then
	source "$ROS2_WS/install/local_setup.bash"
else
	source "/opt/ros/$ROS_DISTRO/setup.bash" 
	source "$ROS_WS/devel/setup.bash"
fi
exec "$@"