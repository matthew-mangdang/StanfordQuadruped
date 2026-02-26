#
# Copyright 2024 MangDang (www.mangdang.net) 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description: You can use the following FPC(Flexible Programmable Choreography) APIs to define your Mini Pupper to dance.
#              There are 3 levels of APIs
#                 Level 1(for beginners): Simple APIs without input parameters
#                 Level 2(for makers): APIs with input parameters
#                 Level 3(for beyond): Samples delicately control the foot locations, move speed, and attitudes at each execution time.
#
# Test method 1 by the controller:
#   step1: Pair the controller to your Mini Pupper after power on
#   step2: Click controller "L1" button
#   step3: Click controller "Circle" button 
#   the mini pupper will dance based on your following script.
#
#
#Test method 2 by command line:
#   After editing this file, run run_danceActionList.py to do your designed movements
#   $python /home/ubuntu/StanfordQuadruped/run_danceActionList.py
#

from src.MovementGroup import MovementGroups

Move = MovementGroups()
Move.obstacle_climb_front(step_x = 0.045, lift_ht = 0.05, time_uni = 1.5, time_acc = 1.5)
Move.obstacle_climb_rear(step_x = 0.045, lift_ht = 0.05, time_uni = 1.5, time_acc = 1.5)

MovementLib = Move.MovementLib
