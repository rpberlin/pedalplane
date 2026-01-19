import bpy
import math

def create_part(name, start, end, radius):
    dx, dy, dz = end[0]-start[0], end[1]-start[1], end[2]-start[2]
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    if dist < 0.001: return None
    
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=dist)
    obj = bpy.context.object
    obj.name = name
    obj.location = (start[0] + dx/2, start[1] + dy/2, start[2] + dz/2)
    
    phi = math.atan2(dy, dx)
    theta = math.acos(dz/dist)
    obj.rotation_euler = (0, theta, phi)
    return obj

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# --- SETTINGS ---
torso_radius = 0.15 
leg_y_offset = 0.12 
arm_y_offset = 0.20 
foot_length = 0.25 # Length of the foot from heel/ankle to toe

# --- COORDINATES ---
hips_l = (0, leg_y_offset, 0)
hips_r = (0, -leg_y_offset, 0)
neck = (-0.3, 0, 0.6)
head_center = (-0.37, 0, 0.74)

# Legs (Parallel)
knee_l = (0.45, leg_y_offset, 0.2)
ankle_l = (0.8, leg_y_offset, 0.1)
# TOES: Now rotated 90 degrees UP from the ankle (Z-axis)
toe_l = (0.9, leg_y_offset, -0.1 + foot_length+.2)

knee_r = (0.45, -leg_y_offset, 0.1)
ankle_r = (0.8, -leg_y_offset, -0.1)
toe_r = (1, -leg_y_offset, -0.1 + foot_length)

# Arms
shoulder_l = (-0.2, arm_y_offset, 0.5)
elbow_l = (0.05, arm_y_offset, 0.3)
hand_l = (0.25, arm_y_offset, 0.35)

shoulder_r = (-0.2, -arm_y_offset, 0.5)
elbow_r = (0.05, -arm_y_offset, 0.3)
hand_r = (0.25, -arm_y_offset, 0.35)

# --- GENERATE ---
create_part("Torso", (0,0,0), neck, torso_radius)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.16, location=head_center)

# Left Side
create_part("Thigh_L", hips_l, knee_l, 0.07)
create_part("Shin_L", knee_l, ankle_l, 0.05)
create_part("Foot_L", ankle_l, toe_l, 0.04)
create_part("UpperArm_L", shoulder_l, elbow_l, 0.04)
create_part("Forearm_L", elbow_l, hand_l, 0.03)

# Right Side
create_part("Thigh_R", hips_r, knee_r, 0.07)
create_part("Shin_R", knee_r, ankle_r, 0.05)
create_part("Foot_R", ankle_r, toe_r, 0.04)
create_part("UpperArm_R", shoulder_r, elbow_r, 0.04)
create_part("Forearm_R", elbow_r, hand_r, 0.03)

# Join all
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.join()