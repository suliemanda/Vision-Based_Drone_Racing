import pyquaternion
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Gate:
    def __init__(self, pos, att, size=1.45):
        self.pos = np.array(pos)
        self.att = pyquaternion.Quaternion(w=att[0], x=att[1], y=att[2], z=att[3])
        self.gate_dim = size
        self.w_border = 0.3
        self.yaw = 2.0 * np.arcsin(self.att.z)
        self.tl_corner = self.pos + self.att.rotate(
            np.array([0.0, 1.0, 1.0]) * self.gate_dim / 2.0
        )
        self.tr_corner = self.pos + self.att.rotate(
            np.array([0.0, -1.0, 1.0]) * self.gate_dim / 2.0
        )
        self.bl_corner = self.pos + self.att.rotate(
            np.array([0.0, 1.0, -1.0]) * self.gate_dim / 2.0
        )
        self.br_corner = self.pos + self.att.rotate(
            np.array([0.0, -1.0, -1.0]) * self.gate_dim / 2.0
        )

    def is_passed(self, pos: np.ndarray) -> int:
        drone_pos_in_gate_frame = self.att.inverse.rotate(pos - self.pos)
        passed=(drone_pos_in_gate_frame[0] >-0.1 #and drone_pos_in_gate_frame[0] < 0.05
            and abs(drone_pos_in_gate_frame[2]) < 0.9
            and abs(drone_pos_in_gate_frame[1]) < 0.9)
        notoverpassed=(drone_pos_in_gate_frame[0] <0.1)
        if passed and notoverpassed:
            return 1.0
        else:
            if passed:
                return -5.0
            else:
                return 0.0
    # def is_passed(self, pos: np.ndarray,vel:np.array) -> bool:
        # drone_pos_in_gate_frame = self.att.inverse.rotate(pos - self.pos)
        # drone_vel_in_gate_frame =self.att.inverse.rotate(vel)/(np.sqrt(vel[0]**2+vel[1]**2+vel[2]**2)+1e-6)
        # passed=(abs(drone_pos_in_gate_frame[0]) <0.05 #and drone_pos_in_gate_frame[0] < 0.05
            # and abs(drone_pos_in_gate_frame[2]) < 0.9
            # and abs(drone_pos_in_gate_frame[1]) < 0.9
            # and (drone_vel_in_gate_frame[0])>0.5)
        # 
        # return passed
            
        
        
    def gate_corners(self, drone_pos: np.ndarray,drone_or:np.ndarray):
        drone_att=pyquaternion.Quaternion(w=drone_or[3],x=drone_or[0], y=drone_or[1], z=drone_or[2])
        tl_corner=drone_att.inverse.rotate(self.tl_corner-drone_pos)
        tr_corner=drone_att.inverse.rotate(self.tr_corner-drone_pos)
        bl_corner=drone_att.inverse.rotate(self.bl_corner-drone_pos)
        br_corner=drone_att.inverse.rotate(self.br_corner-drone_pos)

        return [tl_corner[0],tl_corner[1],tl_corner[2],
                tr_corner[0],tr_corner[1],tr_corner[2],
                bl_corner[0],bl_corner[1],bl_corner[2],
                br_corner[0],br_corner[1],br_corner[2]
                ]
    def dev_angle(self,drone_or:np.ndarray):
        drone_att=pyquaternion.Quaternion(w=drone_or[3],x=drone_or[0], y=drone_or[1], z=drone_or[2])
        
        or_error=drone_att.inverse*self.att
        #rotate the error quaternion around z by 180 degrees
        or_error=pyquaternion.Quaternion(axis=[0,0,1],angle=np.pi)*or_error 
        er_angle=or_error.angle
        # yaw,pitch,roll =drone_att.yaw_pitch_roll
        return er_angle


       

    def __repr__(self):
        return "Gate at [%.2f, %.2f, %.2f] with yaw %.2f deg." % (
            self.pos[0],
            self.pos[1],
            self.pos[2],
            180.0 / np.pi * self.yaw,
        )

    def draw(self, ax):
        # helper corners for drawing
        tl_outer = self.pos + self.att.rotate(
            np.array([0.0, 1.0, 1.0]) * self.gate_dim / 2.0
            + np.array([0.0, 1.0, 1.0]) * self.w_border
        )
        tr_outer = self.pos + self.att.rotate(
            np.array([0.0, -1.0, 1.0]) * self.gate_dim / 2.0
            + np.array([0.0, -1.0, 1.0]) * self.w_border
        )
        tl_lower = self.pos + self.att.rotate(
            np.array([0.0, 1.0, 1.0]) * self.gate_dim / 2.0
            + np.array([0.0, 1.0, 0.0]) * self.w_border
        )
        tr_lower = self.pos + self.att.rotate(
            np.array([0.0, -1.0, 1.0]) * self.gate_dim / 2.0
            + np.array([0.0, -1.0, 0.0]) * self.w_border
        )
        bl_outer = self.pos + self.att.rotate(
            np.array([0.0, 1.0, -1.0]) * self.gate_dim / 2.0
            + np.array([0.0, 1.0, -1.0]) * self.w_border
        )
        br_outer = self.pos + self.att.rotate(
            np.array([0.0, -1.0, -1.0]) * self.gate_dim / 2.0
            + np.array([0.0, -1.0, -1.0]) * self.w_border
        )
        bl_upper = self.pos + self.att.rotate(
            np.array([0.0, 1.0, -1.0]) * self.gate_dim / 2.0
            + np.array([0.0, 1.0, 0.0]) * self.w_border
        )
        br_upper = self.pos + self.att.rotate(
            np.array([0.0, -1.0, -1.0]) * self.gate_dim / 2.0
            + np.array([0.0, -1.0, 0.0]) * self.w_border
        )

        x = [tl_outer[0], tr_outer[0], tr_lower[0], tl_lower[0]]
        y = [tl_outer[1], tr_outer[1], tr_lower[1], tl_lower[1]]
        z = [tl_outer[2], tr_outer[2], tr_lower[2], tl_lower[2]]
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        poly.set_color(
            np.zeros(
                3,
            )
        )
        ax.add_collection3d(poly)
        x = [tl_lower[0], self.tl_corner[0], self.bl_corner[0], bl_upper[0]]
        y = [tl_lower[1], self.tl_corner[1], self.bl_corner[1], bl_upper[1]]
        z = [tl_lower[2], self.tl_corner[2], self.bl_corner[2], bl_upper[2]]
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        poly.set_color(
            np.zeros(
                3,
            )
        )
        ax.add_collection3d(poly)
        x = [self.tr_corner[0], tr_lower[0], br_upper[0], self.br_corner[0]]
        y = [self.tr_corner[1], tr_lower[1], br_upper[1], self.br_corner[1]]
        z = [self.tr_corner[2], tr_lower[2], br_upper[2], self.br_corner[2]]
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        poly.set_color(
            np.zeros(
                3,
            )
        )
        ax.add_collection3d(poly)
        x = [bl_upper[0], br_upper[0], br_outer[0], bl_outer[0]]
        y = [bl_upper[1], br_upper[1], br_outer[1], bl_outer[1]]
        z = [bl_upper[2], br_upper[2], br_outer[2], bl_outer[2]]
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        poly.set_color(
            np.zeros(
                3,
            )
        )
        ax.add_collection3d(poly)
