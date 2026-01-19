import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

airfoil_db_path = '/Users/ryanblanchard/myApplications/Xfoil/airfoils/'

from typing import List, Callable
import numpy as np
import pathlib


class WingSection:
    """
    Represents a spanwise wing section.
    """
    def __init__(self,
                 airfoil_file: str = "dae11.dat",
                 span_frac: float = 0.0,
                 chord_frac: float = 1.0,
                 twist_deg: float = 0.0,
                 offset: tuple = (0.0, 0.0)):
        """
        Parameters
        ----------
        airfoil_file : str
            Path to airfoil .dat file (default 'dae11.dat').
        span_frac : float
            Normalized position along half span [0 = root, 1 = tip].
        chord_frac : float
            Local chord length as a fraction of root chord.
        twist_deg : float
            Twist angle (AoA offset relative to root) in degrees.
        offset : tuple
            Local (x,z) offset for positioning section in plane.
        """
        #self.airfoil_file = pathlib.Path(airfoil_db_path + airfoil_file)
        self.airfoil_file = airfoil_file
        self.span_frac = span_frac
        self.chord_frac = chord_frac
        self.twist_deg = twist_deg
        self.offset = offset
        #self.airfoil_coords = self._load_airfoil()
        label, xy_ptm, self.suction_pts, self.pressure_pts = read_airfoil_xy_pts(airfoil_name)
        self.suction_xyz_coords = None
        self.pressure_xyz_coords = None 
        self.fullsection_xyz_coords = None

    def _load_airfoil(self) -> np.ndarray:
        """Load airfoil coordinates (x,y) from file."""
        coords = []
        if self.airfoil_file.exists():
            with open(self.airfoil_file, "r") as f:
                for line in f:
                    try:
                        x, y = map(float, line.strip().split())
                        coords.append((x, y))
                    except ValueError:
                        continue  # skip header lines
        else:
            raise FileNotFoundError(f"Airfoil file {self.airfoil_file} not found")
        return np.array(coords)

    def __repr__(self):
        return (f"WingSection(span_frac={self.span_frac}, "
                f"chord_frac={self.chord_frac}, twist={self.twist_deg}°, "
                f"airfoil='{self.airfoil_file.name}')")


class Wing:
    """
    Represents an overall wing geometry.
    """
    def __init__(self,
                 span: float,
                 root_chord: float,
                 interpolation: str = "linear"):
        """
        Parameters
        ----------
        span : float
            Total wingspan (tip-to-tip).
        root_chord : float
            Chord length at wing root.
        interpolation : str
            Interpolation scheme for geometry ('linear', 'spline', etc).
        """
        self.span = span
        self.root_chord = root_chord
        self.interpolation = interpolation
        self.half_area = None
        self.full_area = None
        self.aspect_ratio = None
        self.sections: List[WingSection] = []

    def add_section(self, section: WingSection):
        """Attach a WingSection to this wing."""
        self.sections.append(section)
        self.sections.sort(key=lambda s: s.span_frac)
        suction_positioned, pressure_positioned = offset_scale_and_rotate(section.offset, section.suction_pts, section.pressure_pts,section.chord_frac,section.twist_deg)
        y_val = 0.5*self.span*section.span_frac
        
        p_xvals = pressure_positioned[:,0]
        s_xvals = suction_positioned[:,0]
        p_yvals = y_val*np.ones_like(p_xvals)
        s_yvals = y_val*np.ones_like(s_xvals)
        p_zvals = pressure_positioned[:,1]
        s_zvals = suction_positioned[:,1]
        section.pressure_xyz_coords = np.array([p_xvals, p_yvals, p_zvals]).T
        section.suction_xyz_coords = np.array([s_xvals, s_yvals, s_zvals]).T
        section.fullsection_xyz_coords = np.row_stack([section.suction_xyz_coords, np.flipud(section.pressure_xyz_coords)])
        self.update_wing_area_properties()

    def get_allpresssuct2d(self):
        allpress, allsuct = None, None
        n_sections = len(self.sections)
        if n_sections > 0:
            allpress = np.array([self.sections[0].pressure_xyz_coords[:,0],self.sections[0].pressure_xyz_coords[:,2]]).T
            allsuct  = np.array([self.sections[0].suction_xyz_coords[:,0],self.sections[0].suction_xyz_coords[:,2]]).T
        if n_sections > 1:
            for i in range(1,n_sections):
                allpressi = np.array([self.sections[i].pressure_xyz_coords[:,0],self.sections[i].pressure_xyz_coords[:,2]]).T
                allsucti  = np.array([self.sections[i].suction_xyz_coords[:,0],self.sections[i].suction_xyz_coords[:,2]]).T
                allpress = np.row_stack([allpress,allpressi])
                allsuct = np.row_stack([allsuct,allsucti])
        allpress = allpress[allpress[:,0].argsort()]
        allsuct = allsuct[allsuct[:,0].argsort()]
        return allpress, allsuct
        
    def update_wing_area_properties(self):
        sections = self.get_chord_distribution()
        n_sections = len(sections)
        if n_sections ==1:
            chord = sections[0][1]
            self.full_area = chord*self.span
            self.half_area = 0.5*self.full_area
            self.aspect_ratio = self.full_area/self.span
        
        self.half_area = 0
        if n_sections > 1:
            for i in range(1,n_sections):
                chord_0 = sections[i-1][1]
                chord_1 = sections[i][1]
                span_0 = sections[i-1][0]
                span_1 = sections[i][0]
                dspan = span_1 - span_0
                self.half_area += 0.5*dspan*(chord_0+chord_1)
            self.full_area = 2*self.half_area
            self.aspect_ratio = self.full_area/self.span
        #print(f"Full Area: {self.full_area} Half Area: {self.half_area} Aspect Ratio {self.aspect_ratio}")


    def get_chord_distribution(self) -> List[tuple]:
        """Return list of (y, chord) along half-span."""
        return [(s.span_frac * self.span / 2,
                 s.chord_frac * self.root_chord)
                for s in self.sections]

    def __repr__(self):
        return (f"Wing(span={self.span}, root_chord={self.root_chord}, "
                f"sections={len(self.sections)}, interpolation='{self.interpolation}')")
    
    def plot_planview(self):
        planview = None
        leading_edge = []
        trailing_edge = []
        for i, section in enumerate(self.sections):
            y = section.suction_xyz_coords[0,1]
            x_min = -1*np.min(section.fullsection_xyz_coords[:,0])
            x_max = -1*np.max(section.fullsection_xyz_coords[:,0])
            plt.plot([y,y],[x_min, x_max],label=f'Section {i}')
            leading_edge.append([x_min,y])
            trailing_edge.append([x_max,y])
        leading_edge = np.array(leading_edge)
        trailing_edge = np.array(trailing_edge)

        plt.plot(leading_edge[:,1],leading_edge[:,0],'k')
        plt.plot(trailing_edge[:,1],trailing_edge[:,0],'k')
        plt.legend()
        plt.show()
        

    def plot_spanview(self):
        planview = None
        leading_edge = []
        trailing_edge = []
        for i, section in enumerate(self.sections):
            plt.plot(section.fullsection_xyz_coords[:,0],section.fullsection_xyz_coords[:,2],label=f'Section {i}')
        plt.legend()  
        plt.show()          
            

def find_max_thickness_point(suction_pts,pressure_pts):
    n_suction_pts = suction_pts.shape[0]
    camber_line = np.zeros_like(suction_pts)
    thickness_profile = np.zeros_like(suction_pts)
    distance_profile = np.zeros(n_suction_pts)
    max_thickness = 0
    max_thickness_point = [0,0]
    max_distance = 0
    max_dist_point = [0,0]
    
    for i in range(0,n_suction_pts):
        x0, y1 = suction_pts[i,0], suction_pts[i,1]
        y0 = np.interp(x0,pressure_pts[:,0],pressure_pts[:,1])
        thickness = y1-y0
        midpoint = y0 + 0.5 * thickness
        camber_line[i,:] = [x0, midpoint]
        thickness_profile[i,:] = [x0,thickness]
        if thickness > max_thickness:
            max_thickness = thickness
            max_thickness_point = [x0,midpoint]


    for i in range(0,n_suction_pts):
        pt0 = [camber_line[i,0], camber_line[i,1]]
        suction_dist = np.min(np.sum((suction_pts-pt0)**2, axis=1))
        pressure_dist = np.min(np.sum((pressure_pts-pt0)**2, axis=1))
        min_dist = np.sqrt(np.min([suction_dist,pressure_dist]))
        distance_profile[i] = min_dist
        if min_dist > max_distance:
            max_distance = min_dist
            max_dist_point = pt0
            print(i, max_distance)

    thetas = np.linspace(0,2*np.pi,100)
    
    reduced_camber_for_circle_plots = camber_line[0:-1:5,:]
    reduced_distances = distance_profile[0:-1:5]



    plt.plot(suction_pts[:,0], suction_pts[:,1])
    plt.plot(pressure_pts[:,0], pressure_pts[:,1])
    plt.plot(camber_line[:,0], camber_line[:,1])
    plt.plot(thickness_profile[:,0], thickness_profile[:,1])
    plt.plot(max_thickness_point[0],max_thickness_point[1],'o',label='max thick')
    plt.plot(max_dist_point[0],max_dist_point[1],'o',label='max dist')
    for i, ctr_pt in enumerate(reduced_camber_for_circle_plots):
        max_dist_circle = np.zeros([100,2])
        for j, theta in enumerate(thetas): 
            max_dist_circle[j,:] = [ctr_pt[0]+reduced_distances[i]*np.cos(theta),ctr_pt[1]+reduced_distances[i]*np.sin(theta)]
        plt.plot(max_dist_circle[:,0],max_dist_circle[:,1])
    plt.legend()
    plt.show()
    return 0


def largest_inscribed_circle(suction_pts, pressure_pts, plot=True):
    """
    Find the largest circle that fits between suction and pressure curves.
    
    Parameters
    ----------
    suction_pts : ndarray [N,2]
        Points defining suction surface.
    pressure_pts : ndarray [M,2]
        Points defining pressure surface.
    plot : bool
        If True, plots the curves and the fitted circle.
    
    Returns
    -------
    center : ndarray [2]
        Circle center coordinates.
    radius : float
        Circle radius.
    """
    # KD-tree for nearest-neighbor search
    tree_suction = cKDTree(suction_pts)
    tree_pressure = cKDTree(pressure_pts)

    candidates = []

    # For each suction point, find nearest pressure point
    dists, idxs = tree_pressure.query(suction_pts)
    for i, (d, j) in enumerate(zip(dists, idxs)):
        center = 0.5 * (suction_pts[i] + pressure_pts[j])
        radius = 0.5 * d
        candidates.append((radius, center))

    # For each pressure point, find nearest suction point
    dists, idxs = tree_suction.query(pressure_pts)
    for i, (d, j) in enumerate(zip(dists, idxs)):
        center = 0.5 * (pressure_pts[i] + suction_pts[j])
        radius = 0.5 * d
        candidates.append((radius, center))

    # Pick the largest
    radius, center = max(candidates, key=lambda x: x[0])

    if plot:
        fig, ax = plt.subplots()
        ax.plot(suction_pts[:,0], suction_pts[:,1], 'r-', label="Suction surface")
        ax.plot(pressure_pts[:,0], pressure_pts[:,1], 'b-', label="Pressure surface")
        circ = plt.Circle(center, radius, color='g', fill=False, lw=2, label="Max inscribed circle")
        ax.add_patch(circ)
        ax.plot(center[0], center[1], 'go')  # circle center
        ax.set_aspect('equal')
        ax.legend()
        plt.show()

    return center, radius

        



def read_airfoil_xy_pts (airfoil_name):
    airfoil_file_full_path = airfoil_db_path +airfoil_name
    try:
        with open(airfoil_file_full_path, "r") as f:
            lines = f.readlines()
        num_pts = len(lines)-1
        xy_array = np.zeros([num_pts,2])
        
        for i, line in enumerate(lines):
            if i == 0:
                label = line
            else:
                data_row = line.split()
                x_loc = float(data_row[0])
                y_loc = float(data_row[1])
                xy_array[i-1,:] = [x_loc,y_loc]

        split_idx = np.argmin(xy_array[:,0])
        suction_pts = xy_array[0:split_idx+1,:]
        pressure_pts = np.flipud(xy_array[split_idx:-1,:])


        #plt.plot(suction_pts[:,0], suction_pts[:,1])
        #plt.plot(pressure_pts[:,0], pressure_pts[:,1])
        #plt.show()
        
        return label, xy_array, suction_pts, pressure_pts

    except ValueError:
        return None

def offset_scale_and_rotate(spec_offset, suction_pts,pressure_pts,chord,alpha_deg, plot = False):
    default_offset = -0.25*suction_pts[0,:]
    spec_offset = np.array([spec_offset[0], spec_offset[1]])
    net_offset = default_offset+spec_offset
    suction_pts = suction_pts + net_offset
    pressure_pts = pressure_pts + net_offset
    suction_pts = chord * suction_pts
    pressure_pts = chord * pressure_pts
    alpha_rad = np.deg2rad(-1.0*alpha_deg)
    rotation_matrix = [[],[]]
    rotation_matrix = [[np.cos(alpha_rad), -1.0*np.sin(alpha_rad)],[np.sin(alpha_rad), np.cos(alpha_rad)]]
    #suction_pts = np.matmul(rotation_matrix,suction_pts.T).T
    #pressure_pts = np.matmul(rotation_matrix,pressure_pts.T).T
    suction_pts = (rotation_matrix @ suction_pts.T).T
    pressure_pts = (rotation_matrix @ pressure_pts.T).T
    if plot == True:
        plt.plot(suction_pts[:,0], suction_pts[:,1])
        plt.plot(pressure_pts[:,0], pressure_pts[:,1])
        plt.show()
    return suction_pts, pressure_pts 


if __name__ == "__main__":
    wingspan = 17
    rootchord = 1 
    halfspan_ratios = [0, .3,  .6, .8, 1.0]
    chord_ratios =    [1,  1 , .9, .8, .7] 
    twist_angles =    [6,  4,   3,  2,  1]
    offsets = [(0,0),(0,0),(0,0),(-.05,.05),(-.1,.05)]
    airfoil_names = ['dae11.dat','dae21.dat','dae31.dat','dae41.dat','dae51.dat']
    mainwing = Wing(wingspan,rootchord, "linear")
    
    for i, airfoil_name in enumerate(airfoil_names):
        sectioni = WingSection(airfoil_name, halfspan_ratios[i], chord_ratios[i],twist_angles[i],offsets[i])
        mainwing.add_section(sectioni)

    all_pressure2d, all_suction2d = mainwing.get_allpresssuct2d()
    find_max_thickness_point(all_suction2d,all_pressure2d)
    #center, radius = largest_inscribed_circle(all_suction2d,all_pressure2d, True)

    mainwing.plot_spanview()
    mainwing.plot_planview()
    print(mainwing.get_chord_distribution())
    
    


