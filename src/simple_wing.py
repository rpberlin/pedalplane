from xfoil_wrapper import run_xfoil
from pint import Quantity as Q_
#airfoil = 'eppler387.dat'
airfoil = 'dae11.dat'
rho = Q_(1.22, 'kg/m^3')
mu  = Q_(1.81e-5, "pascal*second")  #

span   = Q_(9.0, "meter")
chord  = Q_(1.5, "meter")
aoa    = Q_(10.0, "degree")
V_inf  = Q_(36.0, "kilometer / hour")  # 35 kph
KE = 0.5 * rho * (V_inf**2)

A_wing = span*chord

Re = (rho*V_inf*chord/mu).to_base_units()

result = run_xfoil(
    str(airfoil),
    Re=Re.magnitude,
    AoA=aoa.to("degree").magnitude,
    Ncrit=9,
    Mach=0.0,
    max_iter=200,
    debug=False,
)

Cl = float(result.get('Cl') or 0.0)
Cd = float(result.get('Cd') or 0.0)
Cm = float(result.get('Cm') or 0.0)
Lift = (KE * A_wing * Cl).to('lbf') 
Drag = (KE * A_wing * Cd).to('lbf')
Power = (Drag* V_inf).to('W')
print(Lift,Drag,Power, Re)