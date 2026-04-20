import math
import xml.etree.ElementTree as ET

THETA_UPPER = math.atan2(-0.30,  0.20)
THETA_LOWER = math.atan2(-0.40, -0.35)

#Проверка а можно ли вообще такую модель сделать
def check_reachable(x_attach, L1, L2):
    lo, hi = abs(L1 - L2), L1 + L2
    if not (lo <= x_attach <= hi):
        raise ValueError("Что-то пошло не так")


def build_xml(src="model.xml", x_attach=0.15, L1=0.3, L2=0.4):

    check_reachable(x_attach, L1, L2)

    ux, uz = L1 * math.cos(THETA_UPPER), L1 * math.sin(THETA_UPPER)
    lx, lz = L2 * math.cos(THETA_LOWER), L2 * math.sin(THETA_LOWER)

    tree = ET.parse(src)
    root = tree.getroot()

    def update(name, pos=None, fromto=None):
        body = root.find(f".//body[@name='{name}']")
        if pos is not None:
            body.set("pos", f"{pos[0]:.4f} 0 {pos[1]:.4f}")
        if fromto is not None:
            body.find("geom").set(
                "fromto", f"0 0 0  {fromto[0]:.4f} 0 {fromto[1]:.4f}")

    #Правая нога
    update("right_hip",  pos=( x_attach, 0), fromto=( ux,  uz))
    update("right_knee", pos=( ux, uz),      fromto=( lx,  lz))
    update("right_foot", pos=( lx, lz))

    #Левая нога
    update("left_hip",   pos=(-x_attach, 0), fromto=(-ux,  uz))
    update("left_knee",  pos=(-ux, uz),      fromto=(-lx,  lz))
    update("left_foot",  pos=(-lx, lz))

    return ET.tostring(root, encoding="unicode")


def build_model(src="model.xml", **params):
    import mujoco
    return mujoco.MjModel.from_xml_string(build_xml(src, **params))


if __name__ == "__main__":
    print(build_xml(x_attach=0.12, L1=0.3, L2=0.4))
