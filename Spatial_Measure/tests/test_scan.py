import cv2
import numpy as np

from spatial_measure import SpatialDimensionEstimator
from spatial_measure.scan import PlaneTracker, ScanMap
rng = np.random.default_rng(1)
P=4  # px per mm in world image
W,Hh=720,320  # mm
world=(rng.normal(150,30,(Hh*P//8,W*P//8)).clip(0,255)).astype(np.uint8)
world=cv2.resize(world,(W*P,Hh*P),interpolation=cv2.INTER_CUBIC)
for _ in range(500):
    cv2.circle(world,(int(rng.integers(0,W*P)),int(rng.integers(0,Hh*P))),int(rng.integers(4,20)),int(rng.integers(60,230)),-1)
world=cv2.cvtColor(world,cv2.COLOR_GRAY2BGR)
m=cv2.imread('targets/aruco_id0_50mm.png',0)
e0=SpatialDimensionEstimator(marker_length_mm=50)
t=np.full((1000,1000),255,np.uint8); t[100:900,100:900]=cv2.resize(m,(800,800),interpolation=cv2.INTER_NEAREST)
c,_=e0.detect_marker(t); side=np.linalg.norm(c[0]-c[1]); sc=50*P/side  # scale so black marker =50mm
mk=cv2.resize(t,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
# place marker black-square top-left at world mm (20,20)
x0=int(20*P-c[0][0]*sc); y0=int(50*P-c[0][1]*sc)
sub=cv2.cvtColor(mk,cv2.COLOR_GRAY2BGR); h,w=sub.shape[:2]
xa,ya=max(x0,0),max(y0,0); world[ya:y0+h,xa:x0+w]=sub[ya-y0:,xa-x0:][:world.shape[0]-ya,:world.shape[1]-xa]
# board 400x90mm at world (110,110)
cv2.rectangle(world,(110*P,110*P),(510*P,190*P),(60,110,200),-1)
cv2.putText(world,'LONG BOARD',(200*P,160*P),cv2.FONT_HERSHEY_SIMPLEX,2.5,(255,255,255),6)
FW,FH,S=640,480,3.2
def view(cx):  # camera centred over world mm (cx,120)
    A=np.float64([[S/P,0,-(cx*S)+FW/2-0],[0,S/P,-(120*S)+FH/2],[0,0,1]])
    A[0,2]=-(cx*S)+FW/2; 
    tilt=np.array([[1,0.0,0],[0.02*np.sin(cx/90),1,0],[0.00002*np.sin(cx/70),0,1]])
    Hf=tilt@A
    return cv2.warpPerspective(world,Hf,(FW,FH),flags=cv2.INTER_LINEAR,borderValue=(120,120,120)),Hf


def test_object_larger_than_the_frame_is_measured_by_panning():
    est = SpatialDimensionEstimator(marker_length_mm=50)
    tr, sm = PlaneTracker(50), ScanMap()
    img, Hf = view(100)
    tap = tuple(map(float, cv2.perspectiveTransform(np.float32([[[150 * P, 150 * P]]]), Hf)[0, 0]))
    seed, first, res = None, True, None
    for cx in np.arange(100, 440, 8):
        img, _ = view(cx)
        r = est.process(img, seed_px=tap if first else None, seed_mm=seed, tracker=tr,
                        extra_seeds_mm=sm.seed_candidates() if seed else None, validate=sm.consistent)
        first = False
        seed = r.seed_mm or seed
        if r.selected is not None and r.scan_data:
            sm.add(r.scan_data["cnt"], r.scan_data["valid"], r.scan_data["ppm"], r.scan_data["origin"])
        res = sm.measure(seed)
    assert tr.frames_since_marker > 20  # most of the pan happened without the marker in view
    # the board is 400 x 80 mm, twice the width of one camera frame (200 mm)
    assert abs(res["length_mm"] - 400) < 12 and abs(res["width_mm"] - 80) < 6
    assert not res["open"]
