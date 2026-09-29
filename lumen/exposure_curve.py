"""A direct view/controller of exposure and the four existing tonal sliders."""
import copy
import numpy as np
from PySide6.QtCore import Qt,Signal,QPointF,QRectF
from PySide6.QtGui import QPainter,QPainterPath,QPen,QColor
from PySide6.QtWidgets import QWidget

KEYS=('blacks','shadows','highlights','whites')
CENTERS=np.array([.08,.32,.65,.92])

def linear(x):
    x=np.asarray(x,dtype=float)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def srgb(x):
    return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)

def basis(x,a):
    p=np.clip(linear(x)*2**a['exposure'],0,1)**.45
    return np.stack([(1-p)**6/85,(1-p)**2/65,p**3/65,p**6/85],axis=-1)

def evaluate(a,x):
    value=linear(x)*2**a['exposure']*2**(basis(x,a)@np.array([a[k] for k in KEYS]))
    return np.clip((srgb(value)-.5)*(1+a['contrast']/125)+.5,0,1)

def drag(a,x,target,global_exposure=False):
    """Solve a bounded, tone-local slider update. No second curve is applied."""
    result=copy.deepcopy(a);x=float(np.clip(x,1/255,1));target=float(np.clip(target,0,1))
    if global_exposure:
        # Sample the full legal range, then refine the closest bracket. This also
        # handles extreme four-zone settings whose response is not monotonic.
        grid=np.linspace(-5,5,201);values=[]
        for value in grid:
            result['exposure']=value;values.append(abs(float(evaluate(result,x))-target))
        i=int(np.argmin(values));lo=grid[max(0,i-1)];hi=grid[min(len(grid)-1,i+1)]
        for _ in range(18):
            first=lo+(hi-lo)/3;second=hi-(hi-lo)/3
            result['exposure']=first;e1=abs(float(evaluate(result,x))-target)
            result['exposure']=second;e2=abs(float(evaluate(result,x))-target)
            if e1<=e2:hi=second
            else:lo=first
        result['exposure']=float(np.clip(round((lo+hi)/2,2),-5,5));return result
    weights=basis(x,a);initial=np.array([a[k] for k in KEYS],float)
    uncontrasted=np.clip((target-.5)/(1+a['contrast']/125)+.5,1e-6,1)
    desired=np.log2(max(float(linear(uncontrasted)),1e-9)/max(float(linear(x))*2**a['exposure'],1e-9))
    proximity=np.exp(-.5*((x-CENTERS)/.18)**2)
    values=initial.copy();free=np.ones(4,bool)
    for _ in range(5):
        remaining=desired-float(weights@values)
        denom=np.sum(weights[free]**2*proximity[free])
        if denom<1e-14:break
        proposal=values.copy();proposal[free]+=remaining*weights[free]*proximity[free]/denom
        bounded=np.clip(proposal,-100,100)
        saturated=(np.abs(proposal)>100)&free;values=bounded
        if not saturated.any():break
        free[saturated]=False
    for key,value in zip(KEYS,values):result[key]=round(float(value),2)
    return result


class ExposureCurve(QWidget):
    changed=Signal(dict)
    committed=Signal()

    def __init__(self,parent=None):
        super().__init__(parent)
        self.values=dict(exposure=0.,contrast=0.,**{k:0. for k in KEYS});self.drag_state=None
        self.setFixedHeight(165)
        self.setMouseTracking(True)
        self.setToolTip('任意亮度位置上下拖动，联动黑色／暗部／亮部／白色；按住 Shift 拖动调整整体曝光。双击重置曝光和四段。')

    def area(self):return QRectF(18,12,self.width()-36,self.height()-42)

    def set_values(self,values):
        self.values=copy.deepcopy(values);self.update()

    def screen(self,x,y):
        a=self.area();return QPointF(a.left()+x*a.width(),a.bottom()-y*a.height())

    def mousePressEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton or not self.area().contains(event.position()):return
        x=float(np.clip((event.position().x()-self.area().left())/self.area().width(),1/255,1))
        self.drag_state=(copy.deepcopy(self.values),x,event.position().y(),float(evaluate(self.values,x)),bool(event.modifiers()&Qt.KeyboardModifier.ShiftModifier))

    def mouseMoveEvent(self,event):
        if self.drag_state is None:return
        values,x,start_y,start_value,global_exposure=self.drag_state
        target=start_value+(start_y-event.position().y())/self.area().height()
        self.values=drag(values,x,target,global_exposure)
        self.changed.emit(copy.deepcopy(self.values));self.update()

    def mouseReleaseEvent(self,event):
        if self.drag_state is not None:self.drag_state=None;self.committed.emit()

    def mouseDoubleClickEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton:return
        self.drag_state=None
        for key in (*KEYS,'exposure'):self.values[key]=0.
        self.changed.emit(copy.deepcopy(self.values));self.committed.emit();self.update()

    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.Antialiasing);a=self.area()
        painter.fillRect(a,QColor('#161c17'));painter.setPen(QPen(QColor('#354033'),1))
        for i in range(5):
            t=i/4;painter.drawLine(self.screen(t,0),self.screen(t,1));painter.drawLine(self.screen(0,t),self.screen(1,t))
        painter.setPen(QPen(QColor('#596552'),1,Qt.PenStyle.DashLine));painter.drawLine(a.bottomLeft(),a.topRight())
        axis=np.linspace(0,1,512);values=evaluate(self.values,axis);path=QPainterPath(self.screen(0,float(values[0])))
        for x,y in zip(axis,values):path.lineTo(self.screen(x,y))
        painter.setPen(QPen(QColor('#c6d9b5'),2));painter.drawPath(path)
        painter.setBrush(QColor('#202c1c'))
        for x in CENTERS:painter.drawEllipse(self.screen(x,float(evaluate(self.values,x))),3,3)
        painter.setPen(QColor('#94a68c'))
        for i,text in enumerate(('黑色','暗部','亮部','白色')):
            rect=QRectF(a.left()+i*a.width()/4,a.bottom()+6,a.width()/4,22)
            painter.drawText(rect,Qt.AlignmentFlag.AlignCenter,text)
