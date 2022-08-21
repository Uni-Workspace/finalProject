from cv2 import EVENT_LBUTTONDOWN, EVENT_RBUTTONDOWN
def click_event(event, x, y, flag, param):
    if event == EVENT_LBUTTONDOWN:
        print(x,y,flag,param)
        return (x,y)