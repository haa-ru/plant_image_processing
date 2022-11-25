import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import os
import natsort
import matplotlib.patches as patches



### 색상추출해서 binary mask로 반환 / (string, array, array) -> ndarray
def conv_binary_mask(file_path, low, upp):
    image_bgr = cv2.imread(file_path) # 이미지 읽기
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV) # BGR에서 HSV로 변환 

    ## HSV에서 색의 값 범위 정의 / hue(색상), saturation(채도), value(명도)
    lower = np.array(low)  # green [25,40,40],[85,255,255]
    upper = np.array(upp) 

    mask = cv2.inRange(image_hsv, lower, upper) # 마스크를 만듬 
    np.place(mask,mask!=0, 1) # img 배열을 0,1로 변환
    return mask



### ndarray에서 특정값 전부 제거하는 함수 / (ndarray) -> ndarray
def delet(ndarray, a):
    delete = []
    for i in range(len(ndarray)):
        if ndarray[i] == a:
            delete.append(i)
    return np.delete(ndarray, delete)



### 노이즈제거(영역분할후 특정 픽셀개수 밑의 영역 삭제) / (ndarray,int) -> ndarray
def noise(mask, n): 
    (arr_num, mask) = cv2.connectedComponents(mask) # 그루핑 해서 바뀐 배열값 반환 tuple(배열개수,nbarray) 에서 array값만.

    mask_1D = mask.reshape(-1,) # 1D로 변환, delet함수 사용하기 위해서
    delet_mask_1D = delet(mask_1D, 0) # 0은 배경이라서 제거함

    for i in range(1,arr_num+1): # 1부터 배열개수까지 반복
        if list(delet_mask_1D).count(i) < n: # n보다 작은요소는 0으로 변환
            mask = np.where(mask==i, 0, mask)

    return mask



### img_conv_save_binary (폴더자동생성X)
def img_conv_save(conv_path, save_path):
    fileList = os.listdir(conv_path)
    fileList = natsort.natsorted(fileList) # 파일 이름 정렬
    for i in fileList:
        if i == '.DS_Store': # 메타데이터 파일 무시
            continue
        mask_img = conv_binary_mask(conv_path+'/'+i)
        output_img = noise(mask_img)

        # 바이너리 이미지 저장(imwrite함수가 디폴트로는 그레이스케일로 저장되서..)
        cv2.imwrite(save_path+'/'+i, output_img) 
        cv2.imwrite(save_path+'/'+i ,output_img, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(i+'Saved!')



### cut_img_save (폴더자동생성X)
def cut_img_save(conv_path, save_path):
    fileList = os.listdir(conv_path)
    for i in fileList:
        if i == '.DS_Store': 
            continue
        cut_img_list = cut(conv_path+'/'+i)
        print('pathname: ' + conv_path+'/'+i)
        n = 1
        for img in cut_img_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path+'/'+'['+str(n)+']'+i, img) 
            n = n+1




### 이미지 흑백 저장
def img_to_grauScale(trainpath,savepath):
    for i in os.listdir(trainpath):
        img = cv2.imread(trainpath+'/'+i, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(savepath+'/'+i ,img)
        print(img+' saved')
    return print('success')



### box값만 리스트로 출력 / ndarray -> list
def box(mask):
    box_li = []
    for i in np.unique(mask):
        if i != 0:
            down = np.max(np.where(mask==i)[0])
            up = np.min(np.where(mask==i)[0])
            left = np.min(np.where(mask==i)[1])
            right = np.max(np.where(mask==i)[1])
            box_li.append([left,up,right,down])
    return box_li



### 이미지 박싱 /(ndarray, int, string) -> rectangle 배열로 출력 / Rectangle(xy=(6, 95), width=69, height=-74, angle=0)
def boxing(mask, padding, color):  # boxing(mask, 16, 'black')
    rect_li = []
    box_li = box(mask)
    for i in box_li:
        left,up,right,down = i
        rect = patches.Rectangle((left-padding,down+padding),
                                (right-left)+(padding*2),
                                (up-down)-(padding*2),
                                linewidth=2,
                                edgecolor=color,
                                fill = False)
        rect_li.append(rect)
    return rect_li


### 이미지 분리 / str -> list
def cut(filepath):
    img = cv2.imread(filepath)

    ## green + rad mask
    mask1 = conv_binary_mask(filepath, [20,40,40],[85,255,255]) # green
    mask2 = conv_binary_mask(filepath, [128,40,40],[255,255,255]) # rad
    mask = cv2.add(mask1,mask2)
    mask = noise(mask, 70)
    boxes = box(mask)
    li = []
    p = 10 # padding
    for i in boxes:
        left, up, rigth, down = i
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cropd = img_rgb[up-p: down+p, left-p: rigth+p].copy()
        li.append(img_cropd)
    return li