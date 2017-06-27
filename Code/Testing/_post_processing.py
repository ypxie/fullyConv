import cv2
import scipy
from   skimage.feature import peak_local_max
from   skimage import data

from numba import jit, autojit
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage import io
from local_utils import *
import numpy as np
from skimage import measure
from skimage.filters import frangi

from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max

@autojit
def touchboundary(bbox, shape, tol=-1):
    ''' 
    bbox: (min_row, min_col, max_row, max_col)
    shape: (row, col)
    '''    
    min_row, min_col, max_row, max_col = bbox
    return min_row <= tol or min_col <=tol or max_row>=shape[0]-tol or max_col >= shape[1]-tol


def overal_watershed_marker(probmap,marker, thresh_water = 0.6, thresh_seg = 0.7,ratio = 0.2, dist_f = np.median):
    
    npad2 = ((1,1),(1,1))
    probmap = np.pad(probmap, npad2, mode='constant', constant_values=0)
    marker = np.pad(marker, npad2, mode='constant', constant_values=0)
    
    ret, fg_mask = cv2.threshold(probmap.astype(np.float32),thresh_seg,255,0)
    #ret, seed_mask = cv2.threshold(probmap.astype(np.float32),0.7,255,0)
    
    
    marker[fg_mask == 0] = 0
    
    ret, markers = cv2.connectedComponents(marker)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    fg_mask = cv2.dilate(fg_mask,se)
    
    fg_mask_color = cv2.cvtColor(fg_mask.astype('uint8'),cv2.COLOR_GRAY2RGB)
    #imshow(fg_mask_color)
    markers = cv2.watershed(fg_mask_color, markers)

    remove_boarder_marker = markers[1:-1, 1:-1]
    return remove_boarder_marker

@autojit
def make_img(prob, thresh_hold = 0.5):
    thresh = prob > thresh_hold
    thresh = scipy.ndimage.binary_fill_holes(thresh)
    thresh = (thresh*255).astype(np.uint8)
    cv3Thresh = cv2.cvtColor(thresh.astype('uint8'),cv2.COLOR_GRAY2RGB)    
    return cv3Thresh, thresh

@autojit
def get_thresh(dist_transform):
    return np.max(dist_transform)
#     dist_transform = scipy.ndimage.filters.gaussian_filter(dist_transform, sigma = 2)
#     #ret, sure_fg = cv2.threshold(dist_transform,ratio*dist_f(dist_transform),255,0)
#     coordinates = peak_local_max(dist_transform, min_distance= 7, indices = True) # N by 2,
#     all_peaks = dist_transform[coordinates[:,0],coordinates[:,1] ].flatten()
#     num_peaks = all_peaks.size
#     k = min(num_peaks-1, int(num_peaks* 0.999)) # means we want the top 10 percent peaks
#     rval = np.partition(all_peaks, k)[k]
#     return rval
    
def overal_watershed_(probmap, thresh_water = 0.6, thresh_seg = 0.5,ratio = 0.2, dist_f = np.median):
    npad2 = ((1,1),(1,1))
    probmap = np.pad(probmap, npad2, mode='constant', constant_values=0)
    

    ret, fg_mask = cv2.threshold(probmap.astype(np.float32),0.7,255,0)
    ret, seed_mask = cv2.threshold(probmap.astype(np.float32),0.7,255,0)
    marker[seed_mask == 0] = 0
    ret, markers = cv2.connectedComponents(marker)
    
    fg_mask_color = cv2.cvtColor(fg_mask.astype('uint8'),cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(fg_mask_color, markers)

    remove_boarder_marker = markers[1:-1, 1:-1]
    return remove_boarder_marker


def _watershed(probmap, props,thresh_water = 0.8, thresh_seg = 0.2,ratio = 1, dist_f = np.max):
    if len(probmap.shape) ==2:
        npad2 = ((1,1),(1,1))
        probmap = np.pad(probmap, npad2, mode='constant', constant_values=0)
        cv3Thresh, thresh =  make_img(probmap, thresh_water * np.max(probmap))
        water_cv3Thresh, _ = make_img(probmap, thresh_seg * np.max(probmap))
    else:
        npad = ((1,1),(1,1),(0,0))
        probmap = np.pad(probmap, npad, mode='constant', constant_values=0)
        cv3Thresh = probmap.astype(np.uint8)
        thresh = cv3Thresh[...,0]
        water_cv3Thresh = probmap.astype(np.uint8)
        
    kernel = np.ones((3,3)).astype(np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=6)
    #imshow(sure_bg)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    dist_transform[dist_transform < 0.1*np.max(dist_transform)] = 0    
    #imshow(dist_transform)
    #ret, sure_fg = cv2.threshold(dist_transform,ratio*dist_f(dist_transform),255,0)
    
    #print props.eccentricity
    dist_transform = scipy.ndimage.filters.gaussian_filter(dist_transform, sigma = 2)
    if props.eccentricity >= 0.85:
        ret, sure_fg = cv2.threshold(dist_transform,ratio*dist_f(dist_transform),255,0)
    else:
        ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_f(dist_transform),255,0)
        dist_transform[sure_fg == 0] = 0 
        #imshow(dist_transform)
        coordinates = peak_local_max(dist_transform, min_distance= 9, indices = True) # N by 2,
        #print coordinates.shape
        sure_fg = np.zeros_like(dist_transform)
        sure_fg[coordinates[:,0],coordinates[:,1] ] = 1
        
    # Finding unknown region
    sure_fg = (sure_fg).astype(np.uint8)
    
    unknown = cv2.subtract(sure_bg,sure_fg)
    #imshow(unknown)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    #imshow(markers)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(water_cv3Thresh,markers)
    
    remove_boarder_marker = markers[1:-1, 1:-1]
    return remove_boarder_marker

def one_pass_watershed(probmap, thresh_water = 0.8, thresh_seg = 0.2,ratio = 0.2, dist_f = np.median, use_frangi= True):
    npad2 = ((1,1),(1,1))
    probmap = np.pad(probmap, npad2, mode='constant', constant_values=0)
    cv3Thresh, thresh =  make_img(probmap, thresh_water )
    ######
    #ret, thresh = cv2.threshold(probmap.astype(np.float32),thresh_water,255,0)
    #print thresh.shape
    #cv3Thresh = cv2.cvtColor(thresh.astype('uint8'),cv2.COLOR_GRAY2RGB)
    
    ret, fg_mask = cv2.threshold(probmap.astype(np.float32),thresh_seg,255,0)
    #print 'thresh_seg:', thresh_seg
    #imshow(fg_mask)

    water_cv3Thresh = cv2.cvtColor(fg_mask.astype('uint8'),cv2.COLOR_GRAY2RGB)
    ######
    
    if use_frangi:
        edge = frangi(thresh)     
    else:
        edge = np.zeros_like(thresh)

    boundaries = edge > 0.01 * np.max(edge)
    cv3Thresh[edge > 0.2 * np.max(edge), :] = 0

    #imshow(boundaries)
    #water_cv3Thresh, _ = make_img(probmap, thresh_seg )
    kernel = np.ones((3,3)).astype(np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    opening[boundaries == 1] = 0
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=6)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    #imshow(dist_transform)
    ret, sure_fg = cv2.threshold(dist_transform, ratio*get_thresh(dist_transform),255,0) 
    #imshow(sure_fg)
    # Finding unknown region
    sure_fg = (sure_fg).astype(np.uint8)
    
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    #imshow(markers)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers[boundaries == 1] = 0

    markers = cv2.watershed(water_cv3Thresh,markers)
    remove_boarder_marker = markers[1:-1, 1:-1]
    return remove_boarder_marker

tiny_size = 100
thresh_remove = 190
thre_ratio = 1.2

def overal_watershed(probmap, thresh_water = 0.8, thresh_seg = 0.2, ratio = 0.2, dist_f = np.median, org_img=None):
    first_pass = one_pass_watershed(probmap, thresh_water = thresh_water,
                                    thresh_seg = thresh_seg, ratio = ratio, dist_f = dist_f)
    gray_img = RGB2GRAY(org_img)
    mean_value = np.mean(gray_img)
    #imshow(gray_img)
    residule = probmap.copy()
    residule[first_pass != 1] = 0
    second_ratio = ratio
    second_pass = one_pass_watershed(residule, thresh_water = thresh_water, use_frangi = False,
                                    thresh_seg = thresh_seg, ratio = second_ratio, dist_f = dist_f)
    
    #tmp_img, _ = make_img(np.zeros_like(first_seg))
    new_seg = np.zeros_like(first_pass)
    
    region_count = 1
    first_regions = regionprops(first_pass)[1:]
    second_regions = regionprops(second_pass)[1:]
    
    for props in first_regions :
        this_mean = np.mean(gray_img[first_pass ==  props.label])
        #print this_mean
        if touchboundary(props.bbox, probmap.shape[0:2]) or props.area < tiny_size or (this_mean > thre_ratio*mean_value or this_mean  > thresh_remove):
            pass
        else:
            new_seg[first_pass ==  props.label] = region_count
            region_count = region_count + 1
            #print this_mean
    for props in second_regions :
        this_mean = np.mean(gray_img[second_pass ==  props.label])
        #print this_mean
        if touchboundary(props.bbox, probmap.shape[0:2]) or props.area < tiny_size or (this_mean > thre_ratio*mean_value or this_mean  > thresh_remove):
            pass
        else:
            new_seg[second_pass ==  props.label] = region_count
            region_count = region_count + 1
    return new_seg       


def single_watershed(probmap, thresh_water = 0.8, thresh_seg = 0.2,ratio = 0.3, dist_f = np.max,org_img=None):
    
    gray_img = RGB2GRAY(org_img)
    mean_value = np.mean(gray_img)
    
    first_seg = overal_watershed(probmap, thresh_water = thresh_water,ratio = ratio, 
                                 dist_f = dist_f, thresh_seg = thresh_seg)
    label_img = first_seg
    regions = regionprops(label_img)
    #tmp_img, _ = make_img(np.zeros_like(first_seg))
    tmp_img = np.zeros_like(first_seg)
    new_seg = np.zeros_like(first_seg)
    
    region_count = 1
    for props in regions[1:]:
        this_mean = np.mean(gray_img[label_img ==  props.label])
        #print this_mean
        if props.area < tiny_size or touchboundary(props.bbox, probmap.shape[0:2]) or (this_mean > thre_ratio*mean_value or this_mean  > thresh_remove):
            pass
        elif props.solidity > 0.9:
            new_seg[label_img ==  props.label] = region_count
            region_count = region_count + 1
        else:
            minr, minc, maxr, maxc = props.bbox
            tmp_img[label_img ==  props.label] = 255   
            thispatch = tmp_img[minr:maxr, minc:maxc].copy()
            #print thispatch.shape
            temp_seg = _watershed(thispatch, props, ratio=0.5, thresh_water = 0.5, 
                                        thresh_seg = 0.5,dist_f = np.max)
            tmp_img[minr:maxr, minc:maxc] = temp_seg
            #imshow(temp_seg)
            for tmp_props in regionprops(temp_seg)[1:]:
                this_mean = np.mean(gray_img[label_img ==  tmp_props.label])
                #print this_mean
                if tmp_props.area >= tiny_size and not touchboundary(props.bbox, probmap.shape[0:2]) or (this_mean > thre_ratio*mean_value or this_mean  > thresh_remove):
                    new_seg[tmp_img ==  tmp_props.label] = region_count
                    region_count = region_count + 1   
            tmp_img.fill(0)
            
    return new_seg
