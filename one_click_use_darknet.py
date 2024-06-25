from operator import truediv
from tkinter import E
import open3d
# from visualization import VisOpen3D
import cv2
import numpy as np
import pickle
import os
# from ObjectDetectionService.ODetectorClient import ObjectDetectionClient
from pyquaternion import Quaternion
import sys
import darknet
import glob

class poses:
    def __init__(self, workdir):
        self.poses_file = os.path.join(workdir, 'poses.txt')
        self.new_poses_flie = os.path.join(workdir, 'new_poses.txt')
        if os.path.exists(os.path.join(workdir, 'new_poses.txt')):
            os.remove(os.path.join(workdir, 'new_poses.txt'))

    def fix_poses_light(self):
        count = 1
        with open(self.new_poses_flie, 'a') as file:
            with open(self.poses_file, 'r') as f:
                Lines = f.readlines()

                for line in Lines:

                    li = list(line.split(" "))
                    z = [float(x) for x in li]

                    RT = np.zeros((4, 4))
                    RT[:3, :4] = np.array(z).reshape(3,4)
                    RT[3, 3] = 1
                    # print (RT)
                    file.write(str(count) + "\n")
                    for l in RT:
                        q = [str(x) for x in l]
                        file.writelines(" ".join(q) + "\n")
                    count += 1
        f.close()
        file.close()


    def fix_poses(self):
        count = 1
        with open(self.new_poses_flie, 'a') as file:
            with open(self.poses_file, 'r') as f:
                Lines = f.readlines()

                for line in Lines:
                    if Lines.index(line) == 0:
                        continue
                    li = list(line.split(" "))
                    z = [float(x) for x in li]

                    t = np.array([z[1:4]])
                    qx, qy, qz, qw = z[4:8]
                    quat = Quaternion(qw, qx, qy, qz)
                    R = quat.rotation_matrix
                    RT = np.zeros((4, 4))
                    RT[:3, :3] = R
                    RT[:3, 3] = t
                    RT[3, 3] = 1
                    # print (RT)
                    file.write(str(count) + "\n")
                    for l in RT:
                        q = [str(x) for x in l]
                        file.writelines(" ".join(q) + "\n")
                    count += 1
        f.close()
        file.close()
class intrinsics_from_calibration:
    def __init__(self, workdir):
        self.workdir = workdir
        self.calibration_file = os.path.join(workdir, 'calibration.yaml')
    def get_intrinsics(self):
        import yaml
        with open(self.calibration_file, 'r+') as fp:
            # read an store all lines into list
            lines = fp.readlines()
            # move file pointer to the beginning of a file
            if lines[0] == '%YAML:1.0\n':
                fp.seek(0)
                # truncate the file
                fp.truncate()
                # start writing lines except the first line
                # lines[1:] from line 2 to last line
                fp.writelines(lines[2:])

        with open(self.calibration_file, "r") as stream:
            yaml = yaml.safe_load(stream)
        return yaml["camera_matrix"]["data"]


class detect_fruit:
    def __init__(self, workdir,show_detectios=False ):
       
       

        self.workdir = workdir
        self.rgb_path = os.path.join(workdir, 'rgb')
        self.new_poses_flie = os.path.join(workdir, 'new_poses.txt')
        args = self.parser()
        self.network, self.class_names, self.class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
        )
        self.show_detections = show_detectios
    def parser(self):
        import argparse
        parser = argparse.ArgumentParser(description="YOLO Object Detection")
        parser.add_argument("--input", type=str, default="/home/tevel/Documents/RTAB-Map/try/rgb",
                            help="image source. It can be a single image, a"
                            "txt with paths to them, or a folder. Image valid"
                            " formats are jpg, jpeg or png."
                            "If no input is given, ")
        parser.add_argument("--batch_size", default=1, type=int,
                            help="number of images to be processed at the same time")
        parser.add_argument("--weights", default="/tinyyolo4_3l_generic/yolo4-tiny-3l_best.weights",
                            help="yolo weights path")
        parser.add_argument("--dont_show", action='store_true',
                            help="windown inference display. For headless systems")
        parser.add_argument("--ext_output", action='store_true',
                            help="display bbox coordinates of detected objects")
        parser.add_argument("--save_labels", action='store_true',
                            help="save detections bbox for each image in yolo format")
        parser.add_argument("--config_file", default="/tinyyolo4_3l_generic/yolo4-tiny-3l.cfg",
                            help="path to config file")
        parser.add_argument("--data_file", default="/tinyyolo4_3l_generic/obj.data",
                            help="path to data file")
        parser.add_argument("--thresh", type=float, default=.3,
                            help="remove detections with lower confidence")
        return parser.parse_args()

    def load_images(images_path):
        """
        If image path is given, return it directly
        For txt file, read it and return each line as image path
        In other case, it's a folder, return a list with names of each
        jpg, jpeg and png file
        """
        input_path_extension = images_path.split('.')[-1]
        if input_path_extension in ['jpg', 'jpeg', 'png']:
            return [images_path]
        elif input_path_extension == "txt":
            with open(images_path, "r") as f:
                return f.read().splitlines()
        else:
            return glob.glob(
                os.path.join(images_path, "*.jpg")) + \
                glob.glob(os.path.join(images_path, "*.png")) + \
                glob.glob(os.path.join(images_path, "*.jpeg"))


    def getint(self, name):
        # sortin the rgb files
        num, _ = name.split('.')
        return int(num)

    
    def image_detection(self,image_or_path, network, class_names, class_colors, thresh):
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        darknet_image = darknet.make_image(width, height, 3)
        if type(image_or_path) == str:
            image = cv2.imread(image_or_path)
        else:
            image = image_or_path
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        darknet.free_image(darknet_image)
        if self.show_detections:
            image = darknet.draw_boxes(detections, image_resized, class_colors)
            cv2.imshow("",cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
        
    def main(self):
        # make the main detections and trajectories file
        detection = []
        for rgb in sorted(os.listdir(self.rgb_path), key=self.getint):
            # print(rgb)
            rgb_image = cv2.imread(os.path.join(self.rgb_path, rgb))
            # circles = self.detector.search(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),_threshold=0.4)
            image, detections = self.image_detection(
            rgb_image, self.network, self.class_names, self.class_colors, .45)
            # print (list(detection[2]))
            # if self.show_detections:
                # self.show_detection(rgb_image,detections)
            # print(circles)
        
            temp =[]
            for detect in detections: 
                xc, yc, w,h = list(detect[2])
                xc = int(xc * 848 / 640)
                r = max(w,h)/2
                temp.append([xc,yc,r])
               
                
            if len(detections)>0:
                detection.append([rgb, np.array(temp)])
        # print(detection)
        trajectory = []
        with open(self.new_poses_flie, 'r') as f:
            Lines = f.readlines()
            for line in Lines:
                trajectory.append(line.strip())
        extrins = []
        for line in trajectory:
            if trajectory.index(line) % 5 != 0:
                # print (line)
                extrins.append(line)
        count = 0
        count_b = 0
        count_c = 0
        mini = []
        while count < len(extrins):
            inlis = []
            while count_c < 4:
                if count_b >= len(extrins):
                    break
                inlis.append(extrins[count_b].replace(" ", ","))
                count_b += 1
                count_c += 1
            mini.append(inlis)
            count_c = 0
            count += 1
        # print(mini)
        data = []
        count = 0
        if os.path.exists(os.path.join(workdir, "data")):
            os.remove(os.path.join(workdir, "data"))
        with open(os.path.join(self.workdir, "data"), 'wb') as fp:
            for line in detection:
                line.append(mini[count])
                data.append(line)
                count += 1

            pickle.dump(data, fp)
class ProjectToPCL:
    def __init__(self,intrinsics,workdir,debug):
        self.workdir = workdir
        self.rgb_path = os.path.join(workdir, 'rgb')
        self.depth_path = os.path.join(workdir, 'depth')
        self.intrinsics = intrinsics
        self.deproject_matrix = np.linalg.inv(intrinsics)
        self.outer = np.linalg.inv(np.array([0., 0., 1., 0., -1., 0., 0., 0., 0., -1., 0., 0., 0, 0, 0, 1]).reshape(4, 4))
        self.point_cloud = []
        with open(os.path.join(self.workdir,'data'), 'rb') as fp:
            self.data_list = pickle.load(fp)
        self.debug = debug

    def project(self, depth_image, extrinsics, circles, fname):
        depth_image = np.asarray(depth_image)
        for sample in circles:
            sample_depth = self.get_sample_depth(sample[:2], depth_image)
            if sample_depth == 0:
                continue
            fname = int(fname)
            _3dpos = self.deproject(sample[:2], sample_depth, extrinsics)
            world_radius = sample[2]*2*sample_depth/(intrinsics[0,0])
            self.point_cloud.append(np.append(np.append(_3dpos[0:3],fname),np.round((world_radius*1000)-15)))

        # return circles
    def deproject(self,pix,depth,extrinsics):
        cam_pos = self.to_cam(pix,depth)
        _3d_pos = self.cam_to_world(cam_pos,extrinsics)
        return _3d_pos
    def to_cam(self,Pt,d):
        Pt_aug = np.append(Pt, 1.0)
        World = np.dot(self.deproject_matrix, Pt_aug)
        WorldCam = World * d / World[2]  # For real camera
        return WorldCam

    def cam_to_world(self,cam_coordinate,extrinsics):
        CamWorld_aug = np.append(cam_coordinate, 1)
        Body = np.dot(np.linalg.inv(extrinsics), CamWorld_aug)
        Body = Body[:3]
        return Body

    def get_sample_depth(self,pix,depth_image):
        d_sub =depth_image[int(pix[1])-3:int(pix[1])+3,int(pix[0])-3:int(pix[0])+3]
        return np.median(d_sub[:])

    def getint(self,name):
        num, _ = name.split('.')
        return int(num)

    def getintfromlist(self,name):
        return int(name)

    def main(self):
        # project the detections
        count = 0
        images_suffix = '.jpg'
        depth_suffix = '.png'
        imgs_fnames = [x[:x.index(images_suffix)] for x in sorted(os.listdir(self.rgb_path), key=self.getint)]
        depth_fnames = [x[:x.index(depth_suffix)] for x in sorted(os.listdir(self.depth_path), key=self.getint)]
        fnames = sorted(list(set(imgs_fnames) & set(depth_fnames)), key=self.getintfromlist)
        for fname in fnames:
            # print(fname)
            img_fname = fname + images_suffix
            mask_fname = fname + depth_suffix
            rgb = cv2.imread(os.path.join(self.rgb_path, img_fname))
            depth = cv2.imread(os.path.join(self.depth_path, mask_fname), -1)
            depth[depth > 4000] = 0
            depth_int = depth.copy()
            depth = depth / 1000

            extrinsics = np.array([float(i) for i in (
                    self.data_list[count][2][0] + ',' + self.data_list[count][2][1] + ',' + self.data_list[count][2][2] + ',' +
                    self.data_list[count][2][3]).split(',')]).reshape(4, 4)

            circles = self.data_list[count][1]
            count += 1
            # detection_obj.show_detection(rgb,circles)
            self.project(depth, self.outer@np.linalg.inv(extrinsics), circles, fname)
            if (self.debug == True and count == 1):
                rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(open3d.geometry.Image(rgb),
                                                                                   open3d.geometry.Image(depth_int))
                intrinsics2 = open3d.camera.PinholeCameraIntrinsic(848, 480, 621.91, 621.81, 429.97, 242.29)
                pcd1 = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics2)
                pcd1.estimate_normals()
                print("debug")
                break
                # vis = VisOpen3D(width=w, height=h, visible=window_visible)
                # vis.add_geometry(pcd1)
                # vis.draw_points3D(np.array(detection_obj.point_cloud), color=[0.2, 0.9, 0.8])
                # vis.run()
                # vis.destroy_window()

        for index,i in enumerate(self.point_cloud) :
            if  str(i[0]) == 'nan':
                del self.point_cloud[index]
        np.savetxt(os.path.join(self.workdir, "points.txt"), np.array(self.point_cloud))
        # import pickle
        #
        # with open(os.path.join(self.workdir, "points.txt"), 'wb')as fp:
        #     pickle.dump(self.point_cloud, fp)
        # f = open(os.path.join(self.workdir, "points.txt"), "a")
        # f.write(str(np.array(self.point_cloud)))
        # f.close()

        # return np.array(self.point_cloud)
class DataSet:
    def __init__(self, points):
        self.points = points
        self.number_of_points = points.shape[0]
        self.cluster_name = np.zeros(self.number_of_points)
        self.cluster_frames = []
        self.number_of_neighbours = np.zeros(self.number_of_points)

    def is_all_clustered(self):
        return np.sum(self.cluster_name > 0) == self.number_of_points

    def is_pt_clustered(self, ind):
        return self.cluster_name[ind] > 0
class DBSCAN_TA:
    def __init__(self, Hand_fruit_count,workdir):
        self.Hand_fruit_count = Hand_fruit_count
        self.workdir = workdir
    def metrics(self, x, y):
        m_ = x[:-2] - y[:-2]
        t = (x[-2] - y[-2])
        if t == 0:
            z = np.inf
        else:
            z = 0
        d = np.sqrt(m_ @ m_.T + z)
        return d

    def find_all_neighbours(self, ind, cluster_number):
        distances = np.inf * np.ones(self.db.number_of_points)
        for j in range(self.db.number_of_points):
            if j == ind or self.db.cluster_name[j] > 0 or self.db.points[j, 3] in self.db.cluster_frames[
                cluster_number - 1]:
                continue
            distances[j] = self.metrics(self.db.points[ind], self.db.points[j])
        neigbours_index = np.where(distances < self.eps)[0]
        return neigbours_index

    def find_clustered_initated_at(self, ind, cluster_number):
        neigbours_index = self.find_all_neighbours(ind, cluster_number)
        self.db.cluster_name[ind] = cluster_number
        if len(neigbours_index) >= self.minPts:
            for j in neigbours_index:
                self.db.cluster_frames[cluster_number - 1].append(self.db.points[j, 3])
                self.find_clustered_initated_at(j, cluster_number)
        return 0

    def db_scan(self, pts, eps=0.1, minPts=5):
        self.db = DataSet(pts)
        self.minPts = minPts
        self.eps = eps
        index_perm = np.random.permutation(self.db.number_of_points)
        # index_perm = np.arange(0,self.db.number_of_points)
        current_indx = 0
        cluster_number = 1
        while not self.db.is_all_clustered():
            while self.db.is_pt_clustered(index_perm[current_indx]):
                current_indx += 1
            index_db = index_perm[current_indx]
            self.db.cluster_frames.append([self.db.points[index_db, 3]])
            self.find_clustered_initated_at(index_db, cluster_number)
            cluster_number += 1
        cluster_name = self.db.cluster_name
        NumberOfClusters = np.max(self.db.cluster_name)
        Accuracy = int(((NumberOfClusters/self.Hand_fruit_count )*100))
        print(f'Number Of Fruits Found: {NumberOfClusters}')
        print(f'Accuracy(%): {Accuracy}')
        points = []
        centroids = []
        count = 0
        pts = np.array([np.delete(v, [-2]) for v in pts])
        while count < NumberOfClusters+1:
            points.append(pts[np.where(cluster_name == count)[0]])
            count += 1
        # points = points[1:]
        for i in points:
            centroids.append(np.mean(i, axis=0))
        for index,i in enumerate(centroids) :
            if  str(i[0]) == 'nan':
                del centroids[index]
        np.savetxt(os.path.join(self.workdir, "centroids.txt"), np.array(centroids))
        # return np.array(centroids)
class ColorClassifier():
    def __init__(self,Color_Grading):
        self.Color_Grading = Color_Grading
        if Color_Grading:
            with open('/weights/color-classifier/clf.pkl', 'rb') as file:
                self.classifier_net  = pickle.load(file)
            with open('/weights/color-classifier/scaler.pkl','rb') as file:
            # with open('/weights/color_classifier/scaler.pkl', 'rb') as file:
                self.scalar  = pickle.load(file)



    def grade(self,rgbcrop, maskcrop):
        if self.Color_Grading:
            # bgrcrop = cv2.cvtColor(rgbcrop, cv2.COLOR_RGB2BGR)
            rgb_vector = rgbcrop[maskcrop]
            if len(rgb_vector) == 0:
                return -1
            r, g, b= rgb_vector[:, 0], rgb_vector[:, 1], rgb_vector[:, 2]
            b = b.astype(float)
            g = g.astype(float)
            r = r.astype(float)
            mean_sum_rgb = np.mean(r + g + b)
            Rn = np.mean(r) / mean_sum_rgb  # Normalized red of RGB (Rn)
            Gn = np.mean(g) / mean_sum_rgb  # Normalized green of RGB(Gn)
            Bn = np.mean(b) / mean_sum_rgb  # Normalized blue of RGB(Bn)

            hls = cv2.cvtColor(rgb_vector.reshape((-1, 1, 3)), cv2.COLOR_RGB2HLS).reshape((-1, 3))
            h, l, s = hls[:, 0].astype(float), hls[:, 1].astype(float), hls[:, 2].astype(float)
            meanL = np.mean(l)  # mean of L in HSL
            meanH = np.mean(h)
            meanS = np.mean(s)

            ycbcr = cv2.cvtColor(rgb_vector.reshape((-1, 1, 3)), cv2.COLOR_RGB2YCrCb).reshape((-1, 3))
            y, cr, cb = ycbcr[:, 0].astype(float), ycbcr[:, 1].astype(float), ycbcr[:, 2].astype(float)
            sdCb = np.std(cb)  # SD of Cb in YCbCr
            meanY = np.mean(y)  # mean of L in HSL
            meanCb = np.mean(cb)
            meanCr = np.mean(cr)

            lab = cv2.cvtColor(rgb_vector.reshape((-1, 1, 3)), cv2.COLOR_RGB2LAB).reshape((-1, 3))
            L, a, bu = lab[:, 0].astype(float), lab[:, 1].astype(float), lab[:, 2].astype(float)
            meanA = np.mean(a)  # mean of a in Lab
            meanB = np.mean(bu)  # mean of b in Lab
            meanLa = np.mean(L)
            hue_angle = np.mean(np.arctan(bu / a))

            EXR = 1.4 * Rn - Gn  # excess red
            cive = 0.441 * Rn - 0.811 * Gn + 0.385 * Bn + 18.78  # Color index for extracted vegetation cover(CIVE)
            rbi = (Rn - Bn) / (Rn + Bn)  # Red-blue contrast (RBI)

            parameters = [EXR, Rn, Gn, Bn, meanL, meanH, meanS,meanLa, meanA, meanB, meanCb, meanCr, meanY,sdCb, cive, rbi, hue_angle]
            features = self.scalar.transform(np.array(parameters).reshape(1, -1))
            grade = float(self.classifier_net.predict(features))

            return grade
        else:
            return -1
class modeling:
    def __init__(self,workdir,centroids):

        self.workdir = workdir
        self.pcd = open3d.io.read_point_cloud(os.path.join(workdir, 'cloud.ply'))
        self.centroids = centroids
    def modeling(self):
        import matplotlib.pyplot as plt
        import math
        import seaborn as sns
        vis = VisOpen3D(width=640, height=480, visible=True)
        run_flag = True
        vis.add_geometry(self.pcd)
        vis.draw_points3D(self.centroids[:,0:3], color=[0.2, 0.9, 0.8])
        vis.run()
        # # vis2 = VisOpen3D(width=640, height=480, visible=True)
        # run_flag = True
        # # vis2.add_geometry(self.pcd)
        # for i in self.centroids :
        #       i[0] = 0
        # # print(d)
        # # vis2.draw_points3D(self.centroids, color=[0.2, 0.9, 0.8])
        # # vis2.run()
        # rotated = self.centroids[:,1:3]
        # rotated = np.rot90(rotated, k=1, axes=(0,1))
        # plt.gca().invert_yaxis()
        # plt.gca().invert_xaxis()
        # plt.scatter(rotated[0],rotated[1],s=2)
        # plt.show()
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = False
        tevel = plt.imread('tevel.png')
        fig,ax = plt.subplots(2,2,figsize=(12, 10))
        plt.suptitle('Fruit Count Statistics', size=18)
        newax = fig.add_axes([0.7, 0.8, 0.2, 0.2], anchor='NE', zorder=1)
        newax.imshow(tevel)
        newax.axis('off')
        x= self.centroids[:,2]
        y = self.centroids[:,0]
        radius = self.centroids[:,3]

        his = ax[0][0].hist(radius, bins=8)
        ax[0][0].title.set_text("Distrbution of apple size")
        ax[0][0].legend(['Total fruits count is:\n            %d' %(len(centroids))],loc='upper left',ncol=2,handlelength=0, handletextpad=0,prop={'size': 12})
        ax[0][0].set_xlabel('Size(mm)\n',fontsize=12)
        ax[0][0].set_ylabel('Number of fruits',fontsize=12)


        # plt.gca().invert_yaxis()
        # plt.gca().invert_xaxis()
        # ax2.scatter(x,y,s =2)

        bin_size_x = 0.6
        bin_size_y = 6
        xmax, xmin = np.max(x), np.min(x)
        ymax, ymin = np.max(y), np.min(y)
        xmatsize = math.ceil(abs(xmax - xmin)/bin_size_x)
        ymatsize = math.ceil(abs(ymax - ymin)/bin_size_y)
        ymatsize = 1
        mat = np.zeros((ymatsize,xmatsize))
        mat_r = np.zeros((ymatsize,xmatsize))
        y_label = np.linspace(ymin,ymax,mat.shape[0])
        x_label = np.linspace(xmin,xmax,mat.shape[1])
        for i in range(mat.shape[1]):
            for j in range(mat.shape[0]):
                x_l,x_h = xmin+i*bin_size_x,xmin+(i+1)*bin_size_x
                y_l,y_h = ymin+j*bin_size_y,ymin+(j+1)*bin_size_y
                x_valid = np.logical_and(x<x_h,x>=x_l)
                y_valid = np.logical_and(y<y_h,y>=y_l)
                y_valid = y
                valid = np.logical_and(x_valid,y_valid)

                mat[j,i] = np.sum(valid)
                mat_r[j,i] = np.sum(radius[valid])/mat[j,i]
        mat_r = mat_r[:, 9:-6]
        mat = mat[:,9:-6]
        mat_fl = np.flip(mat,0)
        mat_r[:, 6:-6]
        mat_r_fl = np.flip(mat_r,0)
        im = ax[0][1].imshow(mat_r_fl,cmap = 'coolwarm' , interpolation='nearest')
        cbar = ax[0][1].figure.colorbar(im,ax=ax[0][1],shrink=0.5,label='Avg of fruits size(mm)')
        ax[0][1].set_aspect(10)
        ax[0][1].title.set_text("Heatmap of fruits sizes by tree")
        ax[0][1].set_xlabel('Tree(meter)', fontsize=12)
        ax[0][1].set_ylabel('Size of fruits(avg mm)', fontsize=12)
        im.axes.get_yaxis().set_visible(False)
        im2 = ax[1][1].imshow(mat_fl, cmap='coolwarm', interpolation='nearest',)
        cbar = ax[1][1].figure.colorbar(im2, ax=ax[1][1], shrink=0.5,label='Number of fruits')
        im2.axes.get_yaxis().set_visible(False)
        ax[1][1].set_aspect(10)
        ax[1][1].title.set_text("Heatmap of fruits by tree")
        ax[1][1].set_xlabel('Tree(meter)', fontsize=12)





        grade1 = np.array([x for x in self.centroids if x[3]<80])
        grade2 = np.array([x for x in self.centroids if x[3] >= 80 and x[3] < 90])
        grade3 = np.array([x for x in self.centroids if x[3] >= 90 and x[3] < 100])
        grade4 = np.array([x for x in self.centroids if x[3] >= 100])
        colors = ['red', 'green', '#F6BE00','blue']
        ax[1][0].scatter(grade1[:,2],grade1[:,0],
                            color=colors[0],s=2, label='<80mm')
        ax[1][0].scatter(grade2[:,2],grade2[:,0],
                            color=colors[1], s=2,label='80mm-90mm')
        ax[1][0].scatter(grade3[:,2],grade3[:,0],
                            color=colors[2], s=2,label='90mm-100mm')
        ax[1][0].scatter(grade4[:,2],grade4[:,0],
                            color=colors[3], s=2,label='>100mm')
        ax[1][0].legend(loc='upper center',markerscale=3., bbox_to_anchor=(0.5, -0.08),
                   ncol=2,)
        ax[1][0].title.set_text("2D map of row fruit distribution")

        plt.show()



if __name__ == '__main__':
    workdir = '/home/tevel/Documents/RTAB-Map/try/first_side' # TODO
    # fix poses file
    poses(workdir).fix_poses_light()
    print('poses fixed')

    # make detections file
    show_detectios = True # TODO
    detections = detect_fruit(workdir,show_detectios).main()
    print('detections file done')

    #project
    intrinsics = np.array(intrinsics_from_calibration(workdir).get_intrinsics()).reshape(3, 3)
    debug = False #TODO
    if os.path.exists(os.path.join(workdir,"points.txt")):
        pts = np.loadtxt(os.path.join(workdir, "points.txt"))

    else:
        ProjectToPCL(intrinsics,workdir,debug).main()
        pts = np.loadtxt(os.path.join(workdir, "points.txt"))

    print('3d points done')

    #clastering
    Hand_fruit_count = 2800 #TODO for accuracy validation
    if os.path.exists(os.path.join(workdir,"centroids.txt")):
        centroids = np.loadtxt(os.path.join(workdir, "centroids.txt"))
    else:
        DBSCAN_TA(Hand_fruit_count, workdir).db_scan(pts, eps=0.2, minPts=1)  # TODO test best eps and min sample
        centroids = np.loadtxt(os.path.join(workdir, "centroids.txt"))
    print (len(centroids))
    print('clustering done')
    show_model = True  # TODO
    if show_model:
        #modeling
        model = modeling(workdir,centroids).modeling()
        print('model done')
