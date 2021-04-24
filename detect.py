import json

import cv2
from pytorchyolo import detect, models


class ObjectDetector:

	def __init__(self, img_path, model_cfg, model_weights):
		super(ObjectDetector, self).__init__()

		self.model = models.load_model(model_cfg, model_weights)
		# Load the image as an numpy array
		self.img = cv2.imread(img_path)
		self.img_copy = self.img.copy()
		self.orig_h, self.orig_w = self.img.shape[:2]
		self.n_patches_each_dim = 4
		self.each_patch_height = int(self.orig_h // self.n_patches_each_dim)
		self.each_patch_width = int(self.orig_w // self.n_patches_each_dim)
		self.pred_boxes_list = []
		self.gt_boxes_list = []
		self.iou_list = []

	def process_groundtruth(self, json_file):
		with open(json_file) as file:
			annotation_data = json.load(file)

		for j in annotation_data:
			regions = annotation_data[j]["regions"]
			for region in regions:
				if region["region_attributes"]["category_name"] == "person":
					x = int(region["shape_attributes"]["x"])
					y = int(region["shape_attributes"]["y"])
					w = int(region["shape_attributes"]["width"])
					h = int(region["shape_attributes"]["height"])
					x1, y1, x2, y2 = x, y, x+w, y+h
					self.gt_boxes_list.append([x1, y1, x2, y2])

	def bb_intersection_over_union(self, boxA, boxB):
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
		# compute the area of intersection rectangle
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
		# return the intersection over union value
		return iou

	def run(self):
		for patch_y in range(self.n_patches_each_dim):
			for patch_x in range(self.n_patches_each_dim):
				crop_img = self.img[patch_y * self.each_patch_height:(patch_y + 1) * self.each_patch_height,
						   patch_x * self.each_patch_width:(patch_x + 1) * self.each_patch_width]

				# Runs the YOLO model on the image
				boxes = detect.detect_image(self.model, crop_img, img_size=608, conf_thres=0.1, nms_thres=0.1)

				num_detections = boxes.shape[0]
				for x1, y1, x2, y2, prob, cls in boxes:
					# check for only person class
					if int(cls) == 0:
						x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
						x1, x2 = x1 + patch_x * self.each_patch_width, x2 + patch_x * self.each_patch_width
						y1, y2 = y1 + patch_y * self.each_patch_height, y2 + patch_y * self.each_patch_height
						# cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
						self.pred_boxes_list.append([x1, y1, x2, y2, prob])

				# self.img_copy[patch_y * each_patch_height:(patch_y + 1) * each_patch_height,
				# patch_x * each_patch_width:(patch_x + 1) * each_patch_width] = crop_img
				# cv2.imshow("output", self.img_copy)
				# cv2.waitKey()

	def calculate_avg_iou(self):
		for pred_box in self.pred_boxes_list:
			for gt_box in self.gt_boxes_list:
				iou = self.bb_intersection_over_union(gt_box, pred_box[:4])
				if iou > 0.3:
					self.iou_list.append(iou)
					prob = round(pred_box[4], 3)
					cv2.rectangle(self.img_copy, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 0, 255), 2)
					cv2.putText(self.img_copy, str(prob), (pred_box[0] - 10, pred_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
								0.5, (255, 0, 0), 2, cv2.LINE_AA)

		avg_iou = sum(self.iou_list) / len(self.iou_list)
		print(len(self.iou_list))
		print(avg_iou)
		cv2.putText(self.img_copy, "avg_iou = " + str(avg_iou), (1200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8,
					cv2.LINE_AA)
		cv2.imwrite("output/output_test_img_upscale_iou.jpg", self.img_copy)
		print("please view the output image in the output folder")


if __name__ == "__main__":
	image_path = "test_img/TopDownHumanDetection_4032x3024.jpg"
	json_path = "test_img/person.json"
	model_cfg = "cfg/yolov3.cfg"
	model_weights = "weights/yolov3.weights"

	detector = ObjectDetector(image_path, model_cfg, model_weights)    # initialize model and read image
	detector.run()												 # perform object detection on the image
	detector.process_groundtruth(json_path)						 # extract ground truth bbox from annotation
	detector.calculate_avg_iou()								 # calculate average iou of the detections


