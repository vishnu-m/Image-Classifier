from collections import defaultdict
import face_recognition
import os,cv2,shutil 
 
unclassified_images_path = "/home/vishnu/Desktop/testimages/accept" #IMPORTANT : Change the path to your unclassified images directory
unclassified_images_list = os.listdir(unclassified_images_path)
  
 
individuals_images_path = "/home/vishnu/Desktop/testimages/check" #IMPORTANT : Change the path to your targetted individuals directory
individuals_images_list = os.listdir(individuals_images_path)
 
individuals_name = []
for person in individuals_images_list: #jpg images only 
	individuals_name.append(person[:-4])
 

individuals_face_encodings = []
try:
	for individuals in individuals_images_list:
		image = face_recognition.load_image_file(individuals_images_path + "/" + individuals)
		face_encoding = face_recognition.face_encodings(image)[0]
		individuals_face_encodings.append(face_encoding)
 
except Exception as e:
	print("Error: [" + e.errno + "] " + e.strerr)
 
 
# Original dictionary which will contain the classified images for each individuals 
individual_pictures = defaultdict(lambda:list())
 
for images in unclassified_images_list:
	
	frame = cv2.imread(unclassified_images_path + "/" + images)
	small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
	rgb_small_frame = small_frame[:,:,::-1]
	face_locations = face_recognition.face_locations(rgb_small_frame)
	face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
 
	for face_encoding in face_encodings:
		name = "unknown"
		for index in range(len(individuals_face_encodings)):
			face_distance = face_recognition.face_distance([individuals_face_encodings[index]],face_encoding)
 
			if face_distance < 0.5:
				name = individuals_name[index]
		individual_pictures[name].append(unclassified_images_path + "/" + images)
 
results_dir = "/home/vishnu/Desktop/testimages/out"   #IMPORTANT : Change the path to the directory where you want to store the images of each individual.
if not os.path.exists(results_dir):
	os.makedirs(results_dir)
else:
	shutil.rmtree(results_dir)
	os.makedirs(results_dir)
 
for person in individuals_name:
	dest = results_dir + "/" + person
	os.makedirs(results_dir + "/" + person)
	for picture in individual_pictures[person]:
		shutil.copy(picture,dest)