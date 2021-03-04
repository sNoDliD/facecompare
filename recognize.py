# from keras_facenet import FaceNet
# import cv2
# embedder = FaceNet()
# print(1)
# # Gets a detection dict for each face
# # in an image. Each one has the bounding box and
# # face landmarks (from mtcnn.MTCNN) along with
# # the embedding from FaceNet.
# image = cv2.imread('IMG_20210215_141440.jpg')
# detections = embedder.extract(image, threshold=0.95)
# f = 2
# # # If you have pre-cropped images, you can skip the
# # # detection step.
# # embeddings = embedder.embeddings(images)



# from facenet_pytorch import MTCNN, InceptionResnetV1
# mtcnn = MTCNN(image_size=180, margin=0)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
# from PIL import Image
# img = Image.open('photos/IMG_20210215_141440.jpg')
# # Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path='res.jpg')
# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))
# # Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))

from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face, InceptionResnetV1
import cv2


# img = Image.open('photos/IMG_20210215_141538.jpg').resize((1000, int(3880*1000/5184)))
img = cv2.imread('photos/IMG_20210215_141521.jpg')
scale = 400/img.shape[1]
img = cv2.resize(img, None, fx=scale, fy=scale)
cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
boxes, probs, points = mtcnn.detect(img, landmarks=True)
print(len(boxes))
# Draw boxes and save faces

# Get cropped and prewhitened image tensor
mtcnn = MTCNN()
img_cropped = mtcnn(img, save_path='photo.jpg')

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))
for i, (box, point) in enumerate(zip(boxes, points)):
    x1, y1, x2, y2 = (int(x) for x in box)
    # img_embedding = resnet(img[x1:x2, y1:y2])

    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
    # extract_face(img, box, save_path='detected_face_{}.png'.format(i))
# img_draw.save('annotated_faces.png')

# Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path=<optional save path>)

# Calculate embedding (unsqueeze to add batch dimension)
cv2.imshow('f', img)
cv2.waitKey()
print('end')