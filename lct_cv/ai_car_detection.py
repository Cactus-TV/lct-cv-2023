import cv2
import pika
import numpy as np
import threading
import os
from dotenv import load_dotenv

load_dotenv()
MQ_POST_NAME = os.environ.get('RABBITMQ_QUEUE_POST')
MQ_GET_NAME = os.environ.get('RABBITMQ_QUEUE_GET')
MQ_LOGIN = os.environ.get('RABBITMQ_LOGIN')
MQ_PASSWORD = os.environ.get('RABBITMQ_PASSSWORD')
MQ_PORT = os.environ.get('RABBITMQ_PORT')
MQ_HOST = os.environ.get('RABBITMQ_HOST')
CV_PATH = os.environ.get('CV_PATH')
 
credentials = pika.PlainCredentials(MQ_LOGIN, MQ_PASSWORD)
parameters = pika.ConnectionParameters(port=MQ_PORT, host=MQ_HOST, credentials=credentials)

connection_get = pika.BlockingConnection(parameters)
connection_post = pika.BlockingConnection(parameters)
channel_get = connection_get.channel()
channel_post = connection_post.channel()

queue_info = channel_get.queue_declare(queue=MQ_GET_NAME, durable=True)
message_count = queue_info.method.message_count
if message_count > 0:
    channel_get.queue_purge(queue=MQ_GET_NAME)

queue_info = channel_post.queue_declare(queue=MQ_POST_NAME, durable=True)
message_count = queue_info.method.message_count
if message_count > 0:
    channel_get.queue_purge(queue=MQ_POST_NAME)

car_cascade = cv2.CascadeClassifier(CV_PATH)
 
def callback(ch, method, properties, body):
    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #convert video into gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w, channels = frame.shape
    frame = np.zeros((h, w, channels), dtype=np.uint8)

    #detect cars in the video
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    #to draw a rectangle in each cars 
    for (x,y,w,h) in cars:
        try:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            img_str = cv2.imencode('.jpg', frame)[1].tobytes()

            threading.Thread(target=post_queue, kwargs={'img_str': img_str}).start()
        except:
            continue
 
def get_queue():
    channel_get.basic_consume(queue=MQ_GET_NAME, on_message_callback=callback, auto_ack=True)
    channel_get.start_consuming()

def post_queue(img_str):
    channel_post.basic_publish(exchange='', routing_key=MQ_POST_NAME, body=img_str)

threading.Thread(target=get_queue, args=()).start()

while True:
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        connection_post.close()
        connection_get.close()
        break