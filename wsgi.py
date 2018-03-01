from flask import Flask, render_template, jsonify, request
from six.moves import urllib
import numpy as np
import tensorflow as tf
import glob, os
import sys
import os

def create_graph():
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
       if os.path.isfile("retrained_graph.pb"):
          print("esiste")
       else:
          print("non esiste")
       graph_def = tf.GraphDef()
       graph_def.ParseFromString(f.read())
       _ = tf.import_graph_def(graph_def, name='')


def classify_image(image_url):
    print("data : " , image_url)
    req = urllib.request.Request(image_url)
    response = urllib.request.urlopen(req)
    image_data = response.read()

    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions.argsort()[-len(predictions):][::-1]

        label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("labels.txt")]

        results = []

        var = 0

        for node_id in top_k:
           var += 1
           human_string = label_lines[node_id]
           score = predictions[node_id]
           results.append([human_string,score])
           if var == 10:
              break

        print(results)

        return results


application = Flask(__name__)

@application.route('/classify/', methods=['POST'])
def photoRecognize():
    #answer = classify_image(request.form['image_data'])
    print (os.getcwd())
    print("qui")
    answer = classify_image(request.form.get('image_data',0))
    #return jsonify(status='OK', results=answer)
    return render_template('index.html', text_to_render=answer)


@application.route("/")
def main():
    return render_template('index.html')

if __name__ == "__main__":
   application.run()
