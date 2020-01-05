from flask_restplus import Api, Resource, fields
from flask import Flask, send_file, make_response
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
import cv2
import numpy as np
import base64
from fault_detector import FaultDetector
from waitress import serve

app = Flask(__name__)
api = Api(app)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)

test = api.model('Test', {'msg': fields.String('The language')})

fd = FaultDetector('./bin/tmp1/output_graph.pb', './bin/tmp1/output_labels.txt', './bin/tmp/output_graph.pb', './bin/tmp/output_labels.txt')

@api.route('/process')
class ADRSRest(Resource):
    @api.expect(upload_parser)
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file'].read()
        npimg = np.fromstring(uploaded_file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return fd.mark_faults(img)
