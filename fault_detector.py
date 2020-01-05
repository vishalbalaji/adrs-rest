from bin.defectanalyzernew import load_graph, load_labels, predict, read_tensor_from_image_file
from bin.defectmarkingcall import defmarking
import os
import cv2
import base64

class FaultDetector():
    def __init__(self, graph1_path, labels1_path, graph2_path, labels2_path):
        self.graph1 = load_graph(graph1_path)
        self.labels1 = load_labels(labels1_path)
        self.graph2 = load_graph(graph2_path)
        self.labels2 = load_labels(labels2_path)

    def classify_discontinuity(self, img_path):
        img_tensor = read_tensor_from_image_file(img_path)
        discontinuity = 'Processed RT film is ' + predict(img_tensor, self.graph1, self.labels1, self.graph2, self.labels2)
        if discontinuity == 'Processed RT film is Non discontinuity':
            normal = True
        else:
            normal = False

        return discontinuity, normal

    def img_to_bytes(self, img):
        retval, imgbuf = cv2.imencode('.png', img)
        png_as_text = base64.b64encode(imgbuf).decode()
        return png_as_text

    def mark_faults(self, img):
        cv2.imwrite('./bin/temp.png', img)
        (discontinuity, normal) = self.classify_discontinuity('./bin/temp.png')
        print(discontinuity, normal)
        if normal:
            final_result = {
                    'success': True,
                    'discontinuity': discontinuity,
                    'faulty': False,
                    }
            return final_result

        else:
            results = defmarking('./bin/temp.png', "Penetrometer 5", "Normal")
            final_img = self.img_to_bytes(results[1])
            direction = results[2].lstrip().strip('()')
            faults = []
            for fault in results[3]:
                if fault[2] == 'R':
                    fault_length = f'{fault[1]:.2f}' + ' (Ø)'
                else:
                    fault_length = f'{fault[1]:.2f}'
                new_fault = {'fault_length': fault_length}
                faults.append(new_fault)

            final_result = {
                    'new_img': final_img,
                    'success': True,
                    'direction': direction,
                    'linear': f'{results[4]}',
                    'rounded': f'{results[5]}',
                    'faulty': True,
                    'discontinuity': discontinuity,
                    'faults': faults,
                    }

            return final_result


