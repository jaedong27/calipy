import os
import numpy as np
import json

class Transform():
    def __init__(self, path = None):
        self.R = np.eye(3)
        self.T = np.zeros((3,1))
        if path is not None:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if "R" in data:
                    self.R = np.array(data["R"])
                    self.T = np.array(data["T"])
                elif "rotation" in data:
                    self.R = np.reshape(np.array(data["rotation"]),(3,3))
                    self.T = np.reshape(np.array(data["translation"]),(3,1))
                else:
                    print("RT Parameter is wrong", path)
            else: 
                print("RT File does not exist :", path)

        self.R_inv = np.linalg.inv(self.R)
        self.T_inv = -np.dot(self.R_inv, self.T)
        return

    def __str__(self):
        string = "rotation : \n"
        string += self.R.__str__() + "\n"
        string += "translation : \n"
        string += self.T.__str__() + "\n"
        return string

    def setParam(self, R, T):
        self.R = R
        self.T = T
        self.R_inv = np.linalg.inv(self.R)
        self.T_inv = -np.dot(self.R_inv, self.T)

    def translate(self, pointcloud): # (3, -)
        return np.dot(self.R, pointcloud) + self.T

    def move(self, pointcloud):
        return np.dot(self.R, pointcloud) + self.T

    def dot(self, transform):
        output = Transform()
        output.R = np.dot(self.R, transform.R)
        output.T = np.dot(self.R, transform.T) + self.T
        output.R_inv = np.linalg.inv(output.R)
        output.T_inv = -np.dot(output.R_inv, output.T)
        return output

    def inv(self):
        output = Transform()
        output.R = self.R_inv
        output.T = self.T_inv
        output.R_inv = self.R
        output.T_inv = self.T
        return output

    def saveJson(self, path):
        data = {}
        data["rotation"] = self.R.tolist()
        data["translation"] = self.T.tolist()

        with open(path, 'w') as outfile:
            json.dump(data, outfile, indent=4, sort_keys=True)

if __name__=="__main__":
    po = Transform("rs_to_window_result.json")
    print(po.R)
    print(po.T)