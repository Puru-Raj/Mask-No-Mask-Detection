import pandas as pd
import os
from PIL import Image
from numpy import asarray


class ImageClassifier():
    def __init__(self) -> None:
        self.directory1 = 'DataSet\without_mask'
        self.directory2 = 'DataSet\without_mask'
        self.dataset = []

    def process_image(self,dir,flag):
        for image_path in os.listdir(dir):
            temp = []
            input_path = os.path.join(dir, image_path)
            img = Image.open(input_path)
            img = img.resize((64,64), Image.ANTIALIAS)
            img.save(input_path)
            numpydata = asarray(img)
            numpydata = numpydata.ravel()
            for i in numpydata:
                temp.append(i)
            if flag == 1:
                temp.append(1)
            else:
                temp.append(0)
            self.dataset.append(temp)



if __name__ == "__main__":
    clf = ImageClassifier()
    clf.process_image(clf.directory1,1)
    clf.process_image(clf.directory2,0)
    df = pd.DataFrame(clf.dataset)
    df.to_csv('data.csv')