import ast
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

class CleanData:
    def __init__(self, file_path='./final_labels.csv',
                 image_path = "./final_images/",
                 train_ratio = 0.70,validation_ratio=0.15,test_ratio=0.15):
        self.file_path = file_path
        self.image_path = image_path
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

    
    def read_file(self):
        file = pd.read_csv(self.file_path)
        return file 
    
    def remove_blank_labels(self):
        file = self.read_file()
        final_df = file[file["annotation_result"]!="[]"]
        final_df = final_df.reset_index(drop=True)
        final_df["annotation_result"] = final_df["annotation_result"].apply(lambda x:  ast.literal_eval(x))
        return final_df
    
    def create_label(self):
        final_df = self.remove_blank_labels()
        final_df["labels"] = final_df["annotation_result"].apply(lambda x:  ','.join(x))
        mlb = MultiLabelBinarizer()
        mlb_result = mlb.fit_transform([str(final_df.loc[i,'labels']).split(',') for i in range(len(final_df))])
        final_data = pd.concat([final_df['image'],final_df['annotation_result'],pd.DataFrame(mlb_result,columns=list(mlb.classes_))],axis=1)
        final_data['label'] = final_data.apply(lambda x: list([x["CLASS_A"],
                                              x['CLASS_B'],
                                              x['CLASS_C'],
                                              ]),axis=1)  
        return final_data

    def rename_image_name(self):
        final_data = self.create_label()
        final_data["image"] = final_data["image"].apply(lambda x:  x.split("/")[-1])
        final_data["image_id_new"] = final_data["image"].apply(lambda x: x.split('-')[0])
        final_data["image_id_new"] = final_data["image_id_new"].apply(lambda x: x.split('.')[0]+".jpg")
        final_data["image_id_new"] = final_data["image_id_new"].apply(lambda x:  self.image_path+x)
        return final_data
    
    def train_test_split(self):
        final_data = self.rename_image_name()
        x_train, x_test, y_train, y_test = train_test_split(final_data["image_id_new"], final_data["label"], test_size=1 - self.train_ratio)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=self.test_ratio/(self.test_ratio + self.validation_ratio)) 
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def final_labels(self):
        x_train, x_val, x_test, y_train, y_val, y_test = self.train_test_split()
        
        train_images = x_train.tolist()
        train_labels = y_train.tolist()

        val_images = x_val.tolist()
        val_labels = y_val.tolist()

        test_images = x_test.tolist()
        test_labels = y_test.tolist()
        
        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    
def read_img(image_path, image_w = 286, image_h = 286):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None,None,3])
    img = tf.image.resize(img, [image_w, image_h])
    img  = img/255.0
    img = tf.cast(img,tf.float32)
    return img

def load_data(image_path, label):
    image = read_img(image_path)
    return image, label