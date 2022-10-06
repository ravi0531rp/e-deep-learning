from architectures.model_selector import selector

from utils.data_utils import CleanTrainingData, CleanTestData, analyse
from utils.data_gen import data_generator
from utils.confusion_matrix import prediction, plot_confusion_matrix
from utils.lr_scheduler import WarmupExponentialDecay

import tensorflow_addons as tfa
import tensorflow as tf

from loguru import logger
import argparse

import pandas as pd

from datetime import datetime

lst = ["Discoloration",
        "Exposed_Deck",
        "Exposed_Felt",
        "Holes",
        "Missing_Shingle",
        "Good",
        "Streaking",
        "Tarps",
        "Under_Construction_Repair"]

class_dict = {str(i): lst[i] for i in range(9)}

print(class_dict)

def get_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("-a", "--architecture", type=str, default="eff_v1_b0")
    my_parser.add_argument("-e", "--epochs", type=int, default=30)
    my_parser.add_argument("-s", "--save_model_prefix", type=str, required=True)
    my_parser.add_argument("-b", "--batch_size", type=int, default=64)
    my_parser.add_argument("-lr", "--learning_rate", type=float, default=0.002)
    my_parser.add_argument(
        "-l",
        "--labels_path",
        type=str,
        default="./datasets/csv_files/masked_43930_geohash.csv",
    )
    my_parser.add_argument(
        "-vl",
        "--valid_path",
        type=str,
        default="./datasets/csv_files/masked_43930_valid_geohash.csv",
    )
    my_parser.add_argument(
        "-i",
        "--images_path",
        type=str,
        default="./datasets/images/masked_43930_geohash",
    )

    my_parser.add_argument(
        "-bl",
        "--bench_labels_path",
        type=str,
        default="./datasets/csv_files/masked_43930_geohash.csv",
    )
    my_parser.add_argument(
        "-bi",
        "--bench_images_path",
        type=str,
        default="./datasets/images/masked_43930_geohash",
    )

    my_parser.add_argument("-iw", "--image_w", type=float, default=300)
    my_parser.add_argument("-ih", "--image_h", type=int, default=300)
    my_parser.add_argument("-t", "--threshold", type=float, default=0.5)
    my_parser.add_argument("-n", "--num_classes", type=int, default=9)
    my_parser.add_argument(
        "-c",
        "--classes",
        type=list,
        default=[
            "Discoloration",
            "Exposed_Deck",
            "Exposed_Felt",
            "Holes",
            "Missing_Shingle",
            "None",
            "Streaking",
            "Tarps",
            "Under_Construction_Repair",
        ],
    )

    my_parser.add_argument(
        "-m", "--mode", type=str, choices=["train", "eval"], default="eval"
    )

    args = vars(my_parser.parse_args())

    arch = args["architecture"]
    epochs = args["epochs"]
    save_model_prefix = args["save_model_prefix"]
    batch_size = args["batch_size"]
    labels_path = args["labels_path"]
    images_path = args["images_path"]
    threshold = args["threshold"]
    num_classes = args["num_classes"]
    classes = args["classes"]
    mode = args["mode"]
    image_w = args["image_w"]
    image_h = args["image_h"]
    bench_images_path = args["bench_images_path"]
    bench_labels_path = args["bench_labels_path"]
    learning_rate = args["learning_rate"]
    valid_path = args["valid_path"]
    params = {
        "epochs": epochs,
        "batch_size": int(batch_size),
        "labels_path": labels_path,
        "images_path": images_path,
        "image_w": int(image_w),
        "image_h": int(image_h),
        "threshold": threshold,
        "num_classes": int(num_classes),
        "classes": classes,
        "arch": arch,
        "mode": mode,
        "save_model_prefix": save_model_prefix,
        "bench_labels_path": bench_labels_path,
        "bench_images_path": bench_images_path,
        "learning_rate": learning_rate,
        "valid_path": valid_path
    }
    logger.add(f"./logs/{params['mode']}_full_{datetime.today().strftime('%b-%d-%Y')}.log", level="DEBUG", rotation="100 MB")
    logger.info(params)
    return params


def event():
    params = get_args()
    if params["num_classes"] != len(params["classes"]):
        logger.error(f"Number of Classes doesn't match the length of the classes list.")
        return

    if params["batch_size"] <= 0:
        logger.error(f"Batch Size can't be less than or equal to zero")
        return

    if params["epochs"] <= 0:
        logger.error(f"Number of Epochs can't be less than or equal to zero")
        return

    lr = params["learning_rate"]
    
    tf.keras.backend.clear_session()
    logger.info(f"Current base learning rate is {lr}")
    model_skeleton = selector()[params["arch"]](image_shape = (params["image_w"] , params["image_h"],3) , num_classes = params["num_classes"])
    model, preprocess_func , _= model_skeleton.create_base()

    lrate = WarmupExponentialDecay(lr_base=lr)

    save_model = f"{params['save_model_prefix']}_{params['arch']}_{params['batch_size']}_{str(lr).replace('.','_')}.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            save_model,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
            save_freq="epoch",
            mode="min",
            period=1,
        ),
        # tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=7),
        lrate  
    ]

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(), 
        metrics=[
            tfa.metrics.F1Score(
                num_classes=params["num_classes"],
                average="macro",
                threshold=params["threshold"],
            )
        ],
    )
    ctrd = CleanTrainingData(
        params["labels_path"], 
        params["images_path"],
        params["valid_path"]
        )

    (
        train_images,
        train_labels,
        val_images,
        val_labels,
        mlb
    ) = ctrd.final_labels()

    
    logger.info(f"Creating the data generators..")

    train_dataset = data_generator(train_images, train_labels, preprocess_func, params["image_w"] , params["image_h"],  params["batch_size"])
    val_dataset = data_generator(val_images, val_labels, preprocess_func, params["image_w"] , params["image_h"], params["batch_size"])
    
    logger.success(f"Data Generators created!!")

    if params["mode"] == "train":
        logger.debug("Starting the Training process")
        model.fit(
            train_dataset,
            epochs=params["epochs"],
            steps_per_epoch=len(train_images) // params["batch_size"],
            validation_data=val_dataset,
            batch_size=params["batch_size"],
            validation_steps=len(val_images) // params["batch_size"],
            callbacks=callbacks,
        )
        logger.success("Training Complete..")

    if params["mode"] == "eval":
        ctsd = CleanTestData(
            params["bench_labels_path"],
            params["bench_images_path"],
            mlb
        )
        (
            test_images,
            test_labels
        ) = ctsd.final_labels()

        test_dataset = data_generator(
            test_images, test_labels, preprocess_func, params["image_w"] , params["image_h"], params["batch_size"], train=False
        )

        logger.info("Loading Model Weights..")
        model.load_weights(save_model)
        logger.success("Loading Complete..")

        logger.info("Generating Predictions..")
        y_pred = prediction(model, test_dataset, params["threshold"])
        logger.success("Predictions Generated..")
        clf_report_disp , clf_report = plot_confusion_matrix(test_labels, y_pred, params["classes"])
        final_dict = {}
        for k,v in clf_report.items():
            if k in class_dict.keys():
                final_dict[class_dict[k]] = v
            else:
                final_dict[k] = v
        logger.success(clf_report_disp)
        df = pd.DataFrame(final_dict).transpose()
        doc_name = f"./retraining/docs/{params['save_model_prefix'].split('/')[-1]}_{params['arch']}_{params['batch_size']}_{str(lr).replace('.','_')}.csv"
        df.to_csv(doc_name)

if __name__ == "__main__":
    event()
