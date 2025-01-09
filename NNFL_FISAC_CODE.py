import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

import itertools



#load dataset
data = np.load(r'G:\NNFL\Fisac\8TH_NOV_NNFL_FISAC\CODE\ORL_faces\ORL_faces.npz')

 

# load the "Train Images"
x_train = data['trainX']
#normalize every image
x_train = np.array(x_train,dtype='float32')/255

x_test = data['testX']
x_test = np.array(x_test,dtype='float32')/255

# load the Label of Images
y_train= data['trainY']
y_test= data['testY']


# show the train and test Data format
print('x_train : {}'.format(x_train[:]))
print('Y-train shape: {}'.format(y_train))
print('x_test shape: {}'.format(x_test.shape))



x_train, x_valid, y_train, y_valid= train_test_split(
    x_train, y_train, test_size=.05, random_state=1234,)


im_rows=112
im_cols=92
batch_size=512
im_shape=(im_rows, im_cols, 1)

#change the size of images
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_valid = x_valid.reshape(x_valid.shape[0], *im_shape)

print('x_train shape: {}'.format(y_train.shape[0]))
print('x_test shape: {}'.format(y_test.shape))


cnn_model = Sequential([
    Conv2D(filters=36, kernel_size=7, activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=54, kernel_size=5, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(2024, activation='relu'),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    # 20 is the number of output classes
    Dense(20, activation='softmax')
])

cnn_model.compile(
    loss='sparse_categorical_crossentropy',  # 'categorical_crossentropy' for one-hot labels
    optimizer=Adam(learning_rate=0.0001),  # Use learning_rate instead of lr
    metrics=['accuracy']
)


cnn_model.summary()

history=cnn_model.fit(
    np.array(x_train), np.array(y_train), batch_size=512,
    epochs=250, verbose=2,
    validation_data=(np.array(x_valid),np.array(y_valid)),
)
# Save the trained model
cnn_model.save('face_recognition_model.h5')



#create a dataframe of the model training history
results = pd.DataFrame(history.history)

# Print the entire dataframe
print(results)



from keras.models import load_model
from keras.preprocessing import image
from tkinter import Tk, Button, filedialog, Label, Canvas, Frame
import pandas as pd
from PIL import Image, ImageTk
from datetime import datetime
import os
import numpy as np

# Load the model
classifier = load_model('face_recognition_model.h5')

# Define ResultMap with placeholder names (Update with actual names in your dataset)
ResultMap = {
    0: "Alice",
    1: "Bob",
    2: "Charlie",
    3: "David",
    4: "Eva",
    5: "Frank",
    6: "Grace",
    7: "Hank",
    8: "Ivy",
    9: "Jack",
    10: "Karen",
    11: "Leo",
    12: "Mona",
    13: "Nina",
    14: "Oscar",
    15: "Paul",
    16: "Quincy",
    17: "Rita",
    18: "Steve",
    19: "Tina"
}

# Define a function to load an image from a file path
def load_image_from_path(file_path):
    test_image = image.load_img(file_path, target_size=(112, 92), color_mode='grayscale')
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0  # Normalize as done during training
    test_image = np.expand_dims(test_image, axis=0)
    return test_image

# Define the function that runs the prediction
def make_prediction(test_image):
    result = classifier.predict(test_image, verbose=0)
    predicted_index = np.argmax(result)
    prediction_text = f'Prediction is: {ResultMap[predicted_index]} - Attendance Marked'
    print(prediction_text)
    save_to_excel(ResultMap[predicted_index])
    return prediction_text

# Function to save attendance to Excel
def save_to_excel(prediction_name):
    file_name = 'attendance.xlsx'
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {'Name': [prediction_name], 'Date_Time': [current_time]}
    df = pd.DataFrame(data)

    if os.path.exists(file_name):
        existing_df = pd.read_excel(file_name)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel(file_name, index=False)
    else:
        df.to_excel(file_name, index=False)

# Set up the GUI
def create_gui():
    # Create the main window
    root = Tk()
    root.title("Attendance Interface")
    root.geometry("500x600")
    root.configure(bg='#f0f0f0')

    # Create a main frame
    main_frame = Frame(root, bg='#f0f0f0')
    main_frame.pack(pady=20, padx=20, fill='both', expand=True)

    # Header label
    header_label = Label(main_frame, text="Face Recognition Attendance System", font=('Helvetica', 16, 'bold'), bg='#f0f0f0', fg='#333')
    header_label.pack(pady=10)

    # Label to display prediction result
    result_label = Label(main_frame, text="", font=('Helvetica', 14), bg='#f0f0f0')
    result_label.pack(pady=10)

    # Canvas to display the selected image
    image_canvas = Canvas(main_frame, width=300, height=300, bg='white', relief='sunken', bd=2)
    image_canvas.pack(pady=10)

    # Define the function to be called when the "Browse" button is clicked
    def browse_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            # Load and display the image for confirmation
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            # To avoid early garbage collection, assign the image to the canvas itself
            image_canvas.create_image(0, 0, anchor='nw', image=img_tk)
            image_canvas.image = img_tk

            # Proceed with prediction
            test_image = load_image_from_path(file_path)
            prediction = make_prediction(test_image)
            result_label.config(text=prediction)

    # Create a button to browse for an image
    browse_button = Button(main_frame, text="Browse Image", command=browse_file, font=('Helvetica', 12), bg='#4CAF50', fg='white', padx=20, pady=10, relief='raised', bd=3)
    browse_button.pack(pady=20)

    # Run the GUI loop
    root.mainloop()

# Example usage to run the GUI
create_gui()



scor = cnn_model.evaluate( np.array(x_test),  np.array(y_test), verbose=0)

print('test los {:.4f}'.format(scor[0]))
print('test acc {:.4f}'.format(scor[1]))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Get predictions
predicted = np.array(cnn_model.predict(x_test))
ynew = np.argmax(predicted, axis=1)

# Calculate accuracy
Acc = accuracy_score(y_test, ynew)
print("accuracy : ")
print(Acc)

# Confusion Matrix
cnf_matrix = confusion_matrix(np.array(y_test), ynew)

# Optional: convert to categorical (if needed)
from keras.utils import to_categorical

y_test1 = to_categorical(y_test, 20)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Print and plot confusion matrix
print('Confusion matrix, without normalization')
print(cnf_matrix)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=list(range(20)), title='Confusion matrix, without normalization')

# Print classification report
print("Confusion matrix:\n%s" % confusion_matrix(np.array(y_test), ynew))
print(classification_report(np.array(y_test), ynew))