import seaborn as sns
sns.set()
from .utils import *

#git clone https://github.com/baontn/Zalo-AI-Challenge.git


meta = pd.read_csv('Zalo-AI-Challenge/train-zaloAI/train_meta.csv')
print("Display the head of the dataset(first 15 rows): \n",meta.head(15))
print(" \n Display the number of null cells in the dataset: \n", meta.isnull().sum())
print("\n Describe the dataset: \n", meta.describe())



df = meta.copy()
df.head(20)
X_train, X_test, y_train, y_test = get_xy(meta)


# Create model from Inception V1
inceptionv3 = 'https://tfhub.dev/google/imagenet/inception_v3/classification/5'

efficientnet_v2_imgnet1k_b0 = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2'

efficientnet_v2_imgnet21k_b3 = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2'
# Create model
model = create_model(inceptionv3, num_classes=2)
#If you want to run my custom CNN model, run the code below:
#model = built_model(X_train)
# Compile and fit
model.compile(loss='binary_crossentropy',
                     optimizer=tf.keras.optimizers.SGD(),
                     metrics=['accuracy', f1_m])
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=10, callbacks=[tensorboard_hub('tensorhub', 'custom_cnn')])



batch_size =32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(X_train, y_train, batch_size)
steps_per_epoch = X_train.shape[0] // batch_size
r1 = model.fit(train_generator, validation_data=(X_test, y_test), steps_per_epoch=steps_per_epoch, epochs=10, callbacks=[tensorboard_hub('tensorhub', 'custom_cnn augmentation')])

compare_hist(r, r1, 10)

model.evaluate(X_test, y_test)
model.save('inceptionv3.h5')

test_dir = 'Zalo-AI-Challenge/'
x_pred, pub_test = get_Xpred(test_dir)
#y_pred = get_predict(x_pred, model)
create_result(x_pred, model, pub_test)


