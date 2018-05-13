from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils.datasets import DataManager
from utils.data_augmentation import ImageGenerator
from utils.datasets import split_imdb_data
from models.Xceptions import mini_XCEPTION

# 参数设置
batch_size = 32 # 批量训练数据大小
epochs = 10000 # 训练轮数
input_shape = (64, 64, 1) # 图片矩阵
validation_split = .2 # 验证集大小
num_classes = 2 # 类数,男和女
patience = 100 # 信心值，用于后面的EarlyStopping等,在信心值个epochs过去后模型性能不再提升，就执行指定动作
dataset_name = 'imdb' # 数据集名称
do_random_crop = False
# 设为灰度图
if input_shape[2] == 1:
    grayscale = True
# 设置路径
images_path = '../datasets/imdb_crop/'
log_file_path = '../datasets/trained_models/gender_models/gender_training.log'
trained_models_path = '../datasets/trained_models/gender_models/gender_mini_XCEPTION'

# 加载数据
data_loader = DataManager(dataset_name)
ground_truth_data = data_loader.get_data()
train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)
print('Number of training samples:', len(train_keys))
print('Number of validation samples:', len(val_keys))
image_generator = ImageGenerator(ground_truth_data, batch_size,
    input_shape[:2], train_keys, val_keys, None,
    path_prefix=images_path, vertical_flip_probability=0,
    grayscale=grayscale, do_random_crop=do_random_crop)
    
# 模型参数
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
model.summary()

# 模型回调
early_stop = EarlyStopping(monitor='val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=int(patience/2), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(
    model_names, monitor='val_loss', verbose=1, 
    save_best_only=True, save_weights_only=False)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# 训练模型
model.fit_generator(image_generator.flow(mode='train'),
    steps_per_epoch=int(len(train_keys) / batch_size),
    epochs=epochs, verbose=1, callbacks=callbacks,
    validation_data=image_generator.flow('val'),
    validation_steps=int(len(val_keys) / batch_size))