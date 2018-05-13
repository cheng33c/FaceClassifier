from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from models.CNN import mini_XCEPTION
from utils.datasets import DataManager, split_data
from utils.preprocessor import preprocess_input

# 参数设置
batch_size = 32 # 批量训练数据大小
epochs = 10000 # 训练轮数
input_shape = (64, 64, 1) # 图片矩阵
validation_split = .2 # 验证集大小
num_classes = 7 # 类数
patience = 50 # 信心值，用于后面的EarlyStopping等,在信心值个epochs过去后模型性能不再提升，就执行指定动作
base_path = '../datasets/trained_models/emotion_models/'

# 数据生成
data_generator = ImageDataGenerator(
    featurewise_center=False, # 使输入数据集去中心化（均值为0）, 按feature执行
    featurewise_std_normalization=False, # 将输入除以数据集的标准差以完成标准化, 按feature执行
    rotation_range=10, # 数据提升时图片随机转动的角度
    width_shift_range=0.1, # 图片宽度的某个比例，数据提升时图片水平偏移的幅度
    height_shift_range=0.1, # 图片高度的某个比例，数据提升时图片竖直偏移的幅度
    zoom_range=.1, # 浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    horizontal_flip=True, # 进行随机水平翻转
)

# 模型参数/编译
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # 回调
    log_file_path = base_path + dataset_name + '_emotion_training.log' # 构造log文件名
    csv_logger = CSVLogger(log_file_path, append=False) # 将epochs训练结果保存为csv
    early_stop = EarlyStopping('val_loss', patience=patience) # 当patience个epoch过去而模型性能不提升时，停止训练
    reduce_lr = ReduceLROnPlateau( # 当评价指标不在提升时，减少学习率.当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。
        monitor='val_loss', factor=0.1, # 监视val_loss,每次减少学习率的因子为0.1
        patience=int(patience/4), verbose=1, # 该回调函数检测指标的情况，如果在patience个epoch中看不到模型性能提升，则减少学习率, verbose信息展示模式=1
    ) 
    trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(
        model_names, monitor='val_loss', verbose=1, save_best_only=True, # 只保存性能最好的模型
    )
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # 加载数据集
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data
    model.fit_generator(
        data_generator.flow(train_faces, train_emotions, batch_size),
        steps_per_epoch=len(train_faces) / batch_size,
        epochs=epochs, verbose=1, callbacks=callbacks, validation_data=val_data
    )