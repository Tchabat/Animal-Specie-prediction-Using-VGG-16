import os

# Check required dependencies
try:
    import tensorflow
    import scipy
    from tensorflow.keras.applications import VGG16 # type: ignore
    from tensorflow.keras.models import Model # type: ignore
    from tensorflow.keras.layers import Dense, Flatten # type: ignore
    from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
    from tensorflow.keras.optimizers import Adam # type: ignore
except ImportError as e:
    print(f"Missing required dependency: {str(e)}")
    print("Please install required packages using:")
    print("pip install tensorflow scipy")
    exit(1)

# Suppress TensorFlow warnings about oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), 'data')

# Check if data directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found at {data_dir}. Please ensure your dataset is in the correct location.")

# Initialize model
base_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

try:
    train = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        subset='training'
    )
    val = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        subset='validation'
    )
    
    print(f"Found {len(train.classes)} training samples")
    print(f"Found {len(val.classes)} validation samples")
    
    # Training with additional error handling
    try:
        history = model.fit(
            train,
            validation_data=val,
            epochs=5,
            verbose=1
        )
    except Exception as train_error:
        print(f"Training error: {str(train_error)}")
        raise
    
    # Save the model
    model_save_path = os.path.join(current_dir, 'animal_classifier.h5')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
except Exception as e:
    print(f"An error occurred during data preparation or training: {str(e)}")
    raise
