
import os
import tensorflow as tf

override_model = False
##change chosen model to deploy different models available in the TF2 object detection zoo
MODELS_CONFIG = {
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
        'batch_size': 8
    },
    'efficientdet-d1': {
        'model_name': 'efficientdet_d1_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d1_coco17_tpu-32.tar.gz',
        'batch_size': 8
    },
    'efficientdet-d2': {
        'model_name': 'efficientdet_d2_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d2_coco17_tpu-32.tar.gz',
        'batch_size': 8
    },
        'efficientdet-d3': {
        'model_name': 'efficientdet_d3_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d3_coco17_tpu-32.tar.gz',
        'batch_size': 8
    }
}

#in this tutorial we implement the lightweight, smallest state of the art efficientdet model
#if you want to scale up tot larger efficientdet models you will likely need more compute!
chosen_model = 'efficientdet-d0'

num_steps = 700 #The more steps, the longer the training. Increase if your loss function is still decreasing and validation metrics are increasing. 
num_eval_steps = 100 #Perform evaluation after so many steps

model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']
batch_size = MODELS_CONFIG[chosen_model]['batch_size'] #if you can fit a large batch in memory, it may speed up your training

#download pretrained weights
train_dir = '/home/workspace/train/'
deploy_dir = '/home/workspace/train/deploy/'
dataset_location = '/home/workspace/dataset'

pipeline_fname = deploy_dir + base_pipeline_file
fine_tune_checkpoint = deploy_dir + model_name + '/checkpoint/ckpt-0'
label_map_pbtxt_fname = dataset_location + "/Car-detection-1" + '/train/Cars_label_map.pbtxt'
test_record_fname = dataset_location + "/Car-detection-1" + '/test/Cars.tfrecord'
train_record_fname = dataset_location + "/Car-detection-1" + '/train/Cars.tfrecord'

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())
num_classes = get_num_classes(label_map_pbtxt_fname)

if override_model:
    import re

    print('writing custom configuration file')

    with open(pipeline_fname) as f:
        s = f.read()
    with open('pipeline_file.config', 'w') as f:

        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                   'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                   'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                   'num_steps: {}'.format(num_steps), s)

        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                   'num_classes: {}'.format(num_classes), s)

        #fine-tune checkpoint type
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)

        f.write(s)

# os.system(f"cat {deploy_dir}/pipeline_file.config")
pipeline_file = deploy_dir + '/pipeline_file.config'
model_dir = train_dir + '/model'

# os.system("export TF_GPU_ALLOCATOR=cuda_malloc_async")
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True) 

os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/local/cuda-12/"

os.system(f"python /home/workspace/notebook/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_file} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}")