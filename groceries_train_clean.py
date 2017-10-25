from load_data import *
from data_generators import *
import copy
from models_finetune_features import *
from keras.utils import to_categorical
import argparse


parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--min_side', default=256, type=int,  help='Shortest Side Resize Length')
parser.add_argument('--im_size', default=224, type=int,  help='Shortest Side Resize Length')
parser.add_argument('--batch_size', default=32, type=int,  help='Batch Size')
parser.add_argument('--seed', default=1, type=int,  help='seed')
parser.add_argument('--resume', default=False, action='store_true', help='resume')
parser.add_argument('--test',  default=False, action='store_true', help='test')
parser.add_argument('--epochs', default = 80, type = int,   help='epochs')
parser.add_argument('--val_percent', default = 0.1, type = float,   help='Validation Percentage')
parser.add_argument('--model_name', default = 'DenseNet169', type = str,   help='Model Name [ DenseNet169, DenseNet161, VGG16, Xception, Resnet50, Resnet101 ]')

parser.set_defaults(argument=True)
args = parser.parse_args()

INPUT_ROWS = args.im_size
INPUT_COLS = args.im_size
MIN_SIDE = args.min_side
BATCH_SIZE = args.batch_size
SEED = args.seed
VALIDATION_PERCENTAGE = args.val_percent
MODEL_NAME = args.model_name
EPOCHS= args.epochs
SUFFIX = MODEL_NAME
SAVENAME = './Checkpoints/' + MODEL_NAME + '_' + str(SEED)
FILENAME = MODEL_NAME + '_' + str(SEED)

TEST_BATCH_SIZE=64
np.random.seed(SEED)

x_train, x_val,  x_test, y_train, y_val, fwd_key, rev_key = load_data( seed=SEED,  validation_percentage=VALIDATION_PERCENTAGE, min_side=MIN_SIDE  ) 

train_generator, train_generator_feats, val_generator, test_generator, test_generator_rz = get_data_generators( x_train, y_train, x_val, y_val, x_test,  batch_size=BATCH_SIZE, seed=SEED, im_shape=(INPUT_ROWS,INPUT_COLS), test_batch_size=TEST_BATCH_SIZE )

#model =createModel( 'DenseNet169' )
model = None
if not args.test and not args.resume:
    model =createModel( MODEL_NAME )

if not args.test and args.resume:
    model = load_trained_model( MODEL_NAME, SAVENAME )
if not args.test:
    history = fit_model(model, train_generator, val_generator, train_steps=len(y_train)//BATCH_SIZE + 1, val_steps=len(val_generator)//TEST_BATCH_SIZE + 1, epochs=EPOCHS, savename=SAVENAME)

if model is not None:
    del model
model_restored = load_trained_model( MODEL_NAME, SAVENAME )

get_validation_metrics( model_restored, val_generator,  TEST_BATCH_SIZE, FILENAME )
generate_test_submissions( model_restored, test_generator, test_generator_rz, rev_key, TEST_BATCH_SIZE, FILENAME  )
get_features( MODEL_NAME, model_restored,  train_generator_feats,  val_generator,  test_generator, test_generator_rz, TEST_BATCH_SIZE, FILENAME )

